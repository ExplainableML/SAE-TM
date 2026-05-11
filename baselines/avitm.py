#!/usr/bin/env python3
"""
Reference-style ProdLDA (Srivastava & Sutton, 2017) with modern PyTorch, but
**keeping the original reference model architecture & hyperparameters** while
adopting the **data loading and CLI** from the user's AVITM/ProdLDA script:

- Data: JSONL with lines of {"id": str, "document": str}
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV/top-N like the user's script

Key choices to stay close to the reference implementation:
- Encoder: two Softplus layers (en1 -> en2), Dropout(0.2)
- Heads: mean/logvar with BatchNorm1d on each head
- Latent sample z ~ N(mean, diag(var)); **spherical Gaussian prior** with
  variance hyperparam (default 0.995), mean 0 (buffers)
- Mixture p = softmax(z); Dropout(0.2) on p
- Decoder: Linear(num_topic -> vocab_size) + BatchNorm1d on output
- Reconstruction uses explicit softmax over vocab as in the reference
- KL to spherical Gaussian prior as in the reference
- Optional decoder weight init uniform_(0, init_mult)

Modernizations for correctness/stability:
- No use of deprecated Variable/.data; uses torch.randn_like and `dim` args
- Device management with `.to(device)`
- Log/IO, vocab building, JSONL IO from the user's script
"""
from __future__ import annotations
import argparse
import collections
import dataclasses
import io
import json
import math
import os
import random
import re
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Import shared utilities
from utils.dataloading import (
    download_nltk_data,
    read_jsonl,
    write_jsonl,
    Vocab,
    BowDataset,
)
from utils.utils import set_seed, export_topics as export_topics_generic


# -----------------------------
# Reference-style ProdLDA model (modernized I/O)
# -----------------------------
class ProdLDA_Ref(nn.Module):
    """Closest faithful reproduction of the reference model, in modern PyTorch.
    - Encoder: Linear->Softplus, Linear->Softplus + Dropout(0.2)
    - Heads: mean/logvar with BN (scale removed like reference)
    - Latent: z ~ N(mean, diag(var))
    - p = softmax(z) with Dropout(0.2)
    - Decoder: Linear(K->V) + BN (scale removed like reference)
    - Recon: softmax over vocab (prob-space), like the reference
    - Prior: spherical Gaussian N(0, variance*I)
    """
    def __init__(self, num_input: int, num_topic: int,
                 en1_units: int=100, en2_units: int=100,
                 variance: float=0.995, init_mult: float=1.0):
        super().__init__()
        self.num_input = num_input
        self.num_topic = num_topic
        self.en1_units = en1_units
        self.en2_units = en2_units
        self.variance = variance
        self.init_mult = init_mult

        # Encoder
        self.en1_fc = nn.Linear(num_input, en1_units)
        self.en2_fc = nn.Linear(en1_units, en2_units)
        self.en2_drop = nn.Dropout(0.2)

        self.mean_fc = nn.Linear(en2_units, num_topic)
        self.logvar_fc = nn.Linear(en2_units, num_topic)
        self.mean_bn = nn.BatchNorm1d(num_topic, affine=False)
        self.logvar_bn = nn.BatchNorm1d(num_topic, affine=False)

        # Latent dropout on p
        self.p_drop = nn.Dropout(0.2)

        # Decoder
        self.decoder = nn.Linear(num_topic, num_input)
        self.decoder_bn = nn.BatchNorm1d(num_input, affine=False)

        # Prior buffers
        prior_mean = torch.zeros(1, num_topic)
        prior_var = torch.full((1, num_topic), float(variance))
        prior_logvar = torch.log(prior_var)
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # Init decoder weight like reference (uniform 0..init_mult)
        if init_mult != 0:
            with torch.no_grad():
                self.decoder.weight.uniform_(0.0, init_mult)

        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reference uses softplus in encoder layers
        en1 = F.softplus(self.en1_fc(x))
        en2 = F.softplus(self.en2_fc(en1))
        en2 = self.en2_drop(en2)
        mean = self.mean_bn(self.mean_fc(en2))
        logvar = self.logvar_bn(self.logvar_fc(en2))
        return mean, logvar

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(x)
        var = torch.exp(logvar)
        # Reparameterization
        eps = torch.randn_like(mean)
        z = mean + torch.sqrt(var) * eps
        # Mixture over topics
        p = F.softmax(z, dim=-1)
        p = self.p_drop(p)
        # Decoder: probability over vocab
        logits = self.decoder(p)
        probs = F.softmax(self.decoder_bn(logits), dim=-1)
        return {
            'probs': probs,            # p(w|doc)
            'mean': mean,
            'logvar': logvar,
            'var': var,
            'z': z,
            'p': p,
        }

    def loss(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = out['probs']
        mean = out['mean']
        logvar = out['logvar']
        var = out['var']
        # Negative log-likelihood in prob space (as reference)
        nl = -(x * torch.log(probs + 1e-10)).sum(dim=1)
        # KL to spherical Gaussian prior
        prior_mean = self.prior_mean.expand_as(mean)
        prior_var = self.prior_var.expand_as(mean)
        prior_logvar = self.prior_logvar.expand_as(mean)
        var_div = var / prior_var
        diff = mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_div = prior_logvar - logvar
        kld = 0.5 * ((var_div + diff_term + logvar_div).sum(dim=1) - self.num_topic)
        loss = nl + kld
        return loss.mean()

    def topic_word_matrix(self) -> torch.Tensor:
        """Return raw decoder weights transposed to (V x K) for inspection.
        This mirrors the reference printing, but downstream exports can normalize.
        """
        return self.decoder.weight.t()  # (K x V) -> (V x K) via .t() usage at call site


# -----------------------------
# Training / Inference (user-style API, reference hyperparams preserved)
# -----------------------------
@dataclasses.dataclass
class TrainConfig:
    train_jsonl: str
    checkpoint: str
    output_path: str
    # Vocab
    vocab_size: int = 20000
    min_df: int = 5
    bow_cache_dir: Optional[str] = None
    # Model (reference hyperparams)
    num_topic: int = 50
    en1_units: int = 100
    en2_units: int = 100
    variance: float = 0.995
    init_mult: float = 1.0
    # Optimization (reference-style)
    optimizer: str = 'Adam'  # or 'SGD'
    learning_rate: float = 0.002
    momentum: float = 0.99    # Adam beta1 or SGD momentum
    batch_size: int = 200
    num_epoch: int = 80
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 13
    log_every: int = 100


@dataclasses.dataclass
class InferConfig:
    checkpoint: str
    input_jsonl: str
    output_path: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample: bool = True  # default behavior mirrors reference (sample z)


@dataclasses.dataclass
class Checkpoint:
    model_state: Dict
    vocab: Dict
    config: Dict





# -----------------------------
# Train
# -----------------------------

def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # Check for cached dataset and vocab
    if cfg.bow_cache_dir:
        os.makedirs(cfg.bow_cache_dir, exist_ok=True)
        cache_path = os.path.join(cfg.bow_cache_dir, 
                                   f"{cfg.vocab_size}_{cfg.min_df}.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached BowDataset and Vocab from {cache_path}")
            dataset, vocab = BowDataset.load(cache_path)
        else:
            print(f"Building vocab and creating BowDataset, caching to {cache_path}")
            rows = read_jsonl(cfg.train_jsonl)
            vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df)
            dataset = BowDataset(rows, vocab)
            dataset.save(cache_path, vocab)
    else:
        rows = read_jsonl(cfg.train_jsonl)
        vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df)
        dataset = BowDataset(rows, vocab)
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    V = len(vocab.id2token)
    K = cfg.num_topic

    model = ProdLDA_Ref(num_input=V, num_topic=K,
                        en1_units=cfg.en1_units, en2_units=cfg.en2_units,
                        variance=cfg.variance, init_mult=cfg.init_mult)
    model.to(cfg.device)

    # Optimizer (reference-style choice)
    if cfg.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.momentum, 0.999))
    elif cfg.optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    global_step = 0
    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.num_epoch}")
        epoch_loss = 0.0
        for batch in pbar:
            x = batch.to(cfg.device)
            out = model(x)
            loss = model.loss(x, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            epoch_loss += loss.item()
            if global_step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # Optionally print epoch average every 5 epochs, like reference
        if epoch % 5 == 0:
            avg = epoch_loss / max(len(loader), 1)
            print(f"Epoch {epoch}, loss={avg:.4f}")

    # Save checkpoint
    ckpt = {
        'model_state': model.state_dict(),
        'vocab': vocab.save(),
        'config': dataclasses.asdict(cfg),
    }
    os.makedirs(os.path.dirname(cfg.checkpoint) or '.', exist_ok=True)
    torch.save(ckpt, cfg.checkpoint)
    
    # --- Run inference on training data ---
    print("Running inference on training data...")
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_theta = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='infer'):
            x = batch.to(cfg.device)
            mean, logvar = model.encode(x)
            z = mean  # Use posterior mean for consistency
            p = F.softmax(z, dim=-1)
            all_theta.append(p.cpu())
        
        theta_matrix = torch.cat(all_theta, dim=0)
        topics_matrix = model.decoder.weight.t()
    
    os.makedirs(cfg.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(cfg.output_path, 'theta.pt'))
    torch.save(topics_matrix.cpu(), os.path.join(cfg.output_path, 'topics.pt'))
    print(f"Saved inference results to {cfg.output_path}")

    # Save top 100 words for each topic
    top_words = []
    K = topics_matrix.shape[0]
    for k in range(K):
        top_words_k = topics_matrix[k].topk(100).indices.tolist()
        top_words_k = [vocab.id2token[i] for i in top_words_k]
        top_words.append(top_words_k)
    
    with open(os.path.join(cfg.output_path, 'top_words.txt'), 'w') as f:
        for k in range(K):
            f.write(','.join(top_words[k]) + '\n')


# -----------------------------
# Infer
# -----------------------------

def infer(cfg: InferConfig):
    ckpt = torch.load(cfg.checkpoint, map_location=cfg.device)
    vocab = Vocab.load(ckpt['vocab'])
    mcfg = ckpt['config']

    V = len(vocab.id2token)
    K = int(mcfg['num_topic'])
    model = ProdLDA_Ref(num_input=V, num_topic=K,
                        en1_units=int(mcfg['en1_units']), en2_units=int(mcfg['en2_units']),
                        variance=float(mcfg['variance']), init_mult=float(mcfg['init_mult']))
    model.load_state_dict(ckpt['model_state'])
    model.to(cfg.device)
    model.eval()

    rows = read_jsonl(cfg.input_jsonl)
    ds = BowDataset(rows, vocab)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_theta = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='infer'):
            x = batch.to(cfg.device)
            mean, logvar = model.encode(x)
            if cfg.sample:
                var = torch.exp(logvar)
                z = mean + torch.sqrt(var) * torch.randn_like(mean)
            else:
                z = mean  # posterior mean (non-reference but useful)
            p = F.softmax(z, dim=-1)  # proportions over topics (N x K)
            all_theta.append(p.cpu())
        
        # Concatenate all batches
        theta_matrix = torch.cat(all_theta, dim=0)  # (N x K)
        
        # Get topic-word matrix (K x V) - decoder weight is (V x K), need transpose
        topics_matrix = model.decoder.weight.t()  # (K x V)
    
    # Save outputs
    os.makedirs(cfg.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(cfg.output_path, 'theta.pt'))
    torch.save(topics_matrix.cpu(), os.path.join(cfg.output_path, 'topics.pt'))

    # Save top 100 words for each topic
    top_words = []
    for k in range(K):
        top_words_k = topics_matrix[k].topk(100).indices.tolist()
        top_words_k = [vocab.id2token[i] for i in top_words_k]
        top_words.append(top_words_k)
    
    with open(os.path.join(cfg.output_path, 'top_words.txt'), 'w') as f:
        for k in range(K):
            f.write(','.join(top_words[k]) + '\n')


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Reference-style ProdLDA with user's CLI/Data")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train a topic model (reference-style model)')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # Reference hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--en1-units', type=int, default=100)
    pt.add_argument('--en2-units', type=int, default=100)
    pt.add_argument('--variance', type=float, default=0.995)
    pt.add_argument('--init-mult', type=float, default=1.0)
    # Optimization (reference style)
    pt.add_argument('--optimizer', choices=['Adam','SGD'], default='Adam')
    pt.add_argument('--learning-rate', type=float, default=0.002)
    pt.add_argument('--momentum', type=float, default=0.99)
    pt.add_argument('--batch-size', type=int, default=200)
    pt.add_argument('--epochs', type=int, default=80)
    # Misc
    pt.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pt.add_argument('--seed', type=int, default=13)
    pt.add_argument('--log-every', type=int, default=100)

    # Inference
    pi = sub.add_parser('infer', help='Run inference for documents with a saved checkpoint')
    pi.add_argument('--checkpoint', required=True)
    pi.add_argument('--input_jsonl', required=True)
    pi.add_argument('--output_path', required=True)
    pi.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pi.add_argument('--sample', action='store_true', help='Sample z (default); use posterior mean if omitted with --no-sample')
    pi.add_argument('--no-sample', dest='sample', action='store_false')
    pi.set_defaults(sample=True)

    return p


def main(argv=None):
    # Download NLTK data if needed
    download_nltk_data()
    
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    checkpoint = os.path.join(args.output_path, 'model.pt')

    if args.cmd == 'train':
        cfg = TrainConfig(
            train_jsonl=args.train_jsonl,
            checkpoint=checkpoint,
            output_path=args.output_path,
            vocab_size=args.vocab_size,
            min_df=args.min_df,
            bow_cache_dir=getattr(args, 'bow', None),
            num_topic=args.num_topics,
            en1_units=args.en1_units,
            en2_units=args.en2_units,
            variance=args.variance,
            init_mult=args.init_mult,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            batch_size=args.batch_size,
            num_epoch=args.epochs,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
        )
        train(cfg)
    elif args.cmd == 'infer':
        icfg = InferConfig(
            checkpoint=args.checkpoint,
            input_jsonl=args.input_jsonl,
            output_path=args.output_path,
            device=args.device,
            sample=args.sample,
        )
        infer(icfg)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
