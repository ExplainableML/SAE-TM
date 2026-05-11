#!/usr/bin/env python3
"""
DecTM (Wu et al., 2021) topic model with the data loading, CLI, checkpointing,
and topic export functionality from a reference-style ProdLDA script.

- Data: JSONL with lines of {"id": str, "document": str}
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV/top-N

This script integrates the DecTM model into the framework of the first script (S1),
ensuring it can be trained and used with the same data formats and command-line
arguments, providing a consistent user experience.
"""
from __future__ import annotations
import argparse
import dataclasses
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# -----------------------------
# Import shared utilities
# -----------------------------
from utils.dataloading import (
    download_nltk_data,
    read_jsonl,
    write_jsonl,
    Vocab,
    BowDataset,
)
from utils.utils import set_seed, export_topics as export_topics_generic


# -----------------------------
# DecTM model (from S2)
# -----------------------------
class DecTM(nn.Module):
    '''
    Discovering Topics in Long-tailed Corpora with Causal Intervention. ACL 2021 findings.
    Xiaobao Wu, Chunping Li, Yishu Miao.
    '''
    def __init__(self, vocab_size, num_topic=50, en_units=200, dropout=0.4):
        super().__init__()

        self.num_topic = num_topic

        self.a = 1 * np.ones((1, num_topic)).astype(np.float32)
        mu2 = torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        var2 = torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topic))).T + (1.0 / (num_topic * num_topic)) * np.sum(1.0 / self.a, 1)).T)
        self.register_buffer('mu2', mu2)
        self.register_buffer('var2', var2)

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topic)
        self.fc22 = nn.Linear(en_units, num_topic)

        # align with the default parameters of tf.contrib.layers.batch_norm
        self.mean_bn = nn.BatchNorm1d(num_topic, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topic))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topic, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topic))
        self.logvar_bn.weight.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False

        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)

        self.beta = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_topic, vocab_size)))

    def get_beta(self):
        return self.beta
        
    def topic_word_matrix(self) -> torch.Tensor:
        """Return topic-word matrix for export, compatible with S1's framework."""
        return self.get_beta()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        return self.mean_bn(self.fc21(e1)), self.logvar_bn(self.fc22(e1))

    def decode(self, theta):
        norm_theta = F.normalize(theta, dim=1)
        norm_beta = F.normalize(self.beta, dim=0)
        d1 = F.softmax(self.decoder_bn(torch.matmul(norm_theta, norm_beta)), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        theta = self.theta_drop(theta)
        
        recon_x = self.decode(theta)
        loss = self.loss_function(x, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topic)
        loss = (recon_loss + KLD).mean()
        return loss


# -----------------------------
# Configs and Training/Inference API (from S1)
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
    # Model (DecTM hyperparams)
    num_topic: int = 50
    en_units: int = 200
    dropout: float = 0.4
    # Optimization
    optimizer: str = 'Adam'
    learning_rate: float = 0.002
    momentum: float = 0.99
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
    sample: bool = True




# -----------------------------
# Train
# -----------------------------

def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    if cfg.bow_cache_dir:
        os.makedirs(cfg.bow_cache_dir, exist_ok=True)
        cache_path = os.path.join(cfg.bow_cache_dir, f"{cfg.vocab_size}_{cfg.min_df}.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached BowDataset and Vocab from {cache_path}")
            dataset, vocab = BowDataset.load(cache_path)
        else:
            print(f"Building vocab and creating BowDataset, caching to {cache_path}")
            rows = read_jsonl(cfg.train_jsonl)
            vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df, show_progress=True)
            dataset = BowDataset(rows, vocab)
            dataset.save(cache_path, vocab)
    else:
        rows = read_jsonl(cfg.train_jsonl)
        vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df, show_progress=True)
        dataset = BowDataset(rows, vocab)
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    V = len(vocab.id2token)
    K = cfg.num_topic

    model = DecTM(vocab_size=V, num_topic=K,
                  en_units=cfg.en_units, dropout=cfg.dropout)
    model.to(cfg.device)

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
            loss = out['loss'] # DecTM returns loss in the output dict
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            epoch_loss += loss.item()
            if global_step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
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
        topics_matrix = model.get_beta()
    
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
    model = DecTM(vocab_size=V, num_topic=mcfg['num_topic'],
                  en_units=mcfg['en_units'], dropout=mcfg['dropout'])
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
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std
            else:
                z = mean
            p = F.softmax(z, dim=-1) # proportions over topics (N x K)
            all_theta.append(p.cpu())
        
        # Concatenate all batches
        theta_matrix = torch.cat(all_theta, dim=0)  # (N x K)
        
        # Get topic-word matrix (K x V)
        topics_matrix = model.get_beta()  # (K x V)
    
    # Save outputs
    os.makedirs(cfg.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(cfg.output_path, 'theta.pt'))
    torch.save(topics_matrix.cpu(), os.path.join(cfg.output_path, 'topics.pt'))

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
# CLI
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="DecTM topic model with reference-style CLI/Data I/O")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train a DecTM topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # DecTM hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--en-units', type=int, default=200, help="Encoder hidden units")
    pt.add_argument('--dropout', type=float, default=0.4, help="Dropout rate")
    # Optimization
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
    pi.add_argument('--input-jsonl', required=True)
    pi.add_argument('--output-path', required=True)
    pi.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pi.add_argument('--sample', action='store_true', help='Sample z (default); use posterior mean if omitted with --no-sample')
    pi.add_argument('--no-sample', dest='sample', action='store_false')
    pi.set_defaults(sample=True)

    return p

def main(argv=None):
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
            en_units=args.en_units,
            dropout=args.dropout,
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

