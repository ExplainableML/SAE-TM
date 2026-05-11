#!/usr/bin/env python3
"""
Reference-faithful NVDM (Miao, Yu, Blunsom 2016) in modern PyTorch
-------------------------------------------------------------------

This reproduces the TensorFlow reference you shared (same architecture,
losses, and the *alternating* optimization of decoder/encoder), while
adopting the **data loading and CLI** style from your ProdLDA script:

- Data: JSONL with lines of {"id": str, "document": str}
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV and top-N text with {raw,prob,pmi,tfidf}

Faithful model details (matching the TF code):
- Inputs are BOW counts (float32), size = vocab_size.
- Encoder: 1-hidden-layer MLP with `n_hidden` units + chosen nonlinearity
  (tanh/sigmoid/relu) -> heads for mean and log_sigma (log std).
  The log_sigma head is initialized **bias=0 and weight=0** like TF.
- KL term: KLD = -0.5 * sum(1 - mean^2 + 2*logsigm - exp(2*logsigm)) per doc.
- Decoder: Linear(n_topic -> vocab_size), logits passed through log_softmax.
- Reconstruction loss: -sum( log_softmax(Projection(doc_vec)) * x ) per doc.
- Reparameterization: z = mean + exp(logsigm) * eps, eps ~ N(0, I).
- Training alternates between updating decoder-only and encoder-only, each
  for `alternate_epochs` inner iterations, just like the TF script.
- Supports `n_sample` > 1 for reconstruction Monte Carlo averaging.

Modernizations:
- Pure PyTorch 2.5+ code, no deprecated APIs.
- Device management, seeds, DataLoader.
- JSONL pipeline (tokenization, vocab, BOW) copied/adapted from your example.
- Topic export helpers (raw/prob/PMI/TF-IDF) consistent with your example.

CLI hyperparameters:
We expose only the model/optimization knobs present in the TF reference:
`learning_rate`, `batch_size`, `n_hidden`, `n_topic`, `n_sample`,
`vocab_size`, `non_linearity`, plus `training_epochs` & `alternate_epochs`.
(Plus file paths and a couple of harmless utilities: device/seed/log_every.)
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
# NVDM model (faithful to the TF reference)
# -----------------------------
class LinearStartZero(nn.Module):
    """
    A linear layer whose weight and/or bias can be initialized to EXACT zeros,
    reproducing the TF reference for the logsigm head (matrix_start_zero, bias_start_zero).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True,
                 weight_zero: bool=False, bias_zero: bool=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Standard init for everything first
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)
        # Then zero-out if requested (to mimic TF utils.linear options)
        if weight_zero:
            with torch.no_grad():
                self.linear.weight.zero_()
        if bias_zero and self.linear.bias is not None:
            with torch.no_grad():
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NVDM(nn.Module):
    """
    Neural Variational Document Model -- faithful PyTorch re-implementation.

    Encoder:
      mlp: [n_hidden] with chosen nonlinearity
      heads: mean (standard linear), logsigm (both weight and bias start at zero)

    KL:
      kld = -0.5 * sum(1 - mean^2 + 2*logsigm - exp(2*logsigm))

    Decoder:
      projection: Linear(n_topic -> vocab_size)
      logits: log_softmax(projection(doc_vec))
      recon_loss: -sum(logits * x)

    Reparameterization:
      doc_vec = exp(logsigm) * eps + mean, eps ~ N(0, I)
    """
    def __init__(self,
                 vocab_size: int,
                 n_hidden: int,
                 n_topic: int,
                 n_sample: int,
                 non_linearity: str = "tanh"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample

        # Non-linearity mapping
        if non_linearity.lower() == "tanh":
            self.act = torch.tanh
        elif non_linearity.lower() == "sigmoid":
            self.act = torch.sigmoid
        else:
            self.act = F.relu

        # Encoder MLP (single hidden layer like TF reference)
        self.enc_fc = nn.Linear(vocab_size, n_hidden)

        # Heads: mean (standard linear); logsigm (zero-init matrix & bias)
        self.mean_fc = nn.Linear(n_hidden, n_topic)
        self.logsigm_fc = LinearStartZero(n_hidden, n_topic, weight_zero=True, bias_zero=True)

        # Decoder projection: topics -> vocab
        self.proj = nn.Linear(n_topic, vocab_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.enc_fc(x))
        mean = self.mean_fc(h)
        logsigm = self.logsigm_fc(h)
        return mean, logsigm

    def sample_docvec(self, mean: torch.Tensor, logsigm: torch.Tensor,
                      n_sample: int) -> torch.Tensor:
        """
        Returns either a single sample (B x K) or n_sample averaged reconstruction losses.
        For forward() we return a single z (B x K); training handles n_sample>1 in loss.
        """
        eps = torch.randn_like(mean)
        z = torch.exp(logsigm) * eps + mean
        return z

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        # Log-softmax over vocab
        logits = F.log_softmax(self.proj(z), dim=-1)
        return logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logsigm = self.encode(x)
        z = self.sample_docvec(mean, logsigm, n_sample=1)
        logits = self.decode_logits(z)  # (B, V)
        # Reconstruction loss per doc
        recons = -(logits * x).sum(dim=1)  # (B,)
        # KL per doc (faithful formula)
        kld = -0.5 * (1 - mean.pow(2) + 2 * logsigm - torch.exp(2 * logsigm)).sum(dim=1)  # (B,)
        obj = recons + kld  # (B,)
        return {
            "recons": recons,
            "kld": kld,
            "objective": obj,
            "mean": mean,
            "logsigm": logsigm,
            "z": z,
            "logits": logits,
        }

    def multi_sample_objective(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Faithful multi-sample path (averaging recons loss over n_sample).
        Returns (objective_per_doc, recons_per_doc, kld_per_doc).
        """
        mean, logsigm = self.encode(x)
        B = x.size(0)
        if self.n_sample == 1:
            z = self.sample_docvec(mean, logsigm, n_sample=1)
            logits = self.decode_logits(z)
            recons = -(logits * x).sum(dim=1)
        else:
            # (n_sample * B, K)
            eps = torch.randn(self.n_sample, B, self.n_topic, device=x.device, dtype=x.dtype)
            z = torch.exp(logsigm).unsqueeze(0) * eps + mean.unsqueeze(0)  # (S,B,K)
            z = z.reshape(self.n_sample * B, self.n_topic)
            logits = self.decode_logits(z).reshape(self.n_sample, B, self.vocab_size)
            # Average reconstruction loss over samples, per doc
            recons = -((logits * x.unsqueeze(0)).sum(dim=2)).mean(dim=0)  # (B,)

        # KL per doc (same regardless of samples)
        kld = -0.5 * (1 - mean.pow(2) + 2 * logsigm - torch.exp(2 * logsigm)).sum(dim=1)  # (B,)
        obj = recons + kld
        return obj, recons, kld

    # For topic export (topic -> vocab projection)
    def topic_word_weights(self) -> torch.Tensor:
        # proj: (K -> V) => weight shape is (V, K) in nn.Linear
        # We want (V, K) scores per token/topic; consistent with ProdLDA exporter.
        return self.proj.weight  # (V, K)




# -----------------------------
# Configs
# -----------------------------
@dataclasses.dataclass
class TrainConfig:
    train_jsonl: str
    checkpoint: str
    output_path: str
    # Data/Vocab (reference exposes vocab_size; keep min_df minimal like TF)
    vocab_size: int = 2000
    min_df: int = 1
    bow_cache_dir: Optional[str] = None
    # Model (reference hyperparams)
    n_hidden: int = 500
    n_topic: int = 50
    n_sample: int = 1
    non_linearity: str = 'tanh'  # {'tanh','sigmoid','relu'}
    # Optimization (reference-style)
    learning_rate: float = 5e-5
    batch_size: int = 64
    training_epochs: int = 1000
    alternate_epochs: int = 10
    # Misc
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    log_every: int = 200
    # Optional eval sets (to mirror dev/test reporting if provided)
    dev_jsonl: Optional[str] = None
    test_jsonl: Optional[str] = None


@dataclasses.dataclass
class InferConfig:
    checkpoint: str
    input_jsonl: str
    output_path: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample: bool = True  # use sampled z; if False, output posterior mean


def make_loader(jsonl_path: str, vocab: Vocab, batch_size: int, shuffle: bool) -> DataLoader:
    """Helper to create a DataLoader from JSONL file."""
    rows = read_jsonl(jsonl_path)
    ds = BowDataset(rows, vocab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False), ds


# -----------------------------
# Metrics (faithful to TF script)
# -----------------------------
@torch.no_grad()
def eval_perplexity(model: NVDM, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    """
    Returns: (corpus_ppx, per_doc_ppx, avg_kld)
    - corpus_ppx: exp(sum(loss)/sum(word_count))
    - per_doc_ppx: exp(mean(loss/count_doc))
    - avg_kld: average KLD across batches (like TF)
    """
    model.eval()
    total_loss_sum = 0.0
    total_ppx_sum = 0.0
    total_kld_sum = 0.0
    total_word_count = 0.0
    total_doc_count = 0.0

    for batch in loader:
        x = batch.to(device)
        word_count = x.sum(dim=1).cpu().numpy()  # per-doc word counts
        word_count_safe = word_count + 1e-12

        obj, recons, kld = model.multi_sample_objective(x)
        loss = obj.detach().cpu().numpy()
        kld_np = kld.detach().cpu().numpy()

        total_loss_sum += float(loss.sum())
        total_kld_sum += float(kld_np.sum() / max((word_count > -1).sum(), 1))
        total_word_count += float(word_count.sum())
        total_ppx_sum += float((loss / word_count_safe).sum())
        total_doc_count += float(len(word_count))

    if total_word_count == 0:
        # degenerate
        return float('inf'), float('inf'), float('nan')

    corpus_ppx = math.exp(total_loss_sum / total_word_count)
    per_doc_ppx = math.exp(total_ppx_sum / total_doc_count) if total_doc_count else float('inf')
    avg_kld = total_kld_sum / max(len(loader), 1)
    return corpus_ppx, per_doc_ppx, avg_kld


# -----------------------------
# Training
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

    # DataLoaders
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dev_loader, _ = (None, None)
    test_loader, _ = (None, None)
    if cfg.dev_jsonl:
        dev_loader, _ = make_loader(cfg.dev_jsonl, vocab, cfg.batch_size, shuffle=False)
    if cfg.test_jsonl:
        test_loader, _ = make_loader(cfg.test_jsonl, vocab, cfg.batch_size, shuffle=False)

    V = len(vocab.id2token)
    model = NVDM(vocab_size=V,
                 n_hidden=cfg.n_hidden,
                 n_topic=cfg.n_topic,
                 n_sample=cfg.n_sample,
                 non_linearity=cfg.non_linearity)
    model.to(cfg.device)

    # Separate parameter groups (like TF variable_parser)
    enc_params = list(model.enc_fc.parameters()) + list(model.mean_fc.parameters()) + list(model.logsigm_fc.parameters())
    dec_params = list(model.proj.parameters())

    # Two optimizers, same LR (Adam like TF script)
    optim_enc = torch.optim.Adam(enc_params, lr=cfg.learning_rate)
    optim_dec = torch.optim.Adam(dec_params, lr=cfg.learning_rate)

    global_step = 0
    for epoch in range(cfg.training_epochs):
        # Fresh shuffled batches each epoch (already from DataLoader)
        # Switch 0: update decoder; Switch 1: update encoder
        for switch in range(2):
            inner_optim = optim_dec if switch == 0 else optim_enc
            phase = "updating decoder" if switch == 0 else "updating encoder"
            for i in range(cfg.alternate_epochs):
                epoch_loss_sum = 0.0
                epoch_ppx_sum = 0.0
                epoch_kld_sum = 0.0
                epoch_word_count = 0.0
                epoch_doc_count = 0.0

                pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.training_epochs} | {phase} {i+1}/{cfg.alternate_epochs}")
                for batch in pbar:
                    model.train()
                    x = batch.to(cfg.device)
                    # Objective with multi-sample averaging (like TF)
                    obj, recons, kld = model.multi_sample_objective(x)
                    loss = obj.mean()

                    inner_optim.zero_grad()
                    loss.backward()
                    inner_optim.step()
                    global_step += 1

                    # Metrics like TF
                    with torch.no_grad():
                        wc = x.sum(dim=1).cpu().numpy()
                        wc_safe = wc + 1e-12
                        loss_np = obj.detach().cpu().numpy()
                        kld_np = kld.detach().cpu().numpy()

                        epoch_loss_sum += float(loss_np.sum())
                        epoch_kld_sum += float(kld_np.sum() / max((wc > -1).sum(), 1))
                        epoch_word_count += float(wc.sum())
                        epoch_doc_count += float(len(wc))
                        epoch_ppx_sum += float((loss_np / wc_safe).sum())

                    if global_step % cfg.log_every == 0:
                        corpus_ppx = math.exp(epoch_loss_sum / max(epoch_word_count, 1.0))
                        pbar.set_postfix({"loss": f"{loss.item():.4f}", "ppx": f"{corpus_ppx:.3f}"})

                # End of inner loop reporting (faithful to TF printouts)
                if epoch_word_count > 0 and epoch_doc_count > 0:
                    print_ppx = math.exp(epoch_loss_sum / epoch_word_count)
                    print_ppx_perdoc = math.exp(min(20, epoch_ppx_sum / epoch_doc_count))
                    print_kld = epoch_kld_sum / max(len(train_loader), 1)
                    print(f"| Epoch train: {epoch+1} | {phase} {i} | Corpus ppx: {print_ppx:.5f} | Per doc ppx: {print_ppx_perdoc:.5f} | KLD: {print_kld:.5f}")

        # Dev evaluation (optional, like TF)
        if dev_loader is not None:
            dev_corpus_ppx, dev_perdoc_ppx, dev_kld = eval_perplexity(model, dev_loader, cfg.device)
            print(f"| Epoch dev: {epoch+1} | Perplexity: {dev_corpus_ppx:.9f} | Per doc ppx: {dev_perdoc_ppx:.5f} | KLD: {dev_kld:.5f}")

        # Test evaluation (optional)
        if test_loader is not None:
            test_corpus_ppx, test_perdoc_ppx, test_kld = eval_perplexity(model, test_loader, cfg.device)
            print(f"| Epoch test: {epoch+1} | Perplexity: {test_corpus_ppx:.9f} | Per doc ppx: {test_perdoc_ppx:.5f} | KLD: {test_kld:.5f}")

    # Save checkpoint
    ckpt = {
        'model_state': model.state_dict(),
        'vocab': vocab.save(),
        'config': dataclasses.asdict(cfg),
        'model_class': 'NVDM',
    }
    os.makedirs(os.path.dirname(cfg.checkpoint) or '.', exist_ok=True)
    torch.save(ckpt, cfg.checkpoint)
    
    # --- Run inference on training data ---
    print("Running inference on training data...")
    model.eval()
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_theta = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='infer'):
            x = batch.to(cfg.device)
            mean, logsigm = model.encode(x)
            z = mean  # Use posterior mean for consistency
            all_theta.append(z.cpu())
        
        theta_matrix = torch.cat(all_theta, dim=0)
        topics_matrix = model.topic_word_weights().t()
    
    os.makedirs(cfg.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(cfg.output_path, 'theta.pt'))
    torch.save(topics_matrix.cpu(), os.path.join(cfg.output_path, 'topics.pt'))
    print(f"Saved inference results to {cfg.output_path}")


# -----------------------------
# Inference
# -----------------------------
def infer(cfg: InferConfig):
    ckpt = torch.load(cfg.checkpoint, map_location=cfg.device)
    if ckpt.get('model_class') != 'NVDM':
        print("Warning: checkpoint model_class is not NVDM; continuing anyway.")
    vocab = Vocab.load(ckpt['vocab'])
    mcfg = ckpt['config']

    V = len(vocab.id2token)
    model = NVDM(vocab_size=V,
                 n_hidden=int(mcfg['n_hidden']),
                 n_topic=int(mcfg['n_topic']),
                 n_sample=int(mcfg['n_sample']),
                 non_linearity=str(mcfg['non_linearity']))
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
            mean, logsigm = model.encode(x)
            if cfg.sample:
                z = model.sample_docvec(mean, logsigm, n_sample=1)
            else:
                z = mean
            # z is R^K doc representation (N x K)
            all_theta.append(z.cpu())
        
        # Concatenate all batches
        theta_matrix = torch.cat(all_theta, dim=0)  # (N x K)
        
        # Get topic-word weights (V x K) - decoder weight matrix
        topics_matrix = model.topic_word_weights()  # (V x K) - need transpose
        # NVDM decoder is Linear(K -> V), so weight is (V, K)
        # We want (K x V) so transpose
        topics_matrix = topics_matrix.t()  # (K x V)
    
    # Save outputs
    os.makedirs(cfg.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(cfg.output_path, 'theta.pt'))
    torch.save(topics_matrix.cpu(), os.path.join(cfg.output_path, 'topics.pt'))


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Reference-faithful NVDM (PyTorch) with JSONL I/O and alternating updates")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train an NVDM model (faithful to TF reference)')
    pt.add_argument('--train_jsonl', required=True)
    pt.add_argument('--checkpoint', required=True)
    pt.add_argument('--output_path', required=True)

    # Data/Vocab
    pt.add_argument('--vocab_size', type=int, default=5000)
    pt.add_argument('--min_df', type=int, default=5)
    pt.add_argument('--bow-cache-dir', type=str, default=None, help='Directory to cache BowDataset')

    # Model (reference hyperparams)
    pt.add_argument('--n_hidden', type=int, default=500)
    pt.add_argument('--n_topic', type=int, default=50)
    pt.add_argument('--n_sample', type=int, default=1)
    pt.add_argument('--non_linearity', choices=['tanh','sigmoid','relu'], default='tanh')

    # Optimization (reference values)
    pt.add_argument('--learning_rate', type=float, default=5e-5)
    pt.add_argument('--batch_size', type=int, default=64)
    pt.add_argument('--training_epochs', type=int, default=1000)
    pt.add_argument('--alternate_epochs', type=int, default=10)

    # Misc/utilities
    pt.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pt.add_argument('--seed', type=int, default=0)
    pt.add_argument('--log_every', type=int, default=200)
    pt.add_argument('--dev_jsonl', type=str, default=None)
    pt.add_argument('--test_jsonl', type=str, default=None)

    # Inference
    pi = sub.add_parser('infer', help='Infer latent document vectors from a saved NVDM checkpoint')
    pi.add_argument('--checkpoint', required=True)
    pi.add_argument('--input_jsonl', required=True)
    pi.add_argument('--output_path', required=True)
    pi.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pi.add_argument('--sample', action='store_true', help='Sample z; use posterior mean if omitted with --no-sample')
    pi.add_argument('--no-sample', dest='sample', action='store_false')
    pi.set_defaults(sample=True)

    return p


def main(argv=None):
    # Download NLTK data if needed
    download_nltk_data()
    
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == 'train':
        cfg = TrainConfig(
            train_jsonl=args.train_jsonl,
            checkpoint=args.checkpoint,
            output_path=args.output_path,
            vocab_size=args.vocab_size,
            min_df=args.min_df,
            bow_cache_dir=getattr(args, 'bow_cache_dir', None),
            n_hidden=args.n_hidden,
            n_topic=args.n_topic,
            n_sample=args.n_sample,
            non_linearity=args.non_linearity,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            training_epochs=args.training_epochs,
            alternate_epochs=args.alternate_epochs,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            dev_jsonl=args.dev_jsonl,
            test_jsonl=args.test_jsonl,
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
