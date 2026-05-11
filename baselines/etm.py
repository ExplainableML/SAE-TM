#!/usr/bin/env python3
"""
ETM (Dieng et al., 2020) topic model with the data loading, CLI, checkpointing,
and topic export functionality from a reference-style script.

- Data: JSONL with lines of {"id": str, "document": str}
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV/top-N

This script integrates the ETM model into the framework of the reference script (S1),
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
except ImportError:
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
# ETM model (from S2)
# -----------------------------
class ETM(nn.Module):
    '''
    Topic Modeling in Embedding Spaces. TACL 2020
    Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.
    '''
    def __init__(self, vocab_size, num_topics=50, embed_size=200, en_units=800, dropout=0.5, pretrained_WE=None, train_WE=False):
        super().__init__()
        self.num_topics = num_topics

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))
        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))

        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

    def get_beta(self):
        """Returns the topic-word distribution matrix."""
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1)
        return beta

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
        e1 = self.encoder1(x)
        return self.fc21(e1), self.fc22(e1)

    def get_theta(self, x):
        # Normalize the input as recommended in the original ETM implementation.
        # https://github.com/adjidieng/ETM/issues/3
        norm_x = x / (x.sum(1, keepdim=True) + 1e-10)
        mu, logvar = self.encode(norm_x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        beta = self.get_beta()
        recon_x = torch.matmul(theta, beta)
        loss = self.loss_function(x, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
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
    # Model (ETM hyperparams)
    num_topic: int = 50
    embed_size: int = 200
    en_units: int = 800
    dropout: float = 0.5
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
    model = ETM(vocab_size=V, num_topics=cfg.num_topic,
                embed_size=cfg.embed_size, en_units=cfg.en_units, dropout=cfg.dropout)
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
            loss = out['loss']
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
            p = model.get_theta(x)
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
    model = ETM(vocab_size=V, num_topics=mcfg['num_topic'],
                embed_size=mcfg['embed_size'], en_units=mcfg['en_units'], dropout=mcfg['dropout'])
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
            # In eval mode, get_theta returns topic proportions directly (N x K)
            p = model.get_theta(x)
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
    p = argparse.ArgumentParser(description="ETM topic model with reference-style CLI/Data I/O")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train an ETM topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # ETM hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--embed-size', type=int, default=200, help="Word and topic embedding size")
    pt.add_argument('--en-units', type=int, default=800, help="Encoder hidden units")
    pt.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
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
            embed_size=args.embed_size,
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
        )
        infer(icfg)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
