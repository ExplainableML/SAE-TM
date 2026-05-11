#!/usr/bin/env python3
"""
CombinedTM model integrated with the data loading, CLI, checkpointing, and
topic export functionality from a reference-style script.

- Data:
    - JSONL with lines of {"id": str, "document": str}
    - A .pt file with a tensor of document embeddings
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV/top-N

This script implements the CombinedTM model, which uses both bag-of-words
and pre-computed document embeddings. It is wrapped in a framework that
provides a consistent user experience for training and inference, aligned
with the original DecTM script.
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
from torch.utils.data import DataLoader, Dataset

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
# CombinedTM model (from S2)
# -----------------------------
class CombinedTM(nn.Module):
    def __init__(self, vocab_size, contextual_embed_size, num_topics=50, en_units=200, dropout=0.4):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        mu2 = torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        var2 = torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T)
        self.register_buffer('mu2', mu2)
        self.register_buffer('var2', var2)
        
        self.fc_contextual = nn.Linear(contextual_embed_size, vocab_size)
        self.fc11 = nn.Linear(vocab_size, en_units) # Input is only from contextual projection
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

        # align with the default parameters of tf.contrib.layers.batch_norm
        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn.weight.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False

        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.fcd1.weight)

    def get_beta(self):
        return self.fcd1.weight.T
        
    def topic_word_matrix(self) -> torch.Tensor:
        """Return topic-word matrix for export, compatible with S1's framework."""
        # get_beta returns (K, V), we need (V, K) for the export utility
        return self.get_beta().T

    def get_theta(self, x):
        # The input x is the concatenated tensor of [bow, embedding]
        contextual_embedding = x[:, self.vocab_size:]
        contextual_proj = self.fc_contextual(contextual_embedding)
        
        # The model logic from S2 uses only the projected contextual embedding as input to the encoder
        combined = contextual_proj

        mu, logvar = self.encode(combined)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        
        if self.training:
            theta = self.theta_drop(theta)
            return theta, mu, logvar
        else:
            return theta

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
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        recon_x = self.decode(theta)
        
        # The loss function compares recon_x with the BoW part of the input
        bow_x = x[:, :self.vocab_size]
        loss = self.loss_function(bow_x, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        loss = (recon_loss + KLD).mean()
        return loss


# -----------------------------
# Combined Dataset for BoW + Embeddings
# -----------------------------
class CombinedDataset(Dataset):
    """A PyTorch Dataset that combines a BowDataset with pre-computed embeddings."""
    def __init__(self, bow_dataset: BowDataset, embeddings: torch.Tensor):
        if len(bow_dataset) != len(embeddings):
            raise ValueError(
                f"Mismatch in data points: "
                f"BoW dataset has {len(bow_dataset)} documents, "
                f"but embeddings tensor has {len(embeddings)} entries. "
                f"Please ensure the embedding file corresponds to the input JSONL."
            )
        self.bow_dataset = bow_dataset
        self.embeddings = embeddings

    def __len__(self):
        return len(self.bow_dataset)

    def __getitem__(self, idx):
        bow = self.bow_dataset[idx]
        embedding = self.embeddings[idx].to(dtype=torch.float16)
        # Concatenate bow and embedding features for the model input
        return torch.cat((bow, embedding), dim=0)


# -----------------------------
# Configs and Training/Inference API
# -----------------------------
@dataclasses.dataclass
class TrainConfig:
    train_jsonl: str
    doc_embeddings: str
    checkpoint: str
    output_path: str
    # Vocab
    vocab_size: int = 20000
    min_df: int = 5
    bow_cache_dir: Optional[str] = None
    # Model (CombinedTM hyperparams)
    num_topic: int = 50
    contextual_embed_size: int = 0  # Determined at runtime
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
    doc_embeddings: str
    output_path: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'




# -----------------------------
# Train
# -----------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # --- 1. Load BoW Dataset and Vocab ---
    if cfg.bow_cache_dir:
        os.makedirs(cfg.bow_cache_dir, exist_ok=True)
        cache_path = os.path.join(cfg.bow_cache_dir, f"{cfg.vocab_size}_{cfg.min_df}.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached BowDataset and Vocab from {cache_path}")
            bow_dataset, vocab = BowDataset.load(cache_path)
        else:
            print(f"Building vocab and creating BowDataset, caching to {cache_path}")
            rows = read_jsonl(cfg.train_jsonl)
            vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df, show_progress=True)
            bow_dataset = BowDataset(rows, vocab)
            bow_dataset.save(cache_path, vocab)
    else:
        rows = read_jsonl(cfg.train_jsonl)
        vocab = Vocab.build((r['document'] for r in rows), max_size=cfg.vocab_size, min_df=cfg.min_df, show_progress=True)
        bow_dataset = BowDataset(rows, vocab)
    
    # --- 2. Load Embeddings and Create Combined Dataset ---
    print(f"Loading document embeddings from {cfg.doc_embeddings}")
    embeddings = torch.load(cfg.doc_embeddings, map_location='cpu')
    
    # This will raise a ValueError if counts don't match
    dataset = CombinedDataset(bow_dataset, embeddings)
    cfg.contextual_embed_size = embeddings.shape[1]
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    V = len(vocab.id2token)
    K = cfg.num_topic

    model = CombinedTM(vocab_size=V, num_topics=K,
                       contextual_embed_size=cfg.contextual_embed_size,
                       en_units=cfg.en_units, dropout=cfg.dropout)
    model.to(cfg.device)

    if cfg.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
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
    model = CombinedTM(vocab_size=V, num_topics=mcfg['num_topic'],
                       contextual_embed_size=mcfg['contextual_embed_size'],
                       en_units=mcfg['en_units'], dropout=mcfg['dropout'])
    model.load_state_dict(ckpt['model_state'])
    model.to(cfg.device)
    model.eval()

    rows = read_jsonl(cfg.input_jsonl)
    bow_ds = BowDataset(rows, vocab)
    
    print(f"Loading document embeddings for inference from {cfg.doc_embeddings}")
    embeddings = torch.load(cfg.doc_embeddings, map_location='cpu')
    
    # Create combined dataset for inference
    ds = CombinedDataset(bow_ds, embeddings)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_theta = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='infer'):
            x = batch.to(cfg.device)
            p = model.get_theta(x) # proportions over topics (N x K)
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
    p = argparse.ArgumentParser(description="CombinedTM topic model with reference-style CLI/Data I/O")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train a CombinedTM topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--train-embeddings', required=True, help="Path to .pt file with document embeddings tensor")
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # Model hyperparams
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
    pi.add_argument('--inference-embeddings', required=True, help="Path to .pt file with corresponding document embeddings tensor")
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
            doc_embeddings=args.train_embeddings,
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
            doc_embeddings=args.inference_embeddings,
            output_path=args.output_path,
            device=args.device,
        )
        infer(icfg)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
