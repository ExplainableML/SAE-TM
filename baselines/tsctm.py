#!/usr/bin/env python3
"""
TSCTM (Wu et al., 2022) topic model with the data loading, CLI, checkpointing,
and topic export functionality from a reference-style ProdLDA script.

- Data: JSONL with lines of {"id": str, "document": str}
- Subcommands: `train` and `infer`
- Exports: checkpoint (.pt) + topics TSV/top-N

This script integrates the TSCTM model into the framework of the first script (S1),
ensuring it can be trained and used with the same data formats and command-line
arguments, providing a consistent user experience.
"""
from __future__ import annotations
import argparse
import dataclasses
import os
from typing import List, Dict, Optional
from collections import defaultdict

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
# TSCTM model components (from S2)
# -----------------------------

class TopicDistQuant(nn.Module):
    '''
    Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder. EMNLP 2020
    Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao
    '''
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.copy_(torch.eye(embedding_dim))
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        # Quantize and unflatten
        quantized = self._embedding(encoding_indices)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='none').sum(axis=1).mean()
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='none').sum(axis=1).mean()
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        rst = {
            'loss': loss,
            'quantized': quantized,
            'encoding_indices': encoding_indices,
        }
        return rst


class TSC(nn.Module):
    # Topic-Semantic Contrastive Learning
    def __init__(self, temperature=0.07, weight_contrast=None, use_aug=False):
        super().__init__()
        self.use_aug = use_aug
        self.temperature = temperature
        self.weight_contrast = weight_contrast

    def forward(self, features, quant_idx=None, weight_same_quant=None):
        device = features.device
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases.
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        t_quant_idx = quant_idx.contiguous().view(-1, 1)
        # quant_idx_mask: 1 means same quantization; 0 means different quantization
        quant_idx_mask = torch.eq(t_quant_idx, t_quant_idx.T).float()
        quant_idx_mask = quant_idx_mask.repeat(anchor_count, contrast_count)
        exp_logits = torch.exp(logits) * (1 - quant_idx_mask)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        if not self.use_aug:
            log_prob = logits * logits_mask - torch.log(sum_exp_logits + 1e-10)
            mean_log_prob_pos = (quant_idx_mask * log_prob).sum(1) / quant_idx_mask.sum(1)
        else: # Not used in this implementation, but kept for reference
            log_prob = logits - torch.log(sum_exp_logits + 1e-10)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            same_quant_mask = quant_idx_mask * logits_mask
            same_quant_mean_log_prob_pos = (same_quant_mask * log_prob).sum(1) / (same_quant_mask.sum(1) + 1e-10)
            mean_log_prob_pos += weight_same_quant * same_quant_mean_log_prob_pos

        loss = - self.weight_contrast * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).sum(axis=0).mean()
        return loss


class TSCTM(nn.Module):
    '''
    Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning. EMNLP 2022
    Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong.
    '''
    def __init__(self, vocab_size, num_topic=50, en_units=200, temperature=0.5, weight_contrast=1.0):
        super().__init__()
        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topic)

        self.mean_bn = nn.BatchNorm1d(num_topic)
        self.mean_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.fcd1 = nn.Linear(num_topic, vocab_size, bias=False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.topic_dist_quant = TopicDistQuant(num_topic, num_topic)
        self.contrast_loss = TSC(temperature, weight_contrast)

    def get_beta(self):
        return self.fcd1.weight.T # (num_topic, vocab_size)

    def topic_word_matrix(self) -> torch.Tensor:
        """Return topic-word matrix for export, compatible with S1's framework."""
        return self.get_beta()

    def encode(self, inputs):
        e1 = F.softplus(self.fc11(inputs))
        e1 = F.softplus(self.fc12(e1))
        return self.mean_bn(self.fc21(e1))

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def get_theta(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)
        return softmax_theta

    def forward(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)
        quant_rst = self.topic_dist_quant(softmax_theta)
        recon = self.decode(quant_rst['quantized'])
        recon_loss = self.reconstruction_loss(recon, inputs)
        quant_loss = quant_rst['loss']
        
        features = torch.cat([F.normalize(theta, dim=1).unsqueeze(1)], dim=1)
        contrastive_loss = self.contrast_loss(features, quant_idx=quant_rst['encoding_indices'])
        
        loss = recon_loss + quant_loss + contrastive_loss
        return {'loss': loss, 'contrastive_loss': contrastive_loss, 'recon_loss': recon_loss}

    def reconstruction_loss(self, recon_x, x):
        loss = -(x * (recon_x).log()).sum(axis=1)
        loss = loss.mean()
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
    # Model (TSCTM hyperparams)
    num_topic: int = 50
    en_units: int = 200
    temperature: float = 0.5
    weight_contrast: float = 1.0
    # Optimization
    optimizer: str = 'Adam'
    learning_rate: float = 0.002
    batch_size: int = 200
    num_epoch: int = 80
    # Misc
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

    # --- Data Loading (from S1) ---
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

    # --- Model and Optimizer Setup ---
    model = TSCTM(vocab_size=V, num_topic=K,
                  en_units=cfg.en_units,
                  temperature=cfg.temperature,
                  weight_contrast=cfg.weight_contrast)
    model.to(cfg.device)

    if cfg.optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # --- Training Loop (adapted from S1) ---
    global_step = 0
    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.num_epoch}")
        epoch_losses = defaultdict(float)
        
        for batch in pbar:
            x = batch.to(cfg.device)
            out = model(x)
            loss = out['loss']
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            global_step += 1
            # Log all loss components
            for key, val in out.items():
                epoch_losses[key] += val.item()

            if global_step % cfg.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{out['loss'].item():.4f}",
                    "contrast": f"{out['contrastive_loss'].item():.4f}"
                })
        
        if epoch % 5 == 0:
            avg_losses = {k: v / len(loader) for k, v in epoch_losses.items()}
            log_str = ', '.join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
            print(f"Epoch {epoch}, {log_str}")

    # --- Save checkpoint ---
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
            theta = model.get_theta(x)
            all_theta.append(theta.cpu())
        
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

    model = TSCTM(vocab_size=len(vocab.id2token),
                  num_topic=mcfg['num_topic'],
                  en_units=mcfg['en_units'],
                  temperature=mcfg['temperature'],
                  weight_contrast=mcfg['weight_contrast'])
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
            theta = model.get_theta(x) # Get topic proportions (N x K)
            all_theta.append(theta.cpu())
        
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
# CLI (adapted from S1)
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="TSCTM topic model with reference-style CLI/Data I/O")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train a TSCTM topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # TSCTM hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--en-units', type=int, default=200, help="Encoder hidden units")
    pt.add_argument('--temperature', type=float, default=0.5, help="Temperature for contrastive loss")
    pt.add_argument('--weight-contrast', type=float, default=1.0, help="Weight for contrastive loss")
    # Optimization
    pt.add_argument('--optimizer', choices=['Adam','SGD'], default='Adam')
    pt.add_argument('--learning-rate', type=float, default=0.002)
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
            en_units=args.en_units,
            temperature=args.temperature,
            weight_contrast=args.weight_contrast,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
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
