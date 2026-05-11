#!/usr/bin/env python3
from __future__ import annotations
import argparse
import dataclasses
import os
from typing import Optional

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
    Vocab,
    BowDataset,
)
from utils.utils import set_seed


# -----------------------------
# NSTM model and dependencies (from S2)
# -----------------------------

def sinkhorn_loss(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2):
    """
    Sinkhorn loss function for optimal transport.
    """
    device = a.device

    u = (torch.ones_like(a) / a.size()[0]).to(device)

    K = torch.exp(-M * lambda_sh)
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences


class NSTM(nn.Module):
    '''
    Neural Topic Model via Optimal Transport. ICLR 2021
    He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.25,
                 embed_size=200, recon_loss_weight=0.07, sinkhorn_alpha=20):
        super().__init__()

        self.recon_loss_weight = recon_loss_weight
        self.sinkhorn_alpha = sinkhorn_alpha

        self.e1 = nn.Linear(vocab_size, en_units)
        self.e2 = nn.Linear(en_units, num_topics)
        self.e_dropout = nn.Dropout(dropout)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False

        self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))
        self.topic_embeddings = nn.Parameter(torch.empty((num_topics, embed_size)))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)

    def get_beta(self):
        """Returns the topic-word distribution matrix (K, V)."""
        word_embedding_norm = F.normalize(self.word_embeddings)
        topic_embedding_norm = F.normalize(self.topic_embeddings)
        beta = torch.matmul(topic_embedding_norm, word_embedding_norm.T)
        return beta

    def topic_word_matrix(self) -> torch.Tensor:
        """Return topic-word matrix for export, compatible with S1's framework."""
        # get_beta() returns (K, V), export_topics_generic expects (V, K) from this method
        return self.get_beta().T

    def get_theta(self, input_bow):
        """Returns the document-topic distribution (N, K)."""
        theta = F.relu(self.e1(input_bow))
        theta = self.e_dropout(theta)
        theta = self.mean_bn(self.e2(theta))
        theta = F.softmax(theta, dim=-1)
        return theta

    def forward(self, input_bow):
        theta = self.get_theta(input_bow)
        beta = self.get_beta()
        
        # Sinkhorn loss
        M = 1 - beta
        sh_loss = sinkhorn_loss(M, theta.T, F.softmax(input_bow, dim=-1).T, lambda_sh=self.sinkhorn_alpha)
        
        # Reconstruction loss
        recon = F.softmax(torch.matmul(theta, beta), dim=-1)
        recon_loss = -(input_bow * recon.log().clamp_min(-1000)).sum(axis=1)

        # Total loss
        loss = self.recon_loss_weight * recon_loss + sh_loss
        loss = loss.mean()
        return {'loss': loss}


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
    # Model (NSTM hyperparams)
    num_topics: int = 50
    en_units: int = 200
    dropout: float = 0.25
    embed_size: int = 200
    recon_loss_weight: float = 0.07
    sinkhorn_alpha: float = 20.0
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
    K = cfg.num_topics

    model = NSTM(vocab_size=V, num_topics=K, en_units=cfg.en_units, dropout=cfg.dropout,
                 embed_size=cfg.embed_size, recon_loss_weight=cfg.recon_loss_weight,
                 sinkhorn_alpha=cfg.sinkhorn_alpha)
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

        if epoch % 5 == 0 or epoch == cfg.num_epoch:
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
    print(f"Checkpoint saved to {cfg.checkpoint}")
    
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

    model = NSTM(vocab_size=len(vocab.id2token), num_topics=mcfg['num_topics'],
                 en_units=mcfg['en_units'], dropout=mcfg['dropout'],
                 embed_size=mcfg['embed_size'],
                 recon_loss_weight=mcfg['recon_loss_weight'],
                 sinkhorn_alpha=mcfg['sinkhorn_alpha'])
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
            # NSTM directly computes theta (topic proportions) (N x K)
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
    p = argparse.ArgumentParser(description="NSTM")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train an NSTM topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # NSTM hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--en-units', type=int, default=200, help="Encoder hidden units")
    pt.add_argument('--dropout', type=float, default=0.25, help="Dropout rate")
    pt.add_argument('--embed-size', type=int, default=200, help="Word and topic embedding size")
    pt.add_argument('--recon-loss-weight', type=float, default=0.07, help="Weight for reconstruction loss")
    pt.add_argument('--sinkhorn-alpha', type=float, default=20.0, help="Alpha for Sinkhorn loss")
    # Optimization
    pt.add_argument('--optimizer', choices=['Adam','SGD'], default='Adam')
    pt.add_argument('--learning-rate', type=float, default=0.002)
    pt.add_argument('--momentum', type=float, default=0.99, help="Momentum for SGD or Beta1 for Adam")
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
            num_topics=args.num_topics,
            en_units=args.en_units,
            dropout=args.dropout,
            embed_size=args.embed_size,
            recon_loss_weight=args.recon_loss_weight,
            sinkhorn_alpha=args.sinkhorn_alpha,
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
