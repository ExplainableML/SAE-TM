#!/usr/bin/env python3
from __future__ import annotations
import argparse
import dataclasses
import os
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Import shared utilities
from utils.dataloading import (
    download_nltk_data,
    read_jsonl,
    Vocab,
    BowDataset,
)
from utils.utils import set_seed


# -----------------------------
# Math utils for Dirichlet/Gamma
# -----------------------------
_EPS = 1e-10
_MIN_ALPHA = 1e-5


def log_beta_fn(alpha: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha.sum(dim=-1))


def dirichlet_logpdf(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return -log_beta_fn(alpha) + ((alpha - 1.0) * (z + _EPS).log()).sum(dim=-1)


def kl_dirichlet_analytical(q_alpha: torch.Tensor, p_alpha: torch.Tensor) -> torch.Tensor:
    q_sum = q_alpha.sum(dim=-1)
    p_sum = p_alpha.sum(dim=-1)
    t1 = torch.lgamma(q_sum) - torch.lgamma(p_sum)
    t2 = (torch.lgamma(p_alpha) - torch.lgamma(q_alpha)).sum(dim=-1)
    t3 = ((q_alpha - p_alpha) * (torch.digamma(q_alpha) - torch.digamma(q_sum).unsqueeze(-1))).sum(dim=-1)
    return t1 + t2 + t3


class RSVIGamma:
    @staticmethod
    def sample(alpha: torch.Tensor, shape_aug_B: int) -> torch.Tensor:
        device = alpha.device
        a_tilde = alpha + float(shape_aug_B)
        eps = torch.randn_like(alpha, device=device)
        c = a_tilde - 1.0/3.0
        v = 1.0 + eps / torch.sqrt(9.0 * a_tilde - 3.0)
        z_tilde = c * (v.clamp_min(0.0) ** 3)
        if shape_aug_B > 0:
            prod = torch.ones_like(alpha, device=device)
            for i in range(shape_aug_B):
                u = torch.rand_like(alpha, device=device)
                prod = prod * (u + _EPS) ** (1.0 / (alpha + float(i)))
            z = z_tilde * prod
        else:
            z = z_tilde
        return z.clamp_min(_EPS)


# -----------------------------
# DVAE / DVAE-Sparse model
# -----------------------------
class DVAEEncoder(nn.Module):
    def __init__(self, num_input: int, num_topic: int, dropout_p: float = 0.2, use_sparse_gate: bool = False):
        super().__init__()
        H = 100  # paper: 100 neurons in all hidden layers
        self.fc1 = nn.Linear(num_input, H)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(H, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.fc_alpha = nn.Linear(H, num_topic)
        self.use_sparse_gate = use_sparse_gate
        if use_sparse_gate:
            self.fc_b = nn.Linear(H, num_topic)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = self.bn2(self.fc2(h))
        a_pre = self.fc_alpha(h)
        alpha = F.softplus(a_pre) + _MIN_ALPHA
        out = {"alpha": alpha}
        if self.use_sparse_gate:
            b_logits = self.fc_b(h)
            s = torch.sigmoid(b_logits)
            b_hard = (s > 0.5).float()
            # straight-through estimator
            b = b_hard.detach() - s.detach() + s
            out.update({"b_prob": s, "b": b})
        return out


class DVAE(nn.Module):
    def __init__(self, num_input: int, num_topic: int, prior_alpha0: float, use_sparse_gate: bool,
                 shape_aug_B: int = 5, omit_correction_terms: bool = True):
        super().__init__()
        self.V = num_input
        self.K = num_topic
        self.shape_aug_B = int(shape_aug_B)
        self.omit_corr = bool(omit_correction_terms)
        self.encoder = DVAEEncoder(num_input, num_topic, dropout_p=0.2, use_sparse_gate=use_sparse_gate)
        # Decoder
        self.dec_fc = nn.Linear(num_topic, num_input)
        self.dec_bn = nn.BatchNorm1d(num_input)
        # Prior α0
        self.register_buffer("prior_alpha0", torch.full((1, num_topic), float(prior_alpha0)))

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        # decoder -> batchnorm -> log_softmax over vocab
        l = self.dec_bn(self.dec_fc(z))
        log_probs = F.log_softmax(l, dim=-1)
        return log_probs

    def rsvi_dirichlet(self, alpha: torch.Tensor, b_mask: torch.Tensor | None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Sample z ~ Dir(alpha) via Gamma RSVI; optionally mask by b (DVAE-Sparse)
        if b_mask is not None:
            alpha_eff = alpha * (b_mask > 0.5).float()
        else:
            alpha_eff = alpha
        # Sample K independent gamma
        g = RSVIGamma.sample(alpha_eff, self.shape_aug_B)
        # zero out inactive topics explicitly (for numerical stability)
        if b_mask is not None:
            g = g * (b_mask > 0.5).float()
        g_sum = g.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        z = g / g_sum
        aux = {"gamma": g, "alpha_eff": alpha_eff}
        return z, aux

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        alpha = enc["alpha"]
        b = enc.get("b") if self.encoder.use_sparse_gate else None
        z, aux = self.rsvi_dirichlet(alpha, b)
        log_probs = self.decode_logits(z)
        return {"alpha": alpha, "b": b, "z": z, "log_probs": log_probs, **aux}

    # --- Loss pieces ---
    def recon_loss(self, x: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        # Negative log-likelihood for bag-of-words counts with log softmax decoder
        return -(x * log_probs).sum(dim=1)

    def kl_sampled(self, z: torch.Tensor, q_alpha: torch.Tensor, p_alpha: torch.Tensor, b_mask: torch.Tensor | None) -> torch.Tensor:
        # KL[q||p] estimated as E_q[log q(z) - log p(z)] with the same sampled z
        if b_mask is not None:
            q_alpha = q_alpha * (b_mask > 0.5).float() + _MIN_ALPHA
        log_q = dirichlet_logpdf(z, q_alpha)
        log_p = dirichlet_logpdf(z, p_alpha.expand_as(q_alpha))
        return (log_q - log_p)

    def loss(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = out["z"]
        log_probs = out["log_probs"]
        alpha = out["alpha"]
        b = out.get("b")
        recon = self.recon_loss(x, log_probs)
        kl = self.kl_sampled(z, alpha, self.prior_alpha0, b)
        return (recon + kl).mean()

    # utilities
    def topic_word_matrix(self) -> torch.Tensor:
        return self.dec_fc.weight.t()  # (V x K)


# -----------------------------
# Training / Inference configs
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
    # Model (paper hyperparams only)
    num_topic: int = 50
    prior_alpha0: float = 0.1   # scalar prior α0 for all topics
    use_sparse: bool = False    # DVAE if False, DVAE-Sparse if True
    shape_aug_B: int = 5        # paper suggests ≥5 is safe to omit corrections
    # Optimization (paper-style)
    learning_rate: float = 0.002
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
    sample: bool = True  # sample z as in training; if False use E[z] under q




# -----------------------------
# Train / Infer
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

    model = DVAE(num_input=V, num_topic=K, prior_alpha0=cfg.prior_alpha0,
                 use_sparse_gate=cfg.use_sparse, shape_aug_B=cfg.shape_aug_B,
                 omit_correction_terms=True)
    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.num_epoch}")
        for batch in pbar:
            x = batch.to(cfg.device)
            out = model(x)
            loss = model.loss(x, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            if global_step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
            enc = model.encoder(x)
            alpha = enc['alpha']
            b = enc.get('b') if model.encoder.use_sparse_gate else None
            # Use expectation for consistency
            if b is not None:
                alpha_eff = alpha * (b > 0.5).float() + _MIN_ALPHA
            else:
                alpha_eff = alpha
            p = alpha_eff / alpha_eff.sum(dim=-1, keepdim=True)
            all_theta.append(p.cpu())
        
        theta_matrix = torch.cat(all_theta, dim=0)
        topics_matrix = model.dec_fc.weight.t()
    
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


def infer(cfg: InferConfig):
    ckpt = torch.load(cfg.checkpoint, map_location=cfg.device)
    vocab = Vocab.load(ckpt['vocab'])
    mcfg = ckpt['config']

    V = len(vocab.id2token)
    K = int(mcfg['num_topic'])
    model = DVAE(num_input=V, num_topic=K, prior_alpha0=float(mcfg['prior_alpha0']),
                 use_sparse_gate=bool(mcfg['use_sparse']), shape_aug_B=int(mcfg['shape_aug_B']))
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
            enc = model.encoder(x)
            alpha = enc['alpha']
            b = enc.get('b') if model.encoder.use_sparse_gate else None
            if cfg.sample:
                z, _ = model.rsvi_dirichlet(alpha, b)
                p = z
            else:
                # E[z] under Dirichlet = alpha / sum(alpha)
                if b is not None:
                    alpha_eff = alpha * (b > 0.5).float() + _MIN_ALPHA
                else:
                    alpha_eff = alpha
                p = alpha_eff / alpha_eff.sum(dim=-1, keepdim=True)
            # p is topic proportions (N x K)
            all_theta.append(p.cpu())
        
        # Concatenate all batches
        theta_matrix = torch.cat(all_theta, dim=0)  # (N x K)
        
        # Get topic-word matrix (K x V) - decoder weight is (V x K), need transpose
        topics_matrix = model.dec_fc.weight.t()  # (K x V)
    
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
    p = argparse.ArgumentParser(description="Dirichlet VAE (DVAE/DVAE-Sparse) with user's CLI/Data")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train a Dirichlet VAE topic model')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Vocab
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    # Paper hyperparams
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--prior-alpha0', type=float, default=0.1)
    pt.add_argument('--use-sparse', action='store_true', help='Enable DVAE-Sparse (decoupled b gate)')
    pt.add_argument('--shape-aug-B', type=int, default=5)
    # Optimization
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
    pi.add_argument('--sample', action='store_true')
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
            prior_alpha0=args.prior_alpha0,
            use_sparse=args.use_sparse,
            shape_aug_B=args.shape_aug_B,
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
            sample=args.sample,
        )
        infer(icfg)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
