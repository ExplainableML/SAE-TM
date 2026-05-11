#!/usr/bin/env python3
from __future__ import annotations
import argparse
import dataclasses
import os
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Import shared utilities
from utils.dataloading import download_nltk_data, Vocab
from utils.utils import set_seed


# -----------------------------
# FASTopic core (ETP + model)
# -----------------------------
def pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: (n,d), y: (m,d) => (n,m)
    return (x**2).sum(dim=1, keepdim=True) + (y**2).sum(dim=1) - 2.0 * (x @ y.t())

class ETP(nn.Module):
    """Embedding Transport Plan via log-domain Sinkhorn (faithful to reference)."""
    def __init__(self, sinkhorn_alpha: float,
                 init_a_dist: Optional[torch.Tensor]=None,
                 init_b_dist: Optional[torch.Tensor]=None,
                 OT_max_iter: int=500,
                 stop_thr: float=0.5e-2):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stop_thr = stop_thr
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist
        if init_a_dist is not None:
            self.a_dist = init_a_dist
        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        M = pairwise_euclidean_distance(x, y)                  # (n,m)
        device = M.device
        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0], device=device) / M.shape[0]).unsqueeze(1)    # (n,1)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)
        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1], device=device) / M.shape[1]).unsqueeze(1)    # (m,1)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        log_a = torch.log(a + 1e-30)
        log_b = torch.log(b + 1e-30)
        log_u = torch.zeros_like(log_a)    # (n,1)
        log_v = torch.zeros_like(log_b)    # (m,1)
        log_K = -M * self.sinkhorn_alpha   # (n,m)

        err = 1.0
        cpt = 0
        # log-domain updates with periodic absorption for stability
        while err > self.stop_thr and cpt < self.OT_max_iter:
            log_Ku = log_K.T + log_u.T                    # (m,n)
            log_v = log_b - torch.logsumexp(log_Ku, dim=1, keepdim=True)  # (m,1)

            log_Kv = log_K + log_v.T                      # (n,m)
            log_u = log_a - torch.logsumexp(log_Kv, dim=1, keepdim=True)  # (n,1)

            cpt += 1
            if cpt % 50 == 1:
                # absorb scalings
                log_K = log_K + log_u + log_v.T
                log_u = torch.zeros_like(log_a)
                log_v = torch.zeros_like(log_b)
                # check constraints
                err = self._check_convergence(log_K, log_u, log_v, a, b)

        u = torch.exp(log_u)
        v = torch.exp(log_v)
        K = torch.exp(log_K)
        transp = u * (K * v.T)            # (n,m)
        loss_ETP = torch.sum(transp * M)
        return loss_ETP, transp

    @torch.no_grad()
    def _check_convergence(self, log_K, log_u, log_v, a, b) -> float:
        log_Kv = log_K + log_v.T
        log_row = log_u + torch.logsumexp(log_Kv, dim=1, keepdim=True)
        rows = torch.exp(log_row)

        log_Ku = log_K.T + log_u.T
        log_col = log_v + torch.logsumexp(log_Ku, dim=1, keepdim=True)
        cols = torch.exp(log_col)

        row_err = torch.abs(rows - a).sum()
        col_err = torch.abs(cols - b).sum()
        return max(row_err.item(), col_err.item())


class FASTopicModel(nn.Module):
    """FASTopic with DSR + ETP (faithful to reference code)."""
    def __init__(self, num_topics: int, theta_temp: float=1.0,
                 DT_alpha: float=3.0, TW_alpha: float=2.0):
        super().__init__()
        self.num_topics = num_topics
        self.theta_temp = theta_temp
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.epsilon = 1e-12

        # Initialized in .init()
        self.topic_embeddings: nn.Parameter = None
        self.topic_weights: nn.Parameter = None
        self.word_embeddings: nn.Parameter = None
        self.word_weights: nn.Parameter = None

        self.DT_ETP: ETP = None
        self.TW_ETP: ETP = None

    @torch.no_grad()
    def _norm_like_ref(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)

    def init(self, vocab_size: int, embed_size: int):
        topic_embeddings = self._norm_like_ref(torch.empty((self.num_topics, embed_size)))
        nn.init.trunc_normal_(topic_embeddings)
        topic_embeddings = self._norm_like_ref(topic_embeddings)
        topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)

        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)

        word_embeddings = torch.empty((vocab_size, embed_size))
        nn.init.trunc_normal_(word_embeddings)
        word_embeddings = self._norm_like_ref(word_embeddings)

        word_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)

        self.word_embeddings = nn.Parameter(word_embeddings)
        self.word_weights = nn.Parameter(word_weights)

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        # transport topics -> words; use plan as beta (K,V) scaled by K
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        beta = transp_TW * transp_TW.shape[0]  # (K,V)
        return beta

    @torch.no_grad()
    def get_theta(self, doc_embeddings: torch.Tensor, train_doc_embeddings: torch.Tensor) -> torch.Tensor:
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)          # (n,K)
        train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)  # (N,K)

        # Compute in log-space for numerical stability
        log_exp_dist = -dist / self.theta_temp                         # (n,K)
        log_exp_train_dist = -train_dist / self.theta_temp             # (N,K)

        # log(denom) = logsumexp for the denominator for each topic k
        log_denom = torch.logsumexp(log_exp_train_dist, dim=0)         # (K,)
        log_theta = log_exp_dist - log_denom.unsqueeze(0)              # (n,K)

        # Now normalize each row (document) using logsumexp for stable softmax
        log_theta = log_theta - torch.logsumexp(log_theta, dim=1, keepdim=True)  # (n,K)
        theta = torch.exp(log_theta)                                   # (n,K)

        return theta

    def forward(self, train_bow: torch.Tensor, doc_embeddings: torch.Tensor):
        # ETP terms
        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        loss_ETP = loss_DT + loss_TW

        # relations -> distributions (scaled)
        theta = transp_DT * transp_DT.shape[0]     # (N,K)
        beta = transp_TW * transp_TW.shape[0]      # (K,V)

        recon = theta @ beta                        # (N,V)
        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(dim=1).mean()

        loss = loss_DSR + loss_ETP
        return {"loss": loss}




# -----------------------------
# Memory-efficient BOW helpers
# -----------------------------
def preprocess_bows(bow_counts: List[Dict], vocab: Vocab) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a compact per-document BOW as (idx_tensor[int64], cnt_tensor[float32]).
    This avoids a (num_docs x vocab_size) dense tensor in RAM.
    """
    compact: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for r in tqdm(bow_counts, desc="Preprocessing bows"):
        if r:
            idx = torch.tensor([t for t in r.keys()], dtype=torch.int64)
            cnt = torch.tensor([float(c) for c in r.values()], dtype=torch.float32)
        else:
            idx = torch.empty(0, dtype=torch.int64)
            cnt = torch.empty(0, dtype=torch.float32)
        compact.append((idx, cnt))
    return compact

def dense_bow_batch(compact_bows: List[Tuple[torch.Tensor, torch.Tensor]],
                    vocab_size: int,
                    device: torch.device) -> torch.Tensor:
    """
    Materialize a dense BOW batch on the fly on the target device.
    This is O(batch_size * V) zeroing, which is unavoidable anyway because
    recon = theta @ beta produces (batch_size, V).
    """
    bsz = len(compact_bows)
    xb = torch.zeros((bsz, vocab_size), dtype=torch.float32, device=device)
    for i, (idx, cnt) in enumerate(compact_bows):
        if idx.numel():
            xb[i, idx.to(device)] = cnt.to(device)
    return xb


# -----------------------------
# Configs
# -----------------------------
@dataclasses.dataclass
class TrainConfig:
    train_jsonl: str
    checkpoint: str
    output_path: str
    vocab_size: int
    min_df: int
    bow_cache_path: str  # path to cached vocab/bow (used only to load vocab)
    doc_embeddings: str  # path to .pt file with doc embeddings
    # Model (as in reference)
    num_topics: int = 50
    DT_alpha: float = 3.0
    TW_alpha: float = 2.0
    theta_temp: float = 1.0
    # Optimization (as in reference)
    learning_rate: float = 0.002
    epochs: int = 200
    # Batching (not a model hyperparam; for memory)
    batch_size: int = 2048
    # Misc
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 13
    log_interval: int = 10
    verbose: bool = False  # parsed but currently unused

@dataclasses.dataclass
class InferConfig:
    checkpoint: str
    output_path: str
    doc_embeddings: str  # path to .pt file with doc embeddings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 2048


def batched_index_iter(N: int, batch_size: int):
    """Yield index slices for batching without materializing data."""
    for i in range(0, N, batch_size):
        yield slice(i, min(i + batch_size, N))


# -----------------------------
# Train
# -----------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # Data & vocab - load from cache (to keep exact vocab)
    from utils.dataloading import BowDataset
    ds, vocab = BowDataset.load(os.path.join(cfg.bow_cache_path, f"{cfg.vocab_size}_{cfg.min_df}.pt"))

    # ---- Memory-efficient BOW: compact per-doc indices+counts; no dense matrix in RAM
    compact_bows = preprocess_bows(ds._bow_counts, vocab)

    # Doc embeddings - keep on CPU; move batches to device later
    doc_emb_cpu = torch.load(cfg.doc_embeddings, map_location='cpu')
    if not isinstance(doc_emb_cpu, torch.Tensor):
        raise ValueError("doc_embeddings must be a PyTorch tensor")
    doc_emb_cpu = doc_emb_cpu.to(torch.float32)  # stay on CPU

    # Model
    V = len(vocab.id2token)
    D = doc_emb_cpu.shape[1]
    model = FASTopicModel(num_topics=cfg.num_topics,
                          theta_temp=cfg.theta_temp,
                          DT_alpha=cfg.DT_alpha,
                          TW_alpha=cfg.TW_alpha)
    model.init(vocab_size=V, embed_size=D)
    model = model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Train
    model.train()
    N = len(ds)
    for epoch in range(1, cfg.epochs + 1):
        loss_sum = 0.0
        seen = 0
        for sl in tqdm(batched_index_iter(N, cfg.batch_size), desc=f"[Epoch {epoch:03d}] Training FASTopic", total=N // cfg.batch_size + 1):
            # Build dense BOW batch on device from compact representation
            xb = dense_bow_batch(compact_bows[sl], V, torch.device(cfg.device))
            # Move only the needed doc-embedding slice to device
            eb = doc_emb_cpu[sl].to(cfg.device, non_blocking=False)

            rst = model(xb, eb)
            loss = rst["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        if epoch % cfg.log_interval == 0 or epoch == 1 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d} loss: {loss_sum / max(seen,1):.4f}")

    ckpt = {
        'model_state': model.state_dict(),
        'vocab': vocab.save(),
        'config': dataclasses.asdict(cfg),
        'model_class': 'FASTopic',
    }
    os.makedirs(os.path.dirname(cfg.checkpoint) or '.', exist_ok=True)
    torch.save(ckpt, cfg.checkpoint)
    
    # --- Run inference on training data ---
    print("Running inference on training data...")
    model.eval()
    all_theta = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, N, cfg.batch_size), desc="Inferring on training data", total=N//cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, N)
            batch_emb = doc_emb_cpu[batch_start:batch_end].to(cfg.device)
            theta = model.get_theta(batch_emb, doc_emb_cpu.to(cfg.device))
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
    if ckpt.get('model_class') != 'FASTopic':
        print("Warning: checkpoint model_class is not FASTopic; continuing.")

    vocab = Vocab.load(ckpt['vocab'])
    mcfg = ckpt['config']

    # Rebuild model
    V = len(vocab.id2token)
    doc_emb = torch.load(cfg.doc_embeddings, map_location=cfg.device)
    if not isinstance(doc_emb, torch.Tensor):
        raise ValueError("doc_embeddings must be a PyTorch tensor")
    doc_emb = doc_emb.to(torch.float32).to(cfg.device)
    embed_dim = doc_emb.shape[1]

    model = FASTopicModel(num_topics=int(mcfg['num_topics']),
                          theta_temp=float(mcfg['theta_temp']),
                          DT_alpha=float(mcfg['DT_alpha']),
                          TW_alpha=float(mcfg['TW_alpha']))
    model.init(vocab_size=V, embed_size=embed_dim)  # shapes created
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(cfg.device)
    model.eval()

    all_theta = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(doc_emb), cfg.batch_size), desc="Inferring FASTopic", total=len(doc_emb)//cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, len(doc_emb))
            batch_emb = doc_emb[batch_start:batch_end]
            theta = model.get_theta(batch_emb, doc_emb)  # (n, K)
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
# CLI
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="FASTopic")
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train', help='Train FASTopic (DSR + ETP)')
    pt.add_argument('--train-jsonl', required=True)
    pt.add_argument('--output-path', required=True)
    # Data
    pt.add_argument('--vocab-size', type=int, default=5000)
    pt.add_argument('--min-df', type=int, default=5)    
    pt.add_argument('--bow', type=str, default=None, help='Directory to cache BowDataset')
    pt.add_argument('--train-embeddings', required=True, help='Path to .pt file with doc embeddings')
    # Model (reference hyperparams)
    pt.add_argument('--num-topics', type=int, default=50)
    pt.add_argument('--DT-alpha', type=float, default=3.0)
    pt.add_argument('--TW-alpha', type=float, default=2.0)
    pt.add_argument('--theta-temp', type=float, default=1.0)
    # Optimization (reference)
    pt.add_argument('--learning-rate', type=float, default=0.002)
    pt.add_argument('--batch-size', type=int, default=2048)
    pt.add_argument('--epochs', type=int, default=200)
    # Misc
    pt.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pt.add_argument('--seed', type=int, default=13)
    pt.add_argument('--log-interval', type=int, default=10)
    
    # Inference
    pi = sub.add_parser('infer', help='Infer doc-topic distributions with a saved FASTopic checkpoint')
    pi.add_argument('--checkpoint', required=True)
    pi.add_argument('--output-path', required=True)
    pi.add_argument('--inference-embeddings', required=True, help='Path to .pt file with doc embeddings')
    pi.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    pi.add_argument('--batch-size', type=int, default=2048)
    return p

def main(argv=None):
    # Download NLTK data if needed (kept global to avoid surprising environments)
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
            bow_cache_path=getattr(args, 'bow', None),
            doc_embeddings=args.train_embeddings,
            num_topics=args.num_topics,
            DT_alpha=args.DT_alpha,
            TW_alpha=args.TW_alpha,
            theta_temp=args.theta_temp,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            log_interval=args.log_interval,
            verbose=True,
        )
        train(cfg)
    elif args.cmd == 'infer':
        icfg = InferConfig(
            checkpoint=args.checkpoint,
            output_path=args.output_path,
            doc_embeddings=args.inference_embeddings,
            device=args.device,
            batch_size=args.batch_size,
        )
        infer(icfg)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
