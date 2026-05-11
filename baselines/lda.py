#!/usr/bin/env python3
"""
Train & infer with Gensim LDA using ETM-style BOW caches.

Train:
  python lda_tools.py train-lda \
    --dataset cache/bow_train_20000_5.pt \
    --out-model cache/lda_50topics.gensim \
    --num-topics 50 --passes 5 --iterations 100 --alpha auto --eta auto

Infer (save per-doc topic distributions):
  python lda_tools.py infer \
    --model cache/lda_50topics.gensim \
    --dataset cache/bow_train_20000_5.pt \
    --out-doc-topics cache/lda_theta_train.pt
"""
from __future__ import annotations
import argparse
import glob as _glob
import os
import sys
from typing import Dict, Iterator, List, Tuple

import torch
from gensim.models.ldamodel import LdaModel

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---- Shared utils you provided (import paths may vary in your repo) ----
from utils.dataloading import Vocab  # tokenizer/vocab/BOW cache helpers
from utils.utils import set_seed  # export + seeding


# -----------------------------
# Helpers
# -----------------------------


def _id2word_from_vocab(vocab: Vocab) -> Dict[int, str]:
    return {i: tok for i, tok in enumerate(vocab.id2token)}

def _corpus_from_bow_counts(bow_counts: List[Dict[int, float]]) -> Iterator[List[Tuple[int, float]]]:
    for bow in bow_counts:
        if bow:
            yield sorted(bow.items())

def _iter_corpus_from_shards(paths: List[str]) -> Iterator[List[Tuple[int, float]]]:
    for p in paths:
        shard = torch.load(p, map_location="cpu")
        for bow in shard["bow_counts"]:
            if bow:
                yield sorted(bow.items())

def _iter_ids_from_shards(paths: List[str]) -> Iterator[str]:
    for p in paths:
        shard = torch.load(p, map_location="cpu")
        for _id in shard["ids"]:
            yield _id

def _load_vocab_from_dataset(dataset_path: str) -> Vocab:
    obj = torch.load(dataset_path, map_location="cpu")
    return Vocab.load(obj["vocab"])

def _load_vocab_file(vocab_path: str) -> Vocab:
    pack = torch.load(vocab_path, map_location="cpu")
    vocab_dict = pack["vocab"] if "vocab" in pack else pack
    return Vocab.load(vocab_dict)


# -----------------------------
# Command: train-lda
# -----------------------------

def cmd_train_lda(args):
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    if args.seed is not None:
        set_seed(args.seed)

    if (args.dataset is None) == (len(args.shards) == 0):
        print("[train-lda] Provide exactly one of: --dataset OR --shards ... --vocab", file=sys.stderr)
        sys.exit(2)

    if args.dataset:
        data = torch.load(args.dataset, map_location="cpu")
        vocab = Vocab.load(data["vocab"])
        id2word = _id2word_from_vocab(vocab)
        bow_counts = data["bow_counts"]
        corpus = list(iter(_corpus_from_bow_counts(bow_counts)))       # <- re-iterable + has __len__
        total_docs = len(data["bow_counts"])
        print(f"[train-lda] Using merged dataset: {args.dataset} (docs={total_docs}, |V|={len(vocab.id2token)})")
    else:
        if not args.vocab:
            print("[train-lda] In --shards mode you must also pass --vocab.", file=sys.stderr)
            sys.exit(2)
        shard_paths: List[str] = []
        for pat in args.shards:
            shard_paths.extend(sorted(_glob.glob(pat)))
        if not shard_paths:
            print("[train-lda] No shard files matched the given patterns.", file=sys.stderr)
            sys.exit(2)
        vocab = _load_vocab_file(args.vocab)
        id2word = _id2word_from_vocab(vocab)
        bow_counts = data["bow_counts"]
        corpus = list(iter(_corpus_from_bow_counts(bow_counts)))          # <- re-iterable + has __len__
        print(f"[train-lda] Streaming {len(shard_paths)} shards; |V|={len(vocab.id2token)}")
        print("[train-lda] Tip: keep --passes small (e.g., 1) for streaming input.")

    lda_kwargs = dict(
        corpus=corpus,
        id2word=id2word,
        num_topics=args.num_topics,
        chunksize=args.chunksize,
        passes=args.passes,
        iterations=args.iterations,
        update_every=args.update_every,
        alpha=args.alpha,
        eta=args.eta,
        eval_every=args.eval_every,
        random_state=args.random_state,
        per_word_topics=False,
        dtype=None,
    )
    print("[train-lda] Training LDA with parameters:")
    for k, v in lda_kwargs.items():
        if k in ("corpus", "id2word"):
            continue
        print(f"  - {k}: {v}")

    lda = LdaModel(**lda_kwargs)

    lda.save(args.out_model)
    print(f"[train-lda] Saved LDA model to: {args.out_model}")
    
    # --- Run inference on training data ---
    print("[train-lda] Running inference on training data...")
    K = lda.num_topics
    out_theta: List[List[float]] = []
    
    for bow in tqdm(corpus, desc="Inferring training data"):
        doc_topics = lda.get_document_topics(bow, minimum_probability=0.0)
        theta = _dense_theta(doc_topics, K)
        out_theta.append(theta)
    
    theta_matrix = torch.tensor(out_theta, dtype=torch.float32)
    topics_matrix = torch.from_numpy(lda.get_topics()).float()
    
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(args.output_path, 'theta.pt'))
    torch.save(topics_matrix, os.path.join(args.output_path, 'topics.pt'))
    print(f"[train-lda] Saved inference results to {args.output_path}")



# -----------------------------
# Command: infer
# -----------------------------

def _dense_theta(doc_topics, K: int) -> List[float]:
    """Convert sparse list[(topic_id, prob)] to dense length-K list in topic id order."""
    theta = [0.0] * K
    for k, p in doc_topics:
        if 0 <= k < K:
            theta[k] = float(p)
    return theta

def cmd_infer(args):
    if (args.dataset is None) == (len(args.shards) == 0):
        print("[infer] Provide exactly one of: --dataset OR --shards ... --vocab", file=sys.stderr)
        sys.exit(2)

    # Load model
    lda: LdaModel = LdaModel.load(args.model)
    K = lda.num_topics
    print(f"[infer] Loaded LDA model with K={K} topics from {args.model}")

    if args.dataset:
        data = torch.load(args.dataset, map_location="cpu")
        vocab = Vocab.load(data["vocab"])
        bows = data["bow_counts"]
        ids = data["ids"]
        total = len(bows)
        print(f"[infer] Using merged dataset: {args.dataset} (docs={total})")
        iterator = zip(ids, bows)
    else:
        if not args.vocab:
            print("[infer] In --shards mode you must also pass --vocab.", file=sys.stderr)
            sys.exit(2)
        shard_paths: List[str] = []
        for pat in args.shards:
            shard_paths.extend(sorted(_glob.glob(pat)))
        if not shard_paths:
            print("[infer] No shard files matched the given patterns.", file=sys.stderr)
            sys.exit(2)
        vocab = _load_vocab_file(args.vocab)
        ids_iter = _iter_ids_from_shards(shard_paths)
        bows_iter = _iter_corpus_from_shards(shard_paths)  # yields sorted list[(term_id, count)]
        iterator = zip(ids_iter, bows_iter)
        total = None
        print(f"[infer] Streaming {len(shard_paths)} shards")

    # Prepare outputs
    out_theta: List[List[float]] = []

    # Infer
    for doc_id, bow in tqdm(iterator, desc="Inferring doc-topic distributions", total=total):
        # bow can be dict or list; ensure list of (id, count)
        if isinstance(bow, dict):
            bow = sorted(bow.items())
        # Full dense distribution
        doc_topics = lda.get_document_topics(bow, minimum_probability=0.0)
        theta = _dense_theta(doc_topics, K)
        out_theta.append(theta)

    # Convert to tensors
    theta_matrix = torch.tensor(out_theta, dtype=torch.float32)  # (N x K)
    
    # Get topic-word matrix from LDA model (K x V)
    topics_matrix = torch.from_numpy(lda.get_topics()).float()  # (K x V)
    
    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(theta_matrix, os.path.join(args.output_path, 'theta.pt'))
    torch.save(topics_matrix, os.path.join(args.output_path, 'topics.pt'))
    
    print(f"[infer] Saved topic proportions to: {os.path.join(args.output_path, 'theta.pt')}")
    print(f"[infer] Saved topics to: {os.path.join(args.output_path, 'topics.pt')}")
    print(f"[infer] N={len(out_theta)}  K={K}")

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Train Gensim LDA and infer doc-topic distributions from ETM-style caches.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    tl = sub.add_parser("train", help="Train LDA from a merged dataset or from shards+vocab.")
    tl.add_argument("--dataset", type=str, help="Merged BowDataset cache (.pt).")
    tl.add_argument("--shards", nargs="+", default=[], help="Shard paths or globs (e.g., cache/shards/shard_*.pt).")
    tl.add_argument("--vocab", type=str, help="Vocab file (.pt) used for shards (required for --shards).")
    tl.add_argument("--out-model", required=True, type=str, help="Where to save the Gensim LDA model (.gensim).")
    tl.add_argument("--output_path", required=True, type=str, help="Output directory for theta.pt and topics.pt")
    tl.add_argument("--num-topics", type=int, default=50)
    tl.add_argument("--chunksize", type=int, default=2000)
    tl.add_argument("--passes", type=int, default=1)
    tl.add_argument("--iterations", type=int, default=50)
    tl.add_argument("--update-every", type=int, default=1)
    tl.add_argument("--alpha", default="symmetric")
    tl.add_argument("--eta", default=None)
    tl.add_argument("--eval-every", type=int, default=10)
    tl.add_argument("--random-state", type=int, default=0)
    tl.add_argument("--seed", type=int, default=None)
    tl.set_defaults(func=cmd_train_lda)

    # Infer
    inf = sub.add_parser("infer", help="Infer & save document-topic distributions from a saved LDA model.")
    inf.add_argument("--model", required=True, type=str, help="Path to saved Gensim LDA model (.gensim).")
    inf.add_argument("--dataset", type=str, help="Merged BowDataset cache (.pt).")
    inf.add_argument("--shards", nargs="+", default=[], help="Shard paths or globs (e.g., cache/shards/shard_*.pt).")
    inf.add_argument("--vocab", type=str, help="Vocab file (.pt) used for shards (required for --shards).")
    inf.add_argument("--output_path", required=True, type=str, help="Output directory for theta.pt and topics.pt")

    inf.set_defaults(func=cmd_infer)

    return p

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
