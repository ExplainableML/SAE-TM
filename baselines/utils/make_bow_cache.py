#!/usr/bin/env python3
"""
Build & cache BowDataset shards compatible with the ETM script.

Workflow
--------
1) Build the vocabulary ONCE over the full corpus:
   python make_bow_cache.py build-vocab \
     --jsonl data/train.jsonl \
     --out-vocab cache/vocab.pt \
     --vocab-size 20000 \
     --min-df 5

2) Produce shards in parallel (each worker processes a disjoint slice):
   python make_bow_cache.py build-shard \
     --jsonl data/train.jsonl \
     --vocab cache/vocab.pt \
     --out-shard cache/shards/shard_0000.pt \
     --start 0 \
     --num-rows 100000

   python make_bow_cache.py build-shard \
     --jsonl data/train.jsonl \
     --vocab cache/vocab.pt \
     --out-shard cache/shards/shard_0001.pt \
     --start 100000 \
     --num-rows 100000
   ... (etc.)

3) Merge shards into a single BowDataset cache file for the ETM script:
   python make_bow_cache.py merge-shards \
     --vocab cache/vocab.pt \
     --out-dataset cache/bow_train_20000_5.pt \
     --shards cache/shards/shard_*.pt

Output
------
The merged dataset file matches the format used by BowDataset.save() in the ETM script:
  torch.save({
    'ids': [...],
    'bow_counts': [...],     # list of dict[int->float]
    'vocab_size': V,
    'vocab': vocab.save(),   # dict compatible with Vocab.load()
  }, path)

This lets the ETM script simply do:
  ds, vocab = BowDataset.load("cache/bow_train_20000_5.pt")
"""

from __future__ import annotations
import argparse
import collections
import dataclasses
import glob as _glob
import io
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple
import multiprocessing
import os  # (Already imported, just ensure it's there)
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

# ---- NLTK tokenization identical to the ETM script ----
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        print("Downloading NLTK 'punkt' model...", file=sys.stderr)
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except:
        print("Downloading NLTK 'stopwords'...", file=sys.stderr)
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except:
        print("Downloading NLTK 'wordnet'...", file=sys.stderr)
        nltk.download('wordnet')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except:
        print("Downloading NLTK 'averaged_perceptron_tagger'...", file=sys.stderr)
        nltk.download('averaged_perceptron_tagger')


class DocumentProcessor:
    """Tokenization, filtering, lemmatization (aligned with ETM script)."""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.wordnet_words = set(wordnet.words())
        self.ascii_pattern = re.compile(r'^[a-z]+$')
        # Cache for mapping NLTK tags to WordNet tags
        self.tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    def _get_pos(self, tag: str) -> str:
        """Map NLTK POS tag string to WordNet tag."""
        tag_char = tag[0].upper()
        return self.tag_dict.get(tag_char, wordnet.NOUN)

    def process(self, text: str) -> List[str]:
        lemmas = []
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Run POS-tagging ONCE for all tokens
        tagged_tokens = nltk.pos_tag(tokens)

        for token, tag in tagged_tokens:
            if (len(token) > 2 and
                    self.ascii_pattern.match(token) and
                    token not in self.stop_words and
                    token in self.wordnet_words):
                
                # Use the pre-computed tag to get POS
                lemma = self.lemmatizer.lemmatize(token, self._get_pos(tag))
                lemmas.append(lemma)
        return lemmas

# Global instance of DocumentProcessor
document_processor = DocumentProcessor()


def simple_tokenize(text: str) -> List[str]:
    return document_processor.process(text)


# ---- JSONL helpers ----

def read_jsonl_iter(path: str) -> Iterable[Dict]:
    """Yield JSONL records lazily."""
    with io.open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---- Vocab (binary-compatible with the ETM script) ----

@dataclasses.dataclass
class Vocab:
    token2id: Dict[str, int]
    id2token: List[str]
    doc_freq: List[int]
    term_freq: List[int]
    num_docs: int

    def save(self) -> Dict:
        return {
            "id2token": self.id2token,
            "doc_freq": self.doc_freq,
            "term_freq": self.term_freq,
            "num_docs": self.num_docs,
        }
    
    @staticmethod
    def load(obj: Dict) -> "Vocab":
        id2token = obj["id2token"]
        token2id = {t: i for i, t in enumerate(id2token)}
        return Vocab(
            token2id=token2id,
            id2token=id2token,
            doc_freq=obj.get("doc_freq", [1] * len(id2token)),
            term_freq=obj.get("term_freq", [1] * len(id2token)),
            num_docs=obj.get("num_docs", 1),
        )

    @staticmethod
    def build(docs: Iterable[str], max_size: int = 20000, min_df: int = 5) -> "Vocab":
        global_df = collections.Counter()
        global_tf = collections.Counter()
        global_num_docs = 0
        
        # --- Multiprocessing setup ---
        num_workers = 8
        # Chunk size is a tradeoff: too small=overhead, too large=stragglers
        chunk_size = 1000 
        doc_chunks = _chunk_iterator(docs, chunk_size)
        
        print(f"[Vocab.build] Starting parallel processing with {num_workers} workers (chunk size={chunk_size})...", file=sys.stderr)

        # Use initializer=download_nltk_data to ensure workers have the NLTK models
        with multiprocessing.Pool(num_workers, initializer=download_nltk_data) as pool:
            # Use imap_unordered for efficiency (order doesn't matter for counting)
            # We don't know the total number of chunks, so tqdm will just count chunks
            pbar = tqdm(pool.imap_unordered(_process_vocab_chunk, doc_chunks), 
                        desc="Building vocab (chunks)")
            
            # --- Reduce Step ---
            for local_df, local_tf, num_docs_in_chunk in pbar:
                global_df.update(local_df)
                global_tf.update(local_tf)
                global_num_docs += num_docs_in_chunk
                pbar.set_postfix({"docs_processed": f"{global_num_docs:,}"})

        print(f"[Vocab.build] Parallel processing complete. Processed {global_num_docs:,} docs.", file=sys.stderr)
        # --- End multiprocessing ---

        # The rest of the function is the final filter/sort step (fast)
        items = [(tok, global_df[tok], global_tf[tok]) for tok in global_df if global_df[tok] >= min_df]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:max_size]
        
        id2token = [tok for tok, _, _ in items]
        token2id = {t: i for i, t in enumerate(id2token)}
        doc_freq = [int(global_df[t]) for t in id2token]
        term_freq = [int(global_tf[t]) for t in id2token]
        
        return Vocab(token2id, id2token, doc_freq, term_freq, global_num_docs)


def _chunk_iterator(it: Iterable, size: int) -> Iterable[List]:
    """Yield successive chunks of size `size` from iterator `it`."""
    iterator = iter(it)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk

# This MUST be a top-level function for multiprocessing to "pickle" it
def _process_vocab_chunk(doc_chunk: List[str]) -> Tuple[collections.Counter, collections.Counter, int]:
    """Worker function to process a chunk of documents."""
    # Each worker initializes its own processor
    processor = DocumentProcessor()
    local_df = collections.Counter()
    local_tf = collections.Counter()
    num_docs = 0

    for doc in doc_chunk:
        num_docs += 1
        try:
            toks = processor.process(doc)
            local_tf.update(toks)
            local_df.update(set(toks))
        except Exception as e:
            # Log error for a bad doc but don't crash the whole chunk
            print(f"Warning: Failed to process a document. Error: {e}", file=sys.stderr)
    
    return local_df, local_tf, num_docs


# ---- Bow construction (per shard) ----

def _doc_to_bow_counts(text: str, vocab: Vocab) -> Dict[int, float]:
    toks = simple_tokenize(text)
    counts = collections.Counter(t for t in toks if t in vocab.token2id)
    return {vocab.token2id[t]: float(c) for t, c in counts.items()}


# ---- Commands ----

def cmd_build_vocab(args):
    download_nltk_data()
    os.makedirs(os.path.dirname(args.out_vocab) or ".", exist_ok=True)

    # Stream all documents for vocab building
    docs = (row.get("document", "") for row in read_jsonl_iter(args.jsonl))
    vocab = Vocab.build(docs, max_size=args.vocab_size, min_df=args.min_df)

    # Save vocab wrapper so it can be embedded later in merged dataset
    payload = {"vocab": vocab.save(), "meta": {"vocab_size": args.vocab_size, "min_df": args.min_df}}
    torch.save(payload, args.out_vocab)
    print(f"[build-vocab] Saved vocab to: {args.out_vocab} (|V|={len(vocab.id2token)})")


def _slice_jsonl(jsonl_path: str, start: int, num_rows: int) -> List[Dict]:
    """Read a slice [start, start+num_rows) from a JSONL file."""
    rows = []
    end = start + num_rows if num_rows >= 0 else None
    for i, row in enumerate(read_jsonl_iter(jsonl_path)):
        if i < start:
            continue
        if end is not None and i >= end:
            break
        rows.append(row)
    return rows


def cmd_build_shard(args):
    download_nltk_data()
    os.makedirs(os.path.dirname(args.out_shard) or ".", exist_ok=True)

    # Load vocab
    vocab_pack = torch.load(args.vocab, map_location="cpu")
    vocab = Vocab.load(vocab_pack["vocab"])
    V = len(vocab.id2token)

    rows = _slice_jsonl(args.jsonl, args.start, args.num_rows)
    if not rows:
        print(f"[build-shard] No rows in slice start={args.start} num_rows={args.num_rows}.", file=sys.stderr)
        sys.exit(0)

    ids = [r.get("id", str(i + args.start)) for i, r in enumerate(rows)]
    bow_counts = []
    for r in tqdm(rows, desc=f"Shard(start={args.start}, n={len(rows)})"):
        bow_counts.append(_doc_to_bow_counts(r.get("document", ""), vocab))

    shard_payload = {
        "ids": ids,
        "bow_counts": bow_counts,
        "vocab_size": V,
        # helpful for merge ordering & debugging
        "shard_meta": {"start": int(args.start), "count": int(len(ids)), "jsonl": args.jsonl},
    }
    torch.save(shard_payload, args.out_shard)
    print(f"[build-shard] Saved shard to: {args.out_shard}  (start={args.start}, count={len(ids)}, V={V})")


def _load_shard(path: str) -> Tuple[int, List[str], List[Dict[int, float]], int]:
    obj = torch.load(path, map_location="cpu")
    meta = obj.get("shard_meta", {})
    start = int(meta.get("start", 0))
    ids = obj["ids"]
    bow_counts = obj["bow_counts"]
    V = int(obj["vocab_size"])
    return start, ids, bow_counts, V


def cmd_merge_shards(args):
    os.makedirs(os.path.dirname(args.out_dataset) or ".", exist_ok=True)

    # Load vocab
    vocab_pack = torch.load(args.vocab, map_location="cpu")
    vocab_dict = vocab_pack["vocab"]
    vocab = Vocab.load(vocab_dict)
    V = len(vocab.id2token)

    # Resolve shard file list (supports glob)
    shard_paths: List[str] = []
    for pat in args.shards:
        shard_paths.extend(sorted(_glob.glob(pat)))
    if not shard_paths:
        print("[merge-shards] No shard files matched the given patterns.", file=sys.stderr)
        sys.exit(2)

    # Read & sort shards by their 'start' to preserve original order
    shards = []
    for p in shard_paths:
        start, ids, bows, v_shard = _load_shard(p)
        if v_shard != V:
            print(f"[merge-shards] Vocab size mismatch in shard {p}: shard V={v_shard}, expected V={V}", file=sys.stderr)
            sys.exit(3)
        shards.append((start, ids, bows, p))
    shards.sort(key=lambda x: x[0])

    # Concatenate
    all_ids: List[str] = []
    all_bows: List[Dict[int, float]] = []
    total_docs = 0
    for start, ids, bows, p in shards:
        all_ids.extend(ids)
        all_bows.extend(bows)
        total_docs += len(ids)

    # Final dataset payload (binary-compatible with ETM BowDataset.save)
    final_payload = {
        "ids": all_ids,
        "bow_counts": all_bows,
        "vocab_size": V,
        "vocab": vocab.save(),
    }
    torch.save(final_payload, args.out_dataset)

    print(f"[merge-shards] Merged {len(shards)} shards → {args.out_dataset}")
    print(f"[merge-shards] Total docs: {total_docs} |V|={V}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Create & cache BowDataset shards compatible with ETM.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) Build vocab
    pv = sub.add_parser("build-vocab", help="Scan entire JSONL to build vocab.")
    pv.add_argument("--jsonl", required=True, help="Input JSONL with {'id', 'document'} per line.")
    pv.add_argument("--out-vocab", required=True, help="Output vocab file (.pt).")
    pv.add_argument("--vocab-size", type=int, default=5000)
    pv.add_argument("--min-df", type=int, default=5)
    pv.set_defaults(func=cmd_build_vocab)

    # 2) Build shard from a slice
    ps = sub.add_parser("build-shard", help="Create a BOW shard for a slice of the corpus.")
    ps.add_argument("--jsonl", required=True, help="Input JSONL with {'id', 'document'} per line.")
    ps.add_argument("--vocab", required=True, help="Vocab file from build-vocab (.pt).")
    ps.add_argument("--out-shard", required=True, help="Output shard path (.pt).")
    ps.add_argument("--start", type=int, required=True, help="0-based starting line index in JSONL.")
    ps.add_argument("--num-rows", type=int, required=True, help="Number of rows to process (use -1 for to-the-end).")
    ps.set_defaults(func=cmd_build_shard)

    # 3) Merge shards
    pm = sub.add_parser("merge-shards", help="Merge multiple shards into a single BowDataset cache.")
    pm.add_argument("--vocab", required=True, help="Vocab file from build-vocab (.pt).")
    pm.add_argument("--out-dataset", required=True, help="Output dataset path (.pt) for ETM script.")
    pm.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="One or more shard paths or globs (e.g., cache/shards/shard_*.pt).",
    )
    pm.set_defaults(func=cmd_merge_shards)

    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
