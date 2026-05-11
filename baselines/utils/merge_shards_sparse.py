#!/usr/bin/env python3
"""
Merge pre-built BOW shards into a final dataset, saving the BOW counts
as a sparse matrix (scipy.sparse.csr_matrix) and the metadata as JSON.

This script replaces the 'merge-shards' command from the original
'make_bow_cache.py' script to provide a more memory-efficient
dataset format for inference.

Workflow
--------
This script assumes steps 1 (build-vocab-shard, merge-vocab-shards) and
2 (build-shard) from 'make_bow_cache.py' have already been run.

It replaces step 3:

   python merge_bow_to_sparse.py \
     --vocab cache/vocab_20000_5.pt \
     --out-dataset cache/bow_train_sparse \
     --shards "cache/bow_shards/shard_*.pt"

This will create two files:
1. cache/bow_train_sparse_meta.json (Vocab, doc IDs, etc.)
2. cache/bow_train_sparse_bow.npz   (The [N_docs, V_size] sparse matrix)
"""
from __future__ import annotations
import argparse
import dataclasses
import glob as _glob
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
from tqdm import tqdm


# ---- Vocab Class (copied from original script for compatibility) ----
# This is required to load the .pt vocab file correctly.

@dataclasses.dataclass
class Vocab:
    token2id: Dict[str, int]
    id2token: List[str]
    doc_freq: List[int]
    term_freq: List[int]
    num_docs: int

    def save(self) -> Dict:
        """Saves vocab to a JSON-serializable dict."""
        return {
            "id2token": self.id2token,
            "doc_freq": self.doc_freq,
            "term_freq": self.term_freq,
            "num_docs": self.num_docs,
        }

    @staticmethod
    def load(obj: Dict) -> "Vocab":
        """Loads vocab from a dict (e.g., from torch.load)."""
        id2token = obj["id2token"]
        token2id = {t: i for i, t in enumerate(id2token)}
        return Vocab(
            token2id=token2id,
            id2token=id2token,
            doc_freq=obj.get("doc_freq", [1] * len(id2token)),
            term_freq=obj.get("term_freq", [1] * len(id2token)),
            num_docs=obj.get("num_docs", 1),
        )

# ---- Shard Loader (copied from original script for compatibility) ----
# This is required to load the .pt shard files correctly.

def _load_shard(path: str) -> Tuple[int, List[str], List[Dict[int, float]], int]:
    """Loads a single BOW shard .pt file."""
    obj = torch.load(path, map_location="cpu")
    meta = obj.get("shard_meta", {})
    start = int(meta.get("start", 0))
    ids = obj["ids"]
    bow_counts = obj["bow_counts"]
    V = int(obj["vocab_size"])
    return start, ids, bow_counts, V


# ---- Main Command ----

def cmd_merge_shards_sparse(args):
    """
    (Serial) Merges final BOW shards into a JSON metadata file and a
    scipy.sparse.csr_matrix file.
    """
    # Use the --out-dataset arg as a base path for the two output files
    base_out_path = args.out_dataset
    json_out_path = base_out_path + "_meta.json"
    sparse_out_path = base_out_path + "_bow.npz"
    
    os.makedirs(os.path.dirname(base_out_path) or ".", exist_ok=True)

    # Load final vocab (from .pt file)
    print(f"Loading vocab from {args.vocab}...")
    try:
        vocab_pack = torch.load(args.vocab, map_location="cpu")
        vocab_dict = vocab_pack["vocab"]
        vocab = Vocab.load(vocab_dict)
        V = len(vocab.id2token)
    except Exception as e:
        print(f"Error: Failed to load vocab file {args.vocab}. {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve shard file list (supports glob)
    shard_paths: List[str] = []
    for pat in args.shards:
        shard_paths.extend(sorted(_glob.glob(pat)))
    if not shard_paths:
        print("[merge-shards] No shard files matched the given patterns.", file=sys.stderr)
        sys.exit(2)

    # Read & sort shards by their 'start' to preserve original order
    shards = []
    print(f"Loading {len(shard_paths)} BOW shards...")
    for p in tqdm(shard_paths, desc="Loading shards"):
        try:
            start, ids, bows, v_shard = _load_shard(p)
            if v_shard != V:
                print(f"[merge-shards] Vocab size mismatch in shard {p}: shard V={v_shard}, expected V={V}", file=sys.stderr)
                sys.exit(3)
            shards.append((start, ids, bows, p))
        except Exception as e:
            print(f"Warning: Failed to load shard {p}. Skipping. Error: {e}", file=sys.stderr)
            
    shards.sort(key=lambda x: x[0])
    print("All shards loaded. Concatenating and building sparse matrix...")

    # --- Build Sparse Matrix (CSR format) ---
    # We build the (data, indices, indptr) arrays directly,
    # which is the most efficient way to create a csr_matrix.
    
    all_ids: List[str] = []
    
    # These lists will store the components of the sparse matrix
    sparse_data = []      # Non-zero values
    sparse_indices = []   # Column index for each non-zero value
    sparse_indptr = [0]   # Start/end marker for each row
    
    total_nnz = 0
    total_docs = 0

    for start, ids, bows, p in tqdm(shards, desc="Concatenating shards"):
        all_ids.extend(ids)
        total_docs += len(ids)
        
        # 'bows' is a List[Dict[int, float]]
        for bow_dict in bows:
            # Sort items by token_id (column index)
            # This is required for efficient CSR matrix construction
            sorted_items = sorted(bow_dict.items())
            
            for token_id, count in sorted_items:
                sparse_data.append(count)
                sparse_indices.append(token_id)
            
            # Update the index pointer
            total_nnz += len(sorted_items)
            sparse_indptr.append(total_nnz)

    print(f"Concatenation complete. Total docs: {total_docs}. Total non-zero elements: {total_nnz}")

    # Create the final CSR matrix
    print("Creating SciPy CSR matrix...")
    try:
        data_np = np.array(sparse_data, dtype=np.float32)
        indices_np = np.array(sparse_indices, dtype=np.int32)
        indptr_np = np.array(sparse_indptr, dtype=np.int64)
        
        shape = (total_docs, V)
        
        sparse_matrix = scipy.sparse.csr_matrix(
            (data_np, indices_np, indptr_np), 
            shape=shape
        )
    except Exception as e:
        print(f"Error: Failed to create sparse matrix. {e}", file=sys.stderr)
        sys.exit(4)
        
    # --- Save Outputs ---

    # 1. Save the sparse matrix
    print(f"Saving sparse matrix to: {sparse_out_path}")
    try:
        scipy.sparse.save_npz(sparse_out_path, sparse_matrix)
    except Exception as e:
        print(f"Error: Failed to save sparse matrix. {e}", file=sys.stderr)
        sys.exit(5)

    # 2. Save the metadata as JSON
    print(f"Saving metadata to: {json_out_path}")
    final_payload = {
        "ids": all_ids,
        "num_docs": total_docs,
        "vocab_size": V,
        "sparse_matrix_file": sparse_out_path, # Pointer to the matrix file
        "vocab": vocab.save(),
    }
    
    try:
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(final_payload, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save metadata JSON. {e}", file=sys.stderr)
        sys.exit(6)

    print("\nMerge complete!")
    print(f"  Metadata: {json_out_path}")
    print(f"  BOW Matrix: {sparse_out_path}")
    print(f"  Total docs: {total_docs} | Vocab size: {V}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Merge BOW shards into a sparse matrix (.npz) and metadata (.json).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # These arguments are identical to the original 'merge-shards' command
    p.add_argument("--vocab", required=True, help="*Final* vocab file from merge-vocab-shards (.pt).")
    p.add_argument(
        "--out-dataset", 
        required=True, 
        help="Output *base path* for dataset files. "
             "Will create '{path}_meta.json' and '{path}_bow.npz'."
    )
    p.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="One or more BOW shard paths or globs (e.g., cache/bow_shards/shard_*.pt).",
    )
    p.set_defaults(func=cmd_merge_shards_sparse)
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
