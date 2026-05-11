#!/usr/bin/env python3
"""
Convert an S1-formatted BowDataset (.pt) to an S2-formatted sparse
dataset (_meta.json + _bow.npz).

This script loads the single, large .pt file created by 'make_bow_cache.py merge-shards'
(S1) and saves it as the two-file (metadata + sparse matrix) format
produced by 'merge_bow_to_sparse.py' (S2).

Example
-------
# Given the output from S1:
# - cache/bow_train_20000_5.pt

# Run this converter:
python convert_s1_to_s2.py \
    --input-pt cache/bow_train_20000_5.pt \
    --out-base cache/bow_train_sparse

# This will create the S2-compatible files:
# - cache/bow_train_sparse_meta.json
# - cache/bow_train_sparse_bow.npz
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import scipy.sparse
import torch
from tqdm import tqdm


def convert_s1_to_s2(args):
    """
    Loads S1-formatted .pt file and saves as S2-formatted .json + .npz files.
    """
    
    # --- 1. Define Output Paths ---
    base_out_path = args.out_base
    json_out_path = base_out_path + "_meta.json"
    sparse_out_path = base_out_path + "_bow.npz"
    
    os.makedirs(os.path.dirname(base_out_path) or ".", exist_ok=True)

    # --- 2. Load S1 Dataset ---
    print(f"Loading S1-formatted dataset from: {args.input_pt}...")
    try:
        s1_data = torch.load(args.input_pt, map_location="cpu")
    except Exception as e:
        print(f"Error: Failed to load input file {args.input_pt}. {e}", file=sys.stderr)
        sys.exit(1)

    # Extract components
    try:
        all_ids: List[str] = s1_data['ids']
        all_bows: List[Dict[int, float]] = s1_data['bow_counts']
        V: int = s1_data['vocab_size']
        vocab_dict: Dict = s1_data['vocab'] # This is the vocab.save() dict
        total_docs = len(all_ids)
        
        if len(all_bows) != total_docs:
            print(f"Warning: Mismatch in doc count. Found {total_docs} IDs and {len(all_bows)} BOW lists.", file=sys.stderr)

    except KeyError as e:
        print(f"Error: Input file {args.input_pt} is missing expected key: {e}", file=sys.stderr)
        print("Please ensure this file was created by S1's 'merge-shards' command.", file=sys.stderr)
        sys.exit(2)
        
    print(f"Loaded {total_docs} documents with vocab size {V}.")

    # --- 3. Build Sparse Matrix (CSR format) ---
    # This logic is copied directly from merge_bow_to_sparse.py (S2)
    print("Converting List[Dict] to CSR matrix...")
    
    sparse_data = []      # Non-zero values
    sparse_indices = []   # Column index for each non-zero value
    sparse_indptr = [0]   # Start/end marker for each row
    total_nnz = 0

    for bow_dict in tqdm(all_bows, desc="Building sparse matrix"):
        # Sort items by token_id (column index)
        # This is required for efficient CSR matrix construction
        sorted_items = sorted(bow_dict.items())
        
        for token_id, count in sorted_items:
            sparse_data.append(count)
            sparse_indices.append(token_id)
        
        # Update the index pointer
        total_nnz += len(sorted_items)
        sparse_indptr.append(total_nnz)

    print(f"Matrix construction complete. Total non-zero elements: {total_nnz}")

    # Create the final CSR matrix
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
        sys.exit(3)
        
    # --- 4. Save S2-formatted Outputs ---

    # 4a. Save the sparse matrix
    print(f"Saving sparse matrix to: {sparse_out_path}")
    try:
        scipy.sparse.save_npz(sparse_out_path, sparse_matrix)
    except Exception as e:
        print(f"Error: Failed to save sparse matrix. {e}", file=sys.stderr)
        sys.exit(4)

    # 4b. Save the metadata as JSON
    print(f"Saving metadata to: {json_out_path}")
    s2_metadata = {
        "ids": all_ids,
        "num_docs": total_docs,
        "vocab_size": V,
        "sparse_matrix_file": sparse_out_path, # Pointer to the matrix file
        "vocab": vocab_dict,                   # Pass the vocab dict directly
    }
    
    try:
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(s2_metadata, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save metadata JSON. {e}", file=sys.stderr)
        sys.exit(5)

    print("\nConversion complete!")
    print(f"  Metadata: {json_out_path}")
    print(f"  BOW Matrix: {sparse_out_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Convert S1-formatted BowDataset (.pt) to S2-formatted sparse dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    p.add_argument(
        "--input-pt", 
        required=True, 
        help="Input S1 dataset file (e.g., 'cache/bow_train.pt') from 'merge-shards'."
    )
    p.add_argument(
        "--out-base", 
        required=True, 
        help="Output *base path* for S2 dataset files. "
             "Will create '{path}_meta.json' and '{path}_bow.npz'."
    )
    p.set_defaults(func=convert_s1_to_s2)
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
