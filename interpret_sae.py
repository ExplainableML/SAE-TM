import os
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from typing import Union, Optional
from scipy.sparse import load_npz, csr_matrix, save_npz, vstack
from train_sae_cached_ddp import BatchTopKSAE, AutoEncoderTopK
from libraries.dictionary_learning.dictionary import AutoEncoder


@torch.inference_mode()
def save_full_theta_csr(
    embeddings: torch.Tensor,
    sae,
    K: int,                                  # number of SAE features (columns)
    device: Union[str, torch.device],
    sae_dtype: torch.dtype,
    save_dir: str,
    batch_size: int = 8192,
    # ---- tunables ----
    topk_per_row: Optional[int] = 32,        # if SAE is truly Top-K per row, set k; else set to None
    values_dtype=np.float32,                 # np.float16 halves value memory if acceptable
    save_compressed: bool = False,           # False = faster save, larger file
    positive_only: bool = True,              # drop <=0 activations; set False if you need signed entries
):
    """
    Build CSR directly from dense activations returned by sae.encode.

    Two paths:
      - Top-K path (fastest): uses torch.topk to gather K entries/row, filters nonpositives, normalizes.
      - Generic path: extracts all nonzero (or positive) entries from the dense tensor without
        constructing a dense theta; normalization is done only for gathered entries.

    We avoid Python-scalar loops: we accumulate NumPy chunks and concatenate once.
    """
    assert values_dtype in (np.float16, np.float32)
    if positive_only:
        # We’ll treat activations <= 0 as zero when sparsifying.
        cmp_fn = torch.gt
    else:
        # Keep any nonzero entry.
        cmp_fn = torch.ne

    N = embeddings.shape[0]
    out_path = os.path.join(save_dir, "theta_csr.npz")
    os.makedirs(save_dir, exist_ok=True)

    try:
        sae.eval()
    except Exception:
        pass

    data_chunks: list[np.ndarray] = []
    indices_chunks: list[np.ndarray] = []
    indptr = [0]  # length N+1 in the end

    for start in tqdm(range(0, N, batch_size), desc="Building theta (CSR, dense SAE)"):
        end = min(start + batch_size, N)
        bsz = end - start

        # Dense activations on device
        emb = embeddings[start:end].to(device=device, dtype=sae_dtype, non_blocking=True)
        act = sae.encode(emb)                      # [bsz, K] dense
        act = act.to(torch.float32, copy=False)    # do math in fp32 for stable normalization

        # Row sums for normalization (treat nonpositive rows as empty)
        if positive_only:
            row_sums = torch.clamp_min(act.clamp_min_(0).sum(dim=1), 1e-8)  # sum of positives only
        else:
            row_sums = torch.clamp_min(act.abs().sum(dim=1), 1e-8)          # or abs-sum if keeping signed
        nonempty_rows = row_sums > 0

        if topk_per_row is not None:
            # ---- Top-K fast path on dense matrix ----
            # Take largest entries per row, then drop nonpositives if requested.
            vals_k, cols_k = torch.topk(act, k=topk_per_row, dim=1, largest=True, sorted=True)

            # Filter out rows that are effectively empty and (optionally) nonpositive entries
            if positive_only:
                pos_mask = vals_k > 0
            else:
                pos_mask = vals_k != 0

            # Also drop from rows with zero sum (to avoid uniform normalization / NaNs)
            if nonempty_rows.any():
                pos_mask &= nonempty_rows.view(-1, 1)

            # Gather final (row, col, val)
            nz = pos_mask.nonzero(as_tuple=False)
            if nz.numel() > 0:
                r = nz[:, 0]
                c = cols_k[pos_mask]
                v = vals_k[pos_mask]

                # Normalize only the selected values
                v = v / row_sums.index_select(0, r)

                # Ship to CPU NumPy as big blocks (no .tolist())
                indices_chunks.append(c.cpu().numpy().astype(np.int32, copy=False))
                data_chunks.append(v.cpu().numpy().astype(values_dtype, copy=False))

                # Per-row nnz counts for indptr
                counts = torch.bincount(r, minlength=bsz).cpu().numpy()
            else:
                counts = np.zeros(bsz, dtype=np.int64)

        else:
            # ---- Generic dense path: extract all (strictly) nonzero entries without building theta ----
            # Build a boolean mask at once; this scans K on GPU but only once.
            mask = cmp_fn(act, 0)
            if nonempty_rows.any():
                mask &= nonempty_rows.view(-1, 1)

            row_idx, col_idx = mask.nonzero(as_tuple=True)
            if row_idx.numel() > 0:
                # Values and per-row normalization
                vals = act[row_idx, col_idx]
                if positive_only:
                    # Normalize by positive-row-sum to match earlier behavior
                    vals = vals / row_sums.index_select(0, row_idx)
                else:
                    # Normalize by L1 (abs) to avoid zero-row division while keeping signs
                    vals = vals / row_sums.index_select(0, row_idx)

                indices_chunks.append(col_idx.cpu().numpy().astype(np.int32, copy=False))
                data_chunks.append(vals.cpu().numpy().astype(values_dtype, copy=False))

                counts = torch.bincount(row_idx, minlength=bsz).cpu().numpy()
            else:
                counts = np.zeros(bsz, dtype=np.int64)

        # Update indptr with cumulative counts from this batch
        base = indptr[-1]
        indptr.extend((base + np.cumsum(counts, dtype=np.int64)).tolist())

        # free promptly
        del emb, act, row_sums, nonempty_rows

    # ---- Finalize CSR ----
    if data_chunks:
        data = np.concatenate(data_chunks).astype(values_dtype, copy=False)
        indices = np.concatenate(indices_chunks).astype(np.int32, copy=False)
    else:
        data = np.array([], dtype=values_dtype)
        indices = np.array([], dtype=np.int32)

    # int32 is safe: max(indptr) == nnz, your target ~9.6e8 < 2,147,483,647
    indptr = np.asarray(indptr, dtype=np.int32)

    theta_csr = csr_matrix((data, indices, indptr), shape=(N, K))
    # If you suspect duplicate columns per row (shouldn't happen here), coalesce:
    # theta_csr.sum_duplicates()
    # Column order per row is nondecreasing due to topk(sorted=True); generic path preserves order of mask.nonzero.
    # If you need strictly sorted, call: theta_csr.sort_indices()

    save_npz(out_path, theta_csr, compressed=save_compressed)
    print(f"[Save] Full theta saved as CSR to {out_path} (compressed={save_compressed}, dtype={data.dtype})")


# ---------------------------
# Model: avoids materializing full B; supports subset column access
# ---------------------------
class TopicWordSubset(nn.Module):
    def __init__(self, K: int, V: int, dtype=torch.float32, init_pi=0.3):
        super().__init__()
        self.B_logits = nn.Parameter(0.01 * torch.randn(K, V, dtype=dtype))
        self.bg_logits = nn.Parameter(0.01 * torch.randn(V, dtype=dtype))  # background p0
        # unconstrained param; pass through sigmoid to get pi in (0,1)
        self.register_buffer("pi_logit", torch.tensor(np.log(init_pi/(1-init_pi)), dtype=dtype))

    def forward(
        self,
        theta_subset: torch.Tensor,
        bow_zero_cols_mask: torch.Tensor,
        theta_zero_cols_mask: torch.Tensor
    ):
        """
        Forward pass with subsetted tensors.
        - theta_subset: (B, K_active)
        - bow_zero_cols_mask: (V,) boolean mask (True where col is zero)
        - theta_zero_cols_mask: (K,) boolean mask (True where col is zero)
        """
        # 1. Select active topic rows from B_logits
        # B_logits is (K, V)
        B_logits_active_topics = self.B_logits[~theta_zero_cols_mask, :]  # (K_active, V)
        
        # 2. Compute log-normalizer over the *full* vocabulary (dim=1)
        # This is the stable way to get the denominator for softmax
        log_denominators = torch.logsumexp(B_logits_active_topics, dim=1, keepdim=True) # (K_active, 1)

        # 3. Slice the active vocab *columns* from the active topic logits
        # B_logits_active_topics is (K_active, V)
        # ~bow_zero_cols_mask is (V_active,)
        B_logits_subset = B_logits_active_topics[:, ~bow_zero_cols_mask] # (K_active, V_active)

        # 4. Compute the subsetted softmax
        # B_subset = exp(logits_subset) / exp(log_denominators)
        # B_subset = exp(logits_subset - log_denominators)
        B_subset = (B_logits_subset - log_denominators).exp() # (K_active, V_active)
        # --- End B_subset calculation ---

        # theta_subset is already (B, K_active)
        main = theta_subset @ B_subset                             # (B, V_active)

        # Also apply the same logic to the background p0
        # bg_logits is (V,)
        
        # 1. Compute log-normalizer over full vocab (scalar)
        p0_log_denominator_scalar = torch.logsumexp(self.bg_logits, dim=0) # scalar
        
        # 2. Slice active vocab columns
        bg_logits_subset = self.bg_logits[~bow_zero_cols_mask]     # (V_active,)

        # 3. Compute subsetted softmax
        p0_subset = (bg_logits_subset - p0_log_denominator_scalar).exp()[None, :] # (1, V_active)

        # Combine main and background probabilities
        pi = torch.sigmoid(self.pi_logit)                    # scalar in (0,1)
        return (1 - pi) * main + pi * p0_subset              # (B, V_active)


# ---------------------------
# Dataset: index-only to minimize per-item overhead
# ---------------------------
class SAEInterpretationDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, bow: csr_matrix):
        self.embeddings = embeddings
        self.bow = bow
        assert embeddings.shape[0] == bow.shape[0], "Number of embeddings and bow rows must match"
        self.N = embeddings.shape[0]
        self.V = bow.shape[1]
        self.d = embeddings.shape[1]

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, csr_matrix]:
        embeddings_i = self.embeddings[i]
        bow_i = self.bow.getrow(i) # Return the (1, V) sparse row
        return embeddings_i, bow_i


def collate_fn(batch: list[tuple[torch.Tensor, csr_matrix]]) -> tuple[torch.Tensor, csr_matrix]:
    embeddings = torch.stack([emb for emb, _ in batch])  # (B, D)
    bow_sparse_rows = [bow for _, bow in batch]
    bow_sparse_batch = vstack(bow_sparse_rows) # (B, V) as CSR
    return embeddings, bow_sparse_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sae-type", type=str, required=True, choices=["BatchTopKTrainer", "TopKTrainer", "StandardTrainer"])
    parser.add_argument("--embeddings", type=str, nargs="+", required=True,
                        help="One or more file paths to the embeddings (accepts multiple).")
    
    parser.add_argument("--bow-dataset", type=str, required=True,
                        help="Path to the BowDataset cache file (.json) from make_bow_cache.py (S1).")
    parser.add_argument("--idf-weighting", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of epochs to train the topic-word matrix")
    parser.add_argument("--save-path", type=str, required=True,
                    help="Directory to save outputs: feature_probabilities.pt (avg theta) "
                         "and word_emission_probabilities.pt (KxV word emissions)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-full-theta", action="store_true",
                    help="If set, stream the full N x K theta to save_path/theta_csr.npz as CSR.")
    parser.add_argument("--theta-batch-size", type=int, default=8192)
    parser.add_argument("--init-pi", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- Load SAE checkpoint -----
    assert args.checkpoint.endswith(".pt")
    if args.sae_type == "BatchTopKTrainer":
        sae = BatchTopKSAE.from_pretrained(args.checkpoint)
    elif args.sae_type == "TopKTrainer":
        sae = AutoEncoderTopK.from_pretrained(args.checkpoint)
    elif args.sae_type == "StandardTrainer":
        sae = AutoEncoder.from_pretrained(args.checkpoint)
    else:
        raise ValueError(
            f"Invalid sae-type: {args.sae_type}. "
            f"Expected one of ['BatchTopKTrainer', 'TopKTrainer', 'StandardTrainer']."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    sae = sae.to(device=device, dtype=sae_dtype)
    sae.eval()

    # ----- Load inputs -----
    print("Loading embeddings...")
    embeddings_list = [torch.load(path, map_location="cpu") for path in args.embeddings]
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.to(dtype=torch.float16) # Store on CPU in half precision
    
    print(f"Loading BowDataset from: {args.bow_dataset}")
    with open(args.bow_dataset, "r") as reader:
        bow_data = json.load(reader)
    
    sparse_matrix_file = bow_data["sparse_matrix_file"]
    bow = load_npz(sparse_matrix_file)

    assert isinstance(bow, csr_matrix), "BoW must be CSR"
    assert embeddings.shape[0] == bow.shape[0], \
        f"N mismatch: embeddings={embeddings.shape[0]} vs bow={bow.shape[0]}"

    N, V = bow.shape
    print(f"[Info] Dataset: N={N:,} docs, V={V:,} vocab")
    print(f"[Info] Total tokens (nnz sum): {float(bow.sum()):,.0f}")

    document_frequency = torch.tensor(bow_data["vocab"]["doc_freq"]).float()
    idf = torch.log(N / document_frequency)
    # Normalize idf to be in the range [0, 1]
    idf = idf / idf.max()
    idf = idf.to(device=device, dtype=torch.float32)

    # Probe K from SAE quickly (encode a tiny batch)
    with torch.no_grad():
        probe_emb = embeddings[:1].to(device=device, dtype=sae_dtype)
        probe_act = sae.encode(probe_emb).float()
        K = probe_act.shape[1]
    print(f"[Info] Topics (K) inferred from SAE: K={K:,}")

    # ----- Build loader that yields indices, collate constructs theta and sparse batch -----
    ds = SAEInterpretationDataset(embeddings, bow)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    # ----- Model & optimizer -----
    model = TopicWordSubset(K, V, init_pi=args.init_pi, dtype=torch.float32).to(device)
    # Disable weight decay on logits
    decay_params = [p for n, p in model.named_parameters() if n not in ("B_logits", "bg_logits")]
    optimizer = optim.AdamW(
        [
            {"params": [model.B_logits, model.bg_logits], "weight_decay": 0.0},
            {"params": decay_params, "weight_decay": 1e-5},
        ],
        lr=args.lr,
    )

    # ----- Training -----
    print(f"[Train] epochs={args.num_epochs}  batch_size={args.batch_size}  lr={args.lr}")
    num_epochs = args.num_epochs
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        seen_tokens = 0.0
        batches = 0
        empty_batches = 0

        # Only accumulate average theta on the last epoch
        if epoch == num_epochs:
            theta_sum = torch.zeros(K, device=device, dtype=torch.float32)
            theta_count = 0

        progress_bar = tqdm(loader, desc=f"Training (epoch {epoch}/{num_epochs})", leave=False)
        for emb_batch, bow_batch in progress_bar:
            
            # --- MODIFIED BATCH SUBSETTING (Sparse BoW) ---
            # emb_batch is a tensor, bow_batch is a (B, V) scipy.sparse.csr_matrix
            
            # 1. Process bow on CPU using sparse methods
            bow_col_sums = bow_batch.sum(axis=0)  # This is a (1, V) np.matrix
            bow_zero_cols_mask_cpu = (np.asarray(bow_col_sums).flatten() == 0) # (V,) boolean mask
            
            # If all columns were zero, this batch has no tokens. Skip.
            if bow_zero_cols_mask_cpu.all():
                empty_batches += 1
                continue

            # 2. Create the sparse subset, *then* make it dense
            bow_subset_sparse = bow_batch[:, ~bow_zero_cols_mask_cpu] # (B, V_active)
            bow_subset_array = bow_subset_sparse.toarray() # (B, V_active) as np.array

            # 3. Move only the small, dense subset and mask to GPU
            bow_subset = torch.from_numpy(bow_subset_array).to(device=device, dtype=torch.float32, non_blocking=True)
            bow_zero_cols_mask = torch.from_numpy(bow_zero_cols_mask_cpu).to(device=device, non_blocking=True)
            
            if args.idf_weighting:
                idf_subset = idf[~bow_zero_cols_mask]
                bow_subset_idf = bow_subset * idf_subset.unsqueeze(0) # (B, V_active)
            else:
                bow_subset_idf = bow_subset # (B, V_active)
            
            # --- END MODIFICATION ---

            # 2. Process theta (SAE activations)
            with torch.no_grad():  # Don't train SAE
                theta = sae.encode(emb_batch.to(device=device, dtype=sae_dtype, non_blocking=True)).float()  # (B, K)
            
            # Normalize theta, handling all-zero rows to prevent NaN
            theta_sums = theta.sum(dim=1, keepdim=True)
            theta = torch.where(theta_sums > 0, theta / theta_sums.clamp_min(1e-8), torch.zeros_like(theta)) # (B, K)

            # Find all-zero columns (features) in this theta batch
            theta_zero_cols_mask = (theta.sum(dim=0) == 0)  # (K,)
            # Create subset by selecting non-zero columns
            theta_subset = theta[:, ~theta_zero_cols_mask]   # (B, K_active)

            # If all features were zero, this batch has no activations. Skip.
            if theta_subset.numel() == 0:
                empty_batches += 1
                if epoch == num_epochs:  # Still count docs for avg theta
                    theta_count += theta.size(0)
                continue
            
            # 3. Forward pass with subsets (use model(...) rather than .forward)
            q_subset = model(
                theta_subset,         # (B, K_active)
                bow_zero_cols_mask,   # (V,)
                theta_zero_cols_mask  # (K,)
            )  # (B, V_active)
            
            q_subset = torch.clamp(q_subset, min=1e-8)

            # 4. Loss (use subsetted bow)
            # Accumulate total NLL; normalize by total tokens at the end of the epoch.
            loss = (bow_subset_idf * (-q_subset.log())).sum()
            
            # --- END MODIFICATION ---

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Accumulate *full* theta regardless of token count for avg
            if epoch == num_epochs:
                theta_sum += theta.sum(dim=0) # use full theta (B, K)
                theta_count += theta.size(0)

            running_loss += loss.item()
            seen_tokens += bow_subset.sum().item()  # Count tokens from subset
            batches += 1

            progress_bar.set_postfix_str(f"loss={running_loss / max(batches, 1):.4f}")

        progress_bar.close()

        epoch_time = time.time() - t0
        nll_per_tok = running_loss / max(seen_tokens, 1.0)
        ppl = float(torch.exp(torch.tensor(nll_per_tok)))
        toks_per_sec = seen_tokens / max(epoch_time, 1e-6)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.num_epochs:
            print(
                f"[Epoch {epoch:03d}] nll/token={nll_per_tok:.4f}  ppl={ppl:.3f}  "
                f"tokens={int(seen_tokens):,}  time={epoch_time:.1f}s  tok/s={toks_per_sec:,.0f}  "
                f"empty_batches={empty_batches}"
            )

    # After loop completes, if we accumulated, compute avg theta
    theta_avg_cpu = None
    if num_epochs >= 1 and 'theta_sum' in locals() and theta_count > 0:
        # If the last epoch ran, these exist
        theta_avg = (theta_sum / float(theta_count)).clamp_min(0)
        # Re-normalize just in case of tiny numerical drift (should be ~1 already)
        theta_avg = theta_avg / torch.clamp(theta_avg.sum(), min=1e-8)
        theta_avg_cpu = theta_avg.detach().cpu()
        print(f"[Stats] feature_probabilities (avg theta) sum={float(theta_avg_cpu.sum()):.6f}")

    # ----- (Optional) Save full theta as sparse CSR -----
    if args.save_full_theta:
        print("[Save] Starting full theta CSR generation...")
        # Re-use SAE and embeddings for a simple sequential pass
        save_full_theta_csr(
            embeddings=embeddings,  # CPU half precision
            sae=sae,                # already on device & dtype
            K=K,
            device=device,
            sae_dtype=sae_dtype,
            save_dir=args.save_path,
            batch_size=args.theta_batch_size,
            topk_per_row=None,
            values_dtype=np.float16,
            save_compressed=True,
            positive_only=True,
        )

    # ----- Save outputs -----
    save_dir = args.save_path  # treat as directory
    os.makedirs(save_dir, exist_ok=True)

    # 3a) Save average theta (feature probabilities across documents)
    if theta_avg_cpu is not None:
        fp_path = os.path.join(save_dir, "feature_probabilities.pt")
        torch.save({"theta_avg": theta_avg_cpu, "K": int(K)}, fp_path)
        print(f"[Save] feature_probabilities.pt -> {fp_path}  | shape={tuple(theta_avg_cpu.shape)}")

    # 3b) Save learned B (topic→word emission probabilities) as a single materialization on CPU
    print("[Save] Materializing B to CPU and saving word emission probabilities...")
    with torch.no_grad():
        model.cpu() # Move model to CPU to free GPU memory
        K_full, V_full = model.B_logits.shape
        B_cpu = torch.softmax(model.B_logits, dim=1) # Materialize full (K, V) on CPU

        wep_path = os.path.join(save_dir, "word_emission_probabilities.pt")
        p0_cpu = torch.softmax(model.bg_logits, dim=0)
        pi_cpu = float(torch.sigmoid(model.pi_logit))
        
        torch.save({
            "B": B_cpu, 
            "p0": p0_cpu, 
            "K": int(K_full), 
            "V": int(V_full), 
            "pi": pi_cpu
        }, wep_path)
        print(f"[Save] word_emission_probabilities.pt -> {wep_path}")


if __name__ == "__main__":
    main()