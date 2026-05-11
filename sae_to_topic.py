import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import gensim.downloader as api
import logging # Added import

from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix, save_npz
from sklearn.cluster import KMeans
from collections import defaultdict

# --- Setup basic logging ---
# Messages will be printed to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def sparsify_and_renormalize(input_tensor: torch.Tensor, tau: float = 0.9) -> torch.Tensor:
    """
    Transforms a tensor by keeping only the n largest entries per row and renormalizing.

    For each row in the input tensor, this function determines the smallest number of
    largest entries, n, whose sum is greater than a threshold `tau`. It then sets all
    other entries in that row to zero and renormalizes the row so that its elements
    sum to 1.

    Args:
        input_tensor (torch.Tensor): A K x V tensor with non-negative entries where each
                                     row sums to 1.
        tau (float, optional): The cumulative sum threshold for determining n.
                               Defaults to 0.9.

    Returns:
        torch.Tensor: The transformed K x V tensor where only the n largest entries
                      per row are non-zero, and each row is renormalized to sum to 1.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but got {type(input_tensor)}")
    if input_tensor.dim() != 2:
        raise ValueError(f"Input tensor must be 2-dimensional, but got {input_tensor.dim()} dimensions")
    if not (0 < tau < 1):
        print(f"Warning: tau is {tau}, but it's typically expected to be between 0 and 1.")

    logging.info(f"[sparsify] Received input tensor with shape: {input_tensor.shape}")
    logging.info(f"[sparsify] Using tau threshold: {tau}")

    # Get the shape of the tensor
    K, V = input_tensor.shape
    device = input_tensor.device

    # --- Step 1: Sort the tensor's values in descending order along each row ---
    # We keep the original indices to reconstruct the tensor later.
    sorted_values, sorted_indices = torch.sort(input_tensor, dim=-1, descending=True)
    logging.debug(f"[sparsify] Sorted values shape: {sorted_values.shape}")

    # --- Step 2: Calculate the cumulative sum of the sorted values for each row ---
    cumulative_sums = torch.cumsum(sorted_values, dim=-1)
    logging.debug(f"[sparsify] Cumulative sums shape: {cumulative_sums.shape}")

    # --- Step 3: Determine n for each row ---
    # n is the minimum number of elements whose cumulative sum exceeds tau.
    # We find the index of the first element in each row that is > tau.
    # The number of elements to keep (n) is that index + 1.
    n_elements = torch.argmax((cumulative_sums > tau).int(), dim=-1) + 1

    # Edge case: If for some row, the total sum (which is 1.0) is not > tau,
    # argmax will return 0, making n_elements=1. A more robust behavior is to
    # keep all elements if the condition is never met.
    # This happens if tau >= 1.0
    never_exceeds_tau = (cumulative_sums > tau).sum(dim=-1) == 0
    n_elements[never_exceeds_tau] = V
    logging.info(f"[sparsify] Num elements to keep per row (n_elements): min={n_elements.min().item()}, max={n_elements.max().item()}, mean={n_elements.float().mean().item():.2f}")
    if never_exceeds_tau.sum() > 0:
        logging.warning(f"[sparsify] {never_exceeds_tau.sum().item()} rows never exceeded tau; keeping all {V} elements for them.")


    # --- Step 4: Create a mask to keep only the top n elements ---
    # We create a boolean mask based on the sorted tensor's shape.
    arange_tensor = torch.arange(V, device=device).expand(K, -1)
    mask_sorted = arange_tensor < n_elements.unsqueeze(-1)

    # --- Step 5: Un-sort the mask to match the original tensor's structure ---
    # We use scatter to place the `True` values from `mask_sorted` into a new
    # mask at the positions specified by `sorted_indices`.
    final_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    final_mask.scatter_(dim=1, index=sorted_indices, src=mask_sorted)
    logging.debug(f"[sparsify] Final mask shape: {final_mask.shape}")

    # --- Step 6: Apply the mask to the original tensor ---
    # This sets all elements not in the top n to zero.
    transformed_tensor = input_tensor * final_mask
    logging.debug(f"[sparsify] Transformed tensor shape: {transformed_tensor.shape}")

    # --- Step 7: Renormalize each row to sum to 1 ---
    row_sums = torch.sum(transformed_tensor, dim=-1, keepdim=True)

    # Add a small epsilon to the denominator to prevent division by zero,
    # though this is unlikely if the input tensor has positive entries.
    epsilon = 1e-9
    renormalized_tensor = transformed_tensor / (row_sums + epsilon)

    logging.info(f"[sparsify] Returning renormalized tensor with shape: {renormalized_tensor.shape}")
    return renormalized_tensor


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-results-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--topic-embedding-sparsity", type=float, default=0.9)
    parser.add_argument("--num-clusters", type=int, default=100)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    path_to_sae_results = args.sae_results_path

    logging.info(f"Script started with arguments: {args}")

    # Load word2vec embeddings
    logging.info("Loading 'word2vec-google-news-300' embeddings...")
    embeddings = api.load("word2vec-google-news-300")
    mean_embedding = embeddings.get_mean_vector(embeddings.key_to_index.keys())
    logging.info("Finished loading word2vec embeddings.")

    # Load topic -> word probabilities
    logging.info("Loading word emission probabilities...")
    # Load the file only once
    word_emission_data = torch.load(
        os.path.join(path_to_sae_results, "word_emission_probabilities.pt"), map_location="cpu"
    )
    # Assign the 'B' matrix to a clearly named variable
    B_all_features = word_emission_data["B"]
    logging.info(f"Loaded word emission matrix B with shape: {B_all_features.shape}")

    # Load SAE feature probabilities
    logging.info("Loading feature probabilities (theta_avg)...")
    feature_probabilities = torch.load(
        os.path.join(path_to_sae_results, "feature_probabilities.pt")
    )
    theta_avg = feature_probabilities["theta_avg"]
    logging.info(f"Loaded average feature probabilities theta_avg with shape: {theta_avg.shape}")

    # Load theta
    logging.info("Loading sparse document-feature matrix (theta_csr.npz)...")
    theta = load_npz(os.path.join(path_to_sae_results, "theta_csr.npz"))
    theta = theta.astype(np.float32)
    total = theta.shape[0]
    logging.info(f"Loaded theta matrix with shape: {theta.shape}, Total documents: {total}, Non-zero entries: {theta.nnz}")

    # Load vocabulary
    with open(args.vocab_path, "r") as f:
        vocab = json.load(f)
        idx2lemma = vocab["vocab"]["id2token"]
    logging.info(f"Loaded vocabulary with {len(idx2lemma)} lemmas.")
    
    # Select valid features
    logging.info("Filtering features based on theta_avg > 0...")
    valid_feature_mask = feature_probabilities["theta_avg"] > 0
    valid_feature_idx = valid_feature_mask.nonzero().squeeze(1)
    # Create a new variable for the filtered B matrix
    B_valid_features = B_all_features[valid_feature_mask]
    logging.info(f"Found {len(valid_feature_idx)} valid features. Filtered B shape: {B_valid_features.shape}")

    # Get lemma embeddings
    logging.info("Generating lemma embeddings...")
    lemmas = idx2lemma.copy()
    lemma_embeddings = []
    missing_lemmas = 0
    for lemma in lemmas:
        try:
            lemma_embeddings.append(embeddings[lemma])
        except KeyError:
            lemma_embeddings.append(mean_embedding)
            missing_lemmas += 1
    
    if missing_lemmas > 0:
        logging.warning(f"Could not find {missing_lemmas} lemmas in word2vec; used mean embedding as fallback.")

    lemma_embeddings = np.stack(lemma_embeddings)
    lemma_embeddings = torch.from_numpy(lemma_embeddings).float()
    logging.info(f"Created lemma embeddings tensor with shape: {lemma_embeddings.shape}")

    # Get topic embeddings
    logging.info(f"Generating topic embeddings using sparsify_and_renormalize (tau={args.topic_embedding_sparsity})...")
    with torch.no_grad():
        # Use the filtered B matrix
        topic_embeddings = sparsify_and_renormalize(B_valid_features, tau=args.topic_embedding_sparsity) @ lemma_embeddings
    logging.info(f"Generated topic embeddings with shape: {topic_embeddings.shape}")

    # Get topic weights
    weights = feature_probabilities["theta_avg"][valid_feature_idx]
    logging.info(f"Extracted topic weights with shape: {weights.shape}")

    # Cluster topic embeddings
    logging.info(f"Starting KMeans clustering with n_clusters={args.num_clusters}...")
    # reprs = topic_embeddings.cpu().numpy() # Removed this line
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    # Use topic_embeddings directly and rename the confusing 'clusters' output variable
    kmeans_distances = kmeans.fit_transform(
        topic_embeddings.cpu().numpy(), sample_weight=weights.cpu().numpy()
    )
    logging.info("KMeans clustering finished.")

    # Collect cluster information
    logging.info("Collecting cluster assignments and probabilities...")
    # Rename defaultdicts for clarity
    cluster_to_feature_indices = defaultdict(list)
    cluster_to_feature_weights = defaultdict(list)
    cluster_to_top_words = defaultdict(list)

    for k, label in enumerate(kmeans.labels_.tolist()):
        cluster_to_feature_indices[label].append(valid_feature_idx[k].item())
        cluster_to_feature_weights[label].append(weights[k].item())
    logging.info(f"Assigned {len(valid_feature_idx)} features to {len(cluster_to_feature_indices)} clusters.")

    logging.info("Generating top 50 words for each cluster...")
    # Use the new defaultdict names and the original B matrix
    for k, v in sorted(cluster_to_feature_indices.items(), key=lambda x: len(x[1]), reverse=True):
        avg_probs = B_all_features[v, :] # Use the original, unfiltered B matrix
        avg_thetas = feature_probabilities["theta_avg"][v].flatten()
        avg_probs = (avg_probs * avg_thetas.unsqueeze(1)).sum(dim=0) / avg_thetas.sum()
        n = 50
        top_n_words = torch.argsort(avg_probs, dim=-1, descending=True)[:n]
        top_n_words = [idx2lemma[i.item()] for i in top_n_words]
        cluster_to_top_words[k] = ", ".join(top_n_words)
    logging.info("Finished generating top words.")

    # Use the new defaultdict name
    cluster_total_probs = {k: np.sum(v).item() for k, v in cluster_to_feature_weights.items()}

    feature_idx_to_cluster = {
        valid_feature_idx[k].item(): label for k, label in enumerate(kmeans.labels_.tolist())
    }

    logging.info("Calculating cluster document coverage (this may take a while)...")
    cluster_counts = defaultdict(set)

    for idx, feature_idx in tqdm(zip(*theta.nonzero()), total=theta.nnz, desc="Counting doc coverage"):
        idx = idx.item()
        feature_idx = feature_idx.item()
        if feature_idx in feature_idx_to_cluster:
            cluster_counts[feature_idx_to_cluster[feature_idx]].add(idx)
    logging.info("Finished calculating document coverage.")

    cluster_records = []
    # Use the new defaultdict name
    for k in cluster_to_feature_indices.keys():
        cluster_records.append({
            "cluster_id": k,
            "cluster_size": len(cluster_to_feature_indices[k]), # Use new name
            "cluster_prob": cluster_total_probs[k], # Use new name
            "cluster_words": cluster_to_top_words[k], # Use new name
            "cluster_ratio": len(cluster_counts[k]) / total
        })

    cluster_df = pd.DataFrame(cluster_records)
    logging.info(f"Created cluster DataFrame with {len(cluster_df)} rows.")

    # Save cluster information
    logging.info(f"Saving cluster information to path: {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    
    csv_path = os.path.join(args.save_path, "clusters.csv")
    cluster_df.to_csv(csv_path, index=False)
    logging.info(f"Saved cluster CSV to {csv_path}")
    
    txt_path = os.path.join(args.save_path, "top_words.txt")
    with open(txt_path, "w") as f:
        for _, row in cluster_df.iterrows():
            f.write(f"{row['cluster_words']}\n")
    logging.info(f"Saved top words to {txt_path}")

    with open(os.path.join(args.save_path, "cluster_to_feature_indices.json"), "w") as f:
        json.dump(cluster_to_feature_indices, f)
    logging.info(f"Saved cluster to feature indices to {os.path.join(args.save_path, 'cluster_to_feature_indices.json')}")

    logging.info("Calculating and saving N x K aggregate topic matrix...")
    mapping_rows = valid_feature_idx.cpu().numpy()
    mapping_cols = kmeans.labels_
    mapping_data = np.ones_like(mapping_cols, dtype=np.float32)
    
    M = csr_matrix((mapping_data, (mapping_rows, mapping_cols)), shape=(theta.shape[1], args.num_clusters))
    theta_topic = theta.dot(M)
    
    topic_matrix_path = os.path.join(args.save_path, "theta_topic_csr.npz")
    save_npz(topic_matrix_path, theta_topic)
    logging.info(f"Saved aggregated topic matrix (shape: {theta_topic.shape}) to {topic_matrix_path}")

    logging.info("Script finished successfully.")

