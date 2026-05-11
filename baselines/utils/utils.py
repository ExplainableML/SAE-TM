#!/usr/bin/env python3
"""
Shared utility functions for baseline topic models.
Consolidates common functionality for topic export, seeding, and other utilities.
"""
import io
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_topics(model, vocab, topics_tsv_path: str, topics_topn_txt_path: str, 
                  topn: int = 20, ranking: str = 'raw', 
                  model_type: str = 'generic'):
    """Export topic-word distributions to TSV and top-N text files.
    
    Unified topic export function that works across different model types.
    
    Args:
        model: Trained topic model with appropriate methods
        vocab: Vocab object with token mappings and statistics
        topics_tsv_path: Output path for full topic-word TSV file
        topics_topn_txt_path: Output path for top-N words per topic
        topn: Number of top words to show per topic
        ranking: Ranking method - 'raw', 'prob', 'pmi', or 'tfidf'
        model_type: Model type hint for extracting topic-word matrix
                   ('prodlda', 'dvae', 'etm', 'nvdm', 'fastopic', 'generic')
    
    The function expects model to have one of:
    - topic_word_matrix() method returning (V, K) tensor
    - decoder.weight of shape (V, K) (ProdLDA, DVAE)
    - get_beta() method returning (K, V) tensor (ETM, FASTopic)
    """
    import numpy as np
    
    model.eval()
    
    # Extract topic-word matrix based on model type/structure
    with torch.no_grad():
        if model_type == 'etm':
            # ETM: get_beta() returns (K, V), already softmax normalized
            beta = model.get_beta()  # (K, V)
            tw_prob = beta.t().cpu().numpy()  # (V, K)
            
            # For 'raw' ranking, extract unnormalized logits before softmax
            if hasattr(model, '_rho_is_linear') and model._rho_is_linear:
                VxE = model.rho.weight  # (V, E)
            elif hasattr(model, 'rho'):
                VxE = model.rho  # (V, E) tensor
            else:
                # Fallback: use beta as raw (not ideal but maintains compatibility)
                tw_raw = tw_prob
                VxE = None
            
            if VxE is not None:
                tw_raw = model.alphas(VxE).t().detach().cpu().numpy()  # (V, K)
            
        elif model_type == 'fastopic':
            # FASTopic: get_beta() returns (K, V)
            beta = model.get_beta()  # (K, V)
            tw_raw = beta.t().detach().cpu().numpy()  # (V, K)
            # For FASTopic, normalize across vocab per topic
            tw_prob = (beta.t() / beta.sum(dim=1, keepdim=True).t().clamp_min(1e-12)).cpu().numpy()
        elif hasattr(model, 'topic_word_matrix'):
            # Models with topic_word_matrix() method
            W = model.topic_word_matrix().T  # Should return (V, K)
            tw_raw = W.detach().cpu().numpy()
            tw_prob = torch.softmax(W, dim=0).cpu().numpy()
        elif hasattr(model, 'topic_word_weights'):
            # NVDM: topic_word_weights() returns (V, K)
            W = model.topic_word_weights().T
            tw_raw = W.detach().cpu().numpy()
            tw_prob = torch.softmax(W, dim=0).cpu().numpy()
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'weight'):
            # ProdLDA, DVAE: decoder.weight is (V, K)
            W = model.decoder.weight.T
            tw_raw = W.detach().cpu().numpy()
            tw_prob = torch.softmax(W, dim=0).cpu().numpy()
        elif hasattr(model, 'dec_fc') and hasattr(model.dec_fc, 'weight'):
            # Alternative decoder name
            W = model.dec_fc.weight.T
            tw_raw = W.detach().cpu().numpy()
            tw_prob = torch.softmax(W, dim=0).cpu().numpy()
        elif hasattr(model, 'proj') and hasattr(model.proj, 'weight'):
            # NVDM projection layer
            W = model.proj.weight.T
            tw_raw = W.detach().cpu().numpy()
            tw_prob = torch.softmax(W, dim=0).cpu().numpy()
        else:
            raise ValueError(
                f"Cannot extract topic-word matrix from model. "
                f"Model type: {type(model).__name__}. "
                f"Expected methods: topic_word_matrix(), get_beta(), or decoder.weight attribute."
            )
    
    V, K = tw_raw.shape
    eps = 1e-12
    
    # Compute corpus statistics for PMI and TF-IDF
    term_freq = np.asarray(vocab.term_freq, dtype=np.float64)
    p_w = term_freq / max(term_freq.sum(), 1.0)
    df = np.asarray(vocab.doc_freq, dtype=np.float64)
    idf = np.log((vocab.num_docs + 1.0) / (df + 1.0)) + 1.0
    
    # Select scoring matrix based on ranking method
    def score_matrix():
        if ranking == 'raw':
            return tw_raw
        elif ranking == 'prob':
            return tw_prob
        elif ranking == 'pmi':
            return np.log(tw_prob + eps) - np.log(p_w[:, None] + eps)
        elif ranking == 'tfidf':
            return tw_prob * idf[:, None]
        else:
            return tw_raw
    
    S = score_matrix()  # (V, K)
    
    # Write full TSV
    os.makedirs(os.path.dirname(topics_tsv_path) or '.', exist_ok=True)
    with io.open(topics_tsv_path, 'w', encoding='utf8') as f:
        f.write('topic_id\ttoken\tprob\tscore\n')
        for k in range(K):
            for vid in range(V):
                token = vocab.id2token[vid]
                prob = float(tw_prob[vid, k])
                score = float(S[vid, k])
                f.write(f"{k}\t{token}\t{prob:.8f}\t{score:.8f}\n")
    
    # Write top-N per topic
    os.makedirs(os.path.dirname(topics_topn_txt_path) or '.', exist_ok=True)
    with io.open(topics_topn_txt_path, 'w', encoding='utf8') as f:
        for k in range(K):
            top_idx = np.argsort(-S[:, k])[:topn]
            words = [vocab.id2token[i] for i in top_idx]
            f.write(f"Topic {k}: {' '.join(words)}\n")

