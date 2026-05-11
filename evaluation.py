#!/usr/bin/env python3
"""
Combined topic evaluation script.

Runs three analyses in a single process:
  1. Topic Diversity (Word Mover's Distance)
  2. Coherence Rating (LLM-based)
  3. Intruder Detection (LLM-based)

The vLLM model is loaded once and shared between coherence and intruder tasks.
Supports evaluating all topics under a root directory in one massive batch!
"""

import os
import sys
import json
import random
import argparse
import logging
import itertools
import glob

import numpy as np
import ot

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_keywords(filepath: str) -> list[list[str]]:
    """Load a file where each line is a comma-separated list of keywords."""
    lines_of_keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                keywords = [kw.strip() for kw in stripped.split(",")]
                lines_of_keywords.append(keywords)
    return lines_of_keywords


def get_output_path(source_path: str, topics_root: str, outputs_root: str, filename: str) -> str:
    """Computes the target path reflecting the nested directory structure."""
    if topics_root is None:
        # Fallback for single-dir mode
        return os.path.join(outputs_root, filename)
    rel_path = os.path.relpath(os.path.dirname(source_path), topics_root)
    target_dir = os.path.join(outputs_root, rel_path)
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

# ---------------------------------------------------------------------------
# 1.  Topic Diversity (WMD)
# ---------------------------------------------------------------------------

def load_topics_for_diversity(filepath: str, top_n: int) -> list[list[str]]:
    """Return top_n lower-cased words per line."""
    topics = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                keywords = [kw.strip().lower() for kw in stripped.split(",")]
                topics.append(keywords[:top_n])
    return topics


def calculate_wmd_diversity(lists_of_embeddings) -> float:
    """Average pairwise WMD across all topic pairs."""
    n_lists = len(lists_of_embeddings)
    if n_lists < 2:
        return 0.0

    total_dist = 0.0
    n_pairs = 0

    for i, j in itertools.combinations(range(n_lists), 2):
        emb_a, emb_b = lists_of_embeddings[i], lists_of_embeddings[j]
        na, nb = len(emb_a), len(emb_b)
        if na == 0 or nb == 0:
            continue

        try:
            arr_a = np.asarray(emb_a)
            arr_b = np.asarray(emb_b)
            if arr_a.ndim == 1:
                arr_a = arr_a.reshape(1, -1)
            if arr_b.ndim == 1:
                arr_b = arr_b.reshape(1, -1)
        except Exception as e:
            logging.warning(f"Embedding conversion error for pair ({i}, {j}): {e}")
            continue

        dist_a = np.ones(na) / na
        dist_b = np.ones(nb) / nb
        cost = ot.dist(arr_a, arr_b, metric="euclidean")

        try:
            wmd = ot.emd2(dist_a, dist_b, cost)
            total_dist += wmd
            n_pairs += 1
        except Exception as e:
            logging.warning(f"WMD computation error for pair ({i}, {j}): {e}")

    return total_dist / n_pairs if n_pairs else 0.0


def run_diversity(top_words_file: str, output_path: str, w2v, mean_emb) -> float:
    """Compute topic diversity via WMD and write a single-line JSONL result."""
    TOP_K = 20

    topics = load_topics_for_diversity(top_words_file, top_n=TOP_K)
    all_embs = []
    for words in topics:
        embs = []
        for w in words:
            try:
                embs.append(w2v[w])
            except KeyError:
                embs.append(mean_emb)
        all_embs.append(embs)

    score = calculate_wmd_diversity(all_embs)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"path": top_words_file, "result": {"diversity": score}}) + "\n")

    return score


# ---------------------------------------------------------------------------
# 2.  Coherence Rating  (LLM)
# ---------------------------------------------------------------------------

def create_coherence_prompt(word_list: list[str]) -> str:
    words_str = ", ".join(word_list)
    return (
        "You are an expert in semantics and lexical relationships. Your task is to evaluate "
        f"the coherence of the following list of words: '{words_str}'.\n\n"
        "Coherence is how well the words belong to a single, clear, and specific category.\n"
        "- A score of 100 means the words are extremely coherent (e.g., all are types of citrus fruits).\n"
        "- A score around 50 means the words are moderately coherent (e.g., all are 'vehicles' but mix cars, boats, and planes).\n"
        "- A score of 0 means the words are completely unrelated.\n\n"
        "Provide your analysis as a JSON object with two keys: \"rationale\" and \"score\".\n"
        "- \"rationale\": A brief, one-sentence explanation for your score.\n"
        "- \"score\": An integer between 0 and 100.\n\n"
        "Your response MUST be only the JSON object and nothing else."
    )


def build_coherence_prompts(top_words_file: str, k: int, r: int):
    """Return (prompts, eval_map) for coherence rating."""
    lines = load_keywords(top_words_file)
    prompts, emap = [], []

    for line_idx, kw_group in enumerate(lines):
        if len(kw_group) < k:
            logging.warning(f"Coherence: line {line_idx} has < {k} keywords – skipping.")
            continue
        pool = kw_group[:k]
        for _ in range(r):
            sample = random.sample(pool, k)
            random.shuffle(sample)
            prompts.append(create_coherence_prompt(sample))
            emap.append({"file_path": top_words_file, "line_idx": line_idx})

    return prompts, emap


def parse_coherence_results(outputs, emap, topics_root: str, outputs_root: str) -> None:
    results_by_file = {}

    for i, out in enumerate(outputs):
        info = emap[i]
        file_path = info["file_path"]
        line_idx = info["line_idx"]
        text = out.outputs[0].text.strip().replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
            score = data.get("score")
            rationale = data.get("rationale", "")
            if isinstance(score, int) and 0 <= score <= 100:
                results_by_file.setdefault(file_path, {})
                results_by_file[file_path].setdefault(line_idx, {"scores": [], "rationales": []})
                results_by_file[file_path][line_idx]["scores"].append(score)
                results_by_file[file_path][line_idx]["rationales"].append(rationale)
            else:
                logging.warning(f"Coherence: invalid score '{score}' at line {line_idx} in {file_path}.")
        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Coherence: JSON parse error at line {line_idx} in {file_path}. Text: '{text}'")

    for file_path, file_results in results_by_file.items():
        out_path = get_output_path(file_path, topics_root, outputs_root, "coherence.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"path": file_path, "result": file_results}) + "\n")

    logging.info(f"Coherence: wrote results for {len(results_by_file)} files.")


# ---------------------------------------------------------------------------
# 3.  Intruder Detection  (LLM)
# ---------------------------------------------------------------------------

def create_intruder_prompt(word_list: list[str]) -> str:
    words_str = ", ".join(word_list)
    return (
        "From the following list of words, identify the single word that does not belong "
        f"with the others. The words are: {words_str}. "
        "Your response must be only the single intruder word and nothing else."
    )


def build_intruder_prompts(top_words_file: str, k: int, n: int, r: int):
    """Return (prompts, eval_map) for intruder detection."""
    lines = load_keywords(top_words_file)
    num_lines = len(lines)
    prompts, emap = [], []

    if num_lines < 2:
        logging.warning(f"Intruder: need >= 2 lines to sample intruders – skipping {top_words_file}.")
        return prompts, emap

    for line_idx, kw_group in enumerate(lines):
        if len(kw_group) < k:
            logging.warning(f"Intruder: line {line_idx} in {top_words_file} has < {k} keywords – skipping.")
            continue
        if k < n:
            logging.warning(f"Intruder: k={k} < n={n} at line {line_idx} – skipping.")
            continue

        pool = kw_group[:k]
        for _ in range(r):
            intruder_line = random.choice([i for i in range(num_lines) if i != line_idx])
            intruder_word = random.choice(lines[intruder_line])
            base_words = random.sample(pool, n)
            test_words = base_words + [intruder_word]
            random.shuffle(test_words)

            prompts.append(create_intruder_prompt(test_words))
            emap.append({
                "file_path": top_words_file,
                "line_idx": line_idx,
                "intruder": intruder_word,
            })

    return prompts, emap


def parse_intruder_results(outputs, emap, topics_root: str, outputs_root: str) -> None:
    counts_by_file = {}

    for i, out in enumerate(outputs):
        info = emap[i]
        file_path = info["file_path"]
        line_idx = info["line_idx"]
        intruder = info["intruder"]
        predicted = out.outputs[0].text.strip().lower()

        counts_by_file.setdefault(file_path, {})
        counts_by_file[file_path].setdefault(line_idx, {"correct": 0, "total": 0})
        counts_by_file[file_path][line_idx]["total"] += 1
        if predicted == intruder.lower():
            counts_by_file[file_path][line_idx]["correct"] += 1

    for file_path, file_counts in counts_by_file.items():
        ratios = {
            idx: d["correct"] / d["total"]
            for idx, d in file_counts.items()
            if d["total"] > 0
        }
        if ratios:
            out_path = get_output_path(file_path, topics_root, outputs_root, "intruder.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"path": file_path, "result": ratios}) + "\n")

    logging.info(f"Intruder: wrote results for {len(counts_by_file)} files.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run topic diversity, coherence rating, and intruder detection in one go.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic-dir", type=str,
                        help="Path to single topic directory containing top_words.txt.")
    group.add_argument("--topics-root", type=str,
                        help="Root directory containing nested topic directories with top_words.txt.")
                        
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save the resulting .jsonl metrics (for --topic-dir).")
    parser.add_argument("--outputs-root", type=str,
                        help="Directory to save the resulting .jsonl metrics (for --topics-root).")
                        
    parser.add_argument("--k", type=int, default=5,
                        help="Number of initial keywords per line for coherence & intruder.")
    parser.add_argument("--n", type=int, default=4,
                        help="Keywords subsampled from the main group for intruder detection.")
    parser.add_argument("--r", type=int, default=3,
                        help="Repetitions per line (coherence & intruder).")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HuggingFace model for coherence and intruder detection.")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum sequence length for vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor-parallelism degree for vLLM.")

    args = parser.parse_args()

    is_batch_mode = args.topics_root is not None

    if is_batch_mode and not args.outputs_root:
        parser.error("--outputs-root is required when using --topics-root")
    if not is_batch_mode and not args.output_dir:
        parser.error("--output-dir is required when using --topic-dir")

    # Collect files
    top_words_files = []
    if is_batch_mode:
        topics_root = os.path.abspath(args.topics_root)
        outputs_root = os.path.abspath(args.outputs_root)
        pattern = os.path.join(topics_root, "**", "top_words.txt")
        top_words_files = glob.glob(pattern, recursive=True)
        if not top_words_files:
            logging.error(f"No top_words.txt files found under {topics_root}")
            sys.exit(1)
        logging.info(f"Found {len(top_words_files)} top_words.txt files to evaluate.")
    else:
        topic_dir = os.path.abspath(args.topic_dir)
        top_words_file = os.path.join(topic_dir, "top_words.txt")
        if not os.path.exists(top_words_file):
            logging.error(f"{top_words_file} does not exist.")
            sys.exit(1)
        top_words_files = [top_words_file]
        topics_root = None
        outputs_root = os.path.abspath(args.output_dir)
        os.makedirs(outputs_root, exist_ok=True)

    # ── 1. Diversity (no LLM needed) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Running Topic Diversity (WMD) for {len(top_words_files)} files …")
    print("=" * 60)
    
    logging.info("Loading word2vec embeddings for diversity …")
    w2v_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gensim", "w2v")
    embeddings = np.load(os.path.join(w2v_dir, "embeddings.np.npy"))
    with open(os.path.join(w2v_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
        vocabulary = json.load(f)
    w2v = {word: embeddings[idx] for idx, word in enumerate(vocabulary)}
    mean_emb = embeddings.mean(axis=0)
    del embeddings, vocabulary  # free memory
    
    for file in tqdm(top_words_files, desc="Diversity"):
        out_path = get_output_path(file, topics_root, outputs_root, "diversity.jsonl")
        run_diversity(file, out_path, w2v, mean_emb)

    # ── 2 & 3. Build all LLM prompts before loading the model ────────────
    print("\n" + "=" * 60)
    print("Generating coherence & intruder prompts …")
    print("=" * 60)

    all_coh_prompts, all_coh_map = [], []
    all_int_prompts, all_int_map = [], []

    for file in top_words_files:
        c_prompts, c_map = build_coherence_prompts(file, args.k, args.r)
        all_coh_prompts.extend(c_prompts)
        all_coh_map.extend(c_map)

        i_prompts, i_map = build_intruder_prompts(file, args.k, args.n, args.r)
        all_int_prompts.extend(i_prompts)
        all_int_map.extend(i_map)

    if not all_coh_prompts and not all_int_prompts:
        logging.error("No valid prompts generated for either task. Check k/n vs. input files.")
        sys.exit(1)

    # ── Apply chat template ──────────────────────────────────────────────
    logging.info(f"Applying chat template ({args.model}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def format_prompts(raw_prompts: list[str]) -> list[str]:
        formatted = []
        for p in tqdm(raw_prompts, desc="Formatting"):
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted

    coh_formatted = format_prompts(all_coh_prompts)
    int_formatted = format_prompts(all_int_prompts)

    # ── Load vLLM once ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Loading vLLM model: {args.model}")
    print("=" * 60)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )

    # ── Run coherence ────────────────────────────────────────────────────
    if coh_formatted:
        print("\n" + "=" * 60)
        print(f"Running Coherence Rating ({len(coh_formatted)} prompts) …")
        print("=" * 60)
        coh_params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)
        coh_outputs = llm.generate(coh_formatted, coh_params)
        parse_coherence_results(coh_outputs, all_coh_map, topics_root, outputs_root)

    # ── Run intruder detection ───────────────────────────────────────────
    if int_formatted:
        print("\n" + "=" * 60)
        print(f"Running Intruder Detection ({len(int_formatted)} prompts) …")
        print("=" * 60)
        int_params = SamplingParams(max_tokens=10, temperature=0.0)
        int_outputs = llm.generate(int_formatted, int_params)
        parse_intruder_results(int_outputs, all_int_map, topics_root, outputs_root)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if is_batch_mode:
        print(f"All evaluations completed for {len(top_words_files)} topic directories!")
        print(f"Nested results saved to: {outputs_root}")
    else:
        print("All evaluations completed!")
        print(f"Results saved to: {outputs_root}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()