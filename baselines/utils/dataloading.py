#!/usr/bin/env python3
"""
Shared data loading utilities for baseline topic models.
Consolidates common functionality for tokenization, vocab building, and BOW dataset creation.
"""
from __future__ import annotations
import collections
import dataclasses
import io
import json
import os
import re
import sys
from typing import Iterable, List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# NLTK Data Download
# -----------------------------

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


# -----------------------------
# Tokenization
# -----------------------------

def get_wordnet_pos(word):
    """Map NLTK part-of-speech tags to tags understood by WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class DocumentProcessor:
    """A class to handle tokenization, filtering, and lemmatization of documents."""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.wordnet_words = set(wordnet.words())
        self.ascii_pattern = re.compile(r'^[a-z]+$')

    def process(self, text):
        lemmas = []
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if (len(token) > 2 and
                    self.ascii_pattern.match(token) and
                    token not in self.stop_words and
                    token in self.wordnet_words):
                lemma = self.lemmatizer.lemmatize(token, get_wordnet_pos(token))
                lemmas.append(lemma)
        return lemmas


def simple_tokenize(text: str, lowercase: bool=True, remove_stopwords: bool=True) -> List[str]:
    """Wrapper for backward compatibility - uses DocumentProcessor."""
    processor = DocumentProcessor()
    return processor.process(text)


# -----------------------------
# JSONL I/O
# -----------------------------

def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file and return list of records."""
    rows = []
    with io.open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, records: Iterable[Dict]):
    """Write list of records to JSONL file."""
    with io.open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Vocabulary
# -----------------------------

@dataclasses.dataclass
class Vocab:
    """Vocabulary with token mappings and frequency statistics."""
    token2id: Dict[str, int]
    id2token: List[str]
    doc_freq: List[int]
    term_freq: List[int]
    num_docs: int

    @staticmethod
    def build(docs: Iterable[str], max_size: int=20000, min_df: int=5, show_progress: bool=False) -> "Vocab":
        """Build vocabulary from documents.
        
        Args:
            docs: Iterable of document strings
            max_size: Maximum vocabulary size
            min_df: Minimum document frequency for a token to be included
            show_progress: Whether to show progress bar
        """
        df = collections.Counter()
        tf = collections.Counter()
        num_docs = 0
        
        docs_iter = tqdm(docs, desc='Building vocab') if show_progress else docs
        for doc in docs_iter:
            num_docs += 1
            toks = simple_tokenize(doc)
            tf.update(t for t in toks)
            df.update(set(toks))
        
        items = [(tok, df[tok], tf[tok]) for tok in df if df[tok] >= min_df]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:max_size]
        id2token = [tok for tok, _, _ in items]
        token2id = {t:i for i,t in enumerate(id2token)}
        doc_freq = [int(df[t]) for t in id2token]
        term_freq = [int(tf[t]) for t in id2token]
        return Vocab(token2id, id2token, doc_freq, term_freq, num_docs)

    def save(self) -> Dict:
        """Serialize vocabulary to dictionary."""
        return {
            "id2token": self.id2token,
            "doc_freq": self.doc_freq,
            "term_freq": self.term_freq,
            "num_docs": self.num_docs,
        }

    @staticmethod
    def load(obj: Dict) -> "Vocab":
        """Deserialize vocabulary from dictionary."""
        id2token = obj["id2token"]
        token2id = {t:i for i,t in enumerate(id2token)}
        return Vocab(
            token2id=token2id,
            id2token=id2token,
            doc_freq=obj.get("doc_freq", [1]*len(id2token)),
            term_freq=obj.get("term_freq", [1]*len(id2token)),
            num_docs=obj.get("num_docs", 1),
        )


# -----------------------------
# Bag-of-Words Dataset
# -----------------------------

class BowDataset(Dataset):
    """PyTorch Dataset for bag-of-words representation of documents."""
    
    def __init__(self, rows: List[Dict], vocab: Vocab):
        """Initialize BOW dataset.
        
        Args:
            rows: List of dicts with 'document' and optionally 'id' keys
            vocab: Vocabulary object
        """
        self.rows = rows
        self.vocab = vocab
        self.ids = [r.get("id", str(i)) for i, r in enumerate(rows)]
        # Store only counts (as dictionaries) instead of dense tensors
        self._bow_counts = [self._doc_to_bow_counts(r["document"]) for r in rows]
        self.num_docs = len(self._bow_counts)
        self.vocab_size = len(vocab.id2token)

    def _doc_to_bow_counts(self, text: str) -> Dict[int, float]:
        """Convert document to sparse counts dictionary."""
        toks = simple_tokenize(text)
        counts = collections.Counter(t for t in toks if t in self.vocab.token2id)
        return {self.vocab.token2id[t]: float(c) for t, c in counts.items()}

    def __len__(self):
        return len(self._bow_counts)

    def __getitem__(self, idx):
        """Materialize dense tensor on demand."""
        counts = self._bow_counts[idx]
        v = torch.zeros(self.vocab_size, dtype=torch.float32)
        for token_id, count in counts.items():
            v[token_id] = count
        return v

    def save(self, path: str, vocab: Vocab):
        """Save dataset and vocab to disk for caching."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'ids': self.ids,
            'bow_counts': self._bow_counts,
            'vocab_size': self.vocab_size,
            'num_docs': self.num_docs,
            'vocab': vocab.save(),
        }
        torch.save(data, path)

    @staticmethod
    def load(path: str) -> Tuple["BowDataset", Vocab]:
        """Load cached dataset and vocab from disk."""
        data = torch.load(path, map_location='cpu')
        # Reconstruct vocab
        vocab = Vocab.load(data['vocab'])
        # Create empty dataset and populate
        ds = BowDataset.__new__(BowDataset)
        ds.vocab = vocab
        ds.ids = data['ids']
        ds._bow_counts = data['bow_counts']
        ds.vocab_size = data['vocab_size']
        ds.num_docs = len(ds._bow_counts)
        ds.rows = None  # rows not needed after construction
        return ds, vocab


# -----------------------------
# Utility Functions
# -----------------------------

def build_vocab_from_jsonl(jsonl_path: str, max_size: int, min_df: int, show_progress: bool=False) -> Vocab:
    """Build vocabulary directly from JSONL file."""
    rows = read_jsonl(jsonl_path)
    vocab = Vocab.build((r['document'] for r in rows), max_size=max_size, min_df=min_df, show_progress=show_progress)
    return vocab


def docs_to_bow(rows: List[Dict], vocab: Vocab) -> torch.Tensor:
    """Convert documents to dense BOW matrix (for models that don't use Dataset).
    
    Args:
        rows: List of dicts with 'document' key
        vocab: Vocabulary object
        
    Returns:
        BOW matrix of shape (num_docs, vocab_size)
    """
    V = len(vocab.id2token)
    bows = torch.zeros((len(rows), V), dtype=torch.float32)
    for i, r in enumerate(rows):
        toks = simple_tokenize(r["document"])
        for t, c in collections.Counter(t for t in toks if t in vocab.token2id).items():
            bows[i, vocab.token2id[t]] = float(c)
    return bows

