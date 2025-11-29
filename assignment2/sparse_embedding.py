"""
SparseEmbedding: Builds co-occurrence matrix from Wikipedia corpus.
"""
from collections import Counter
from typing import List, Dict, Set
import numpy as np


class SparseEmbedding:
    """Creates sparse word embeddings using co-occurrence matrix."""

    def __init__(self, corpus_path: str, vocabulary_size: int = 10000,
                 window_size: int = 2, max_lines: int = 8_000_000):
        self.corpus_path = corpus_path
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.max_lines = max_lines

        self.top_words: List[str] = []
        self.word_to_idx: Dict[str, int] = {}

    def build_vocabulary(self) -> None:
        """Build vocabulary from the top-k most frequent words in corpus."""
        word_counts = Counter()
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                tokens = line.lower().split()
                word_counts.update(tokens)

        self.top_words = [word for word, _ in word_counts.most_common(self.vocabulary_size)]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.top_words)}

    def build_cooccurrence_matrix(self, target_words: Set[str]) -> Dict[str, np.ndarray]:
        """Build co-occurrence vectors for target words only."""
        word_vectors = {word: np.zeros(self.vocabulary_size, dtype=np.float32)
                        for word in target_words}

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                tokens = line.lower().split()
                self._update_cooccurrence(tokens, target_words, word_vectors)

        return word_vectors

    def _update_cooccurrence(self, tokens: List[str], target_words: Set[str],
                             word_vectors: Dict[str, np.ndarray]) -> None:
        """Update co-occurrence counts for a single line of tokens."""
        for i, token in enumerate(tokens):
            if token not in target_words:
                continue

            start = max(0, i - self.window_size)
            end = min(len(tokens), i + self.window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                context_word = tokens[j]
                if context_word in self.word_to_idx:
                    idx = self.word_to_idx[context_word]
                    word_vectors[token][idx] += 1

    def get_embeddings(self, words: List[str],
                       word_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Get normalized embedding matrix for a list of words."""
        embeddings = []
        for word in words:
            if word in word_vectors:
                vec = word_vectors[word].copy()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)
            else:
                embeddings.append(np.zeros(self.vocabulary_size, dtype=np.float32))
        return np.array(embeddings)
