import os
import pickle
from collections import Counter
from typing import List, Dict, Set, Tuple
import numpy as np


class SparseEmbedding:
    """Creates sparse word embeddings using co-occurrence matrix."""

    def __init__(self,
                 corpus_path: str,
                 vocabulary_size: int = 10000,
                 window_size: int = 2,
                 max_lines: int = 8_000_000,
                 use_cache: bool = False):
        self.corpus_path = corpus_path
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.max_lines = max_lines
        self.use_cache = use_cache

        self.top_words: List[str] = []
        self.word_to_idx: Dict[str, int] = {}

        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_train_test_embeddings(self,
                                  train_words: List[str],
                                  test_words: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings for train and test words.

        Builds vocabulary and co-occurrence matrix, then returns normalized embeddings.
        """
        print("Building vocabulary...")
        self.build_vocabulary()

        all_words = set(train_words) | set(test_words)
        print(f"Building co-occurrence matrix for {len(all_words)} words...")
        word_vectors = self.build_cooccurrence_matrix(all_words)

        X_train = self.get_embeddings(train_words, word_vectors)
        X_test = self.get_embeddings(test_words, word_vectors)
        return X_train, X_test

    def build_vocabulary(self) -> None:
        """Build vocabulary from the top-k most frequent words in corpus."""
        # Try cache first
        cached = self._load_from_cache('vocab')
        if cached is not None:
            self.top_words = cached['top_words']
            self.word_to_idx = cached['word_to_idx']
            return

        # Build vocabulary from corpus
        word_counts = Counter()
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                tokens = line.lower().split()
                word_counts.update(tokens)

        self.top_words = [word for word, _ in word_counts.most_common(self.vocabulary_size)]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.top_words)}

        # Save to cache
        self._save_to_cache('vocab', {'top_words': self.top_words, 'word_to_idx': self.word_to_idx})

    def build_cooccurrence_matrix(self, target_words: Set[str]) -> Dict[str, np.ndarray]:
        """Build co-occurrence vectors for target words only."""
        # Try cache first
        cached = self._load_from_cache('cooccurrence')
        if cached is not None:
            return cached

        # Build co-occurrence matrix
        word_vectors = {word: np.zeros(self.vocabulary_size, dtype=np.float32)
                        for word in target_words}

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                tokens = line.lower().split()
                self._update_cooccurrence(tokens, target_words, word_vectors)

        # Save to cache
        self._save_to_cache('cooccurrence', word_vectors)

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

    def _get_cache_path(self, name: str) -> str:
        """Get cache file path for a given cache name."""
        return os.path.join(self.cache_dir, f'{name}_{self.vocabulary_size}_{self.max_lines}_{self.window_size}.pkl')

    def _load_from_cache(self, name: str):
        """Load data from cache if available. Returns None if not found."""
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(name)
        if os.path.exists(cache_path):
            print(f"Loading {name} from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, name: str, data) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return
        cache_path = self._get_cache_path(name)
        print(f"Saving {name} to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
