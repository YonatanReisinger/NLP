import json
import time
import numpy as np
import os
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Set, Tuple, Optional
import gensim.downloader as api
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import csv

VOCABULARY_SIZE = 10000
WINDOW_SIZE = 5

class SparseEmbedder:
    """Creates sparse word embeddings using co-occurrence matrix."""

    def __init__(self,
                 corpus_path: str,
                 vocabulary_size: int = 10000,
                 window_size: int = 2,
                 max_lines: Optional[int] = None,
                 use_cache: bool = False):
        """
        Initialize the SparseEmbedder.

        Args:
            corpus_path: Path to the text corpus file
            vocabulary_size: Number of most frequent words to use as vocabulary
            window_size: Context window size for co-occurrence counting
            max_lines: Maximum number of lines to read from corpus (None for all)
            use_cache: Whether to cache intermediate results to disk
        """
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
        """
        Get embeddings for train and test words.

        Builds vocabulary and co-occurrence matrix, then returns normalized embeddings.

        Args:
            train_words: List of words from the training set
            test_words: List of words from the test set

        Returns:
            Tuple of (X_train, X_test) where each is a numpy array of shape
            (num_words, vocabulary_size) containing normalized word embeddings
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
        """
        Build vocabulary from the top-k most frequent words in corpus using multiprocessing.

        Populates self.top_words with the most frequent words and self.word_to_idx
        with a mapping from words to their indices.

        Returns:
            None (modifies self.top_words and self.word_to_idx in place)
        """
        # Try cache first
        cached = self._load_from_cache('vocab')
        if cached is not None:
            self.top_words = cached['top_words']
            self.word_to_idx = cached['word_to_idx']
            return

        print(f"Reading corpus from {self.corpus_path}...")
        lines = self._read_lines_from_corpus()

        # Split into chunks for parallel processing
        num_workers = cpu_count() or 1
        chunk_size = max(100000, len(lines) // num_workers)
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        print(f"Counting word frequencies in {len(lines)} lines using {num_workers} workers...")
        processor = VocabularyProcessor()

        with Pool(processes=num_workers) as pool:
            results = pool.map(processor.process_chunk, enumerate(chunks))

        # Merge word counts from all chunks
        print("Merging word counts from all chunks...")
        word_counts = Counter()
        for chunk_counts in results:
            word_counts.update(chunk_counts)

        self.top_words = [word for word, _ in word_counts.most_common(self.vocabulary_size)]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.top_words)}

        print(f"Vocabulary built with {len(self.top_words)} words")

        # Save to cache
        self._save_to_cache('vocab', {'top_words': self.top_words, 'word_to_idx': self.word_to_idx})

    def build_cooccurrence_matrix(self, target_words: Set[str]) -> Dict[str, np.ndarray]:
        """
        Build co-occurrence vectors for target words only using multiprocessing.

        Args:
            target_words: Set of words to build co-occurrence vectors for

        Returns:
            Dictionary mapping each target word to its co-occurrence vector
            (numpy array of shape (vocabulary_size,))
        """
        # Try cache first
        cached = self._load_from_cache('cooccurrence')
        if cached is not None:
            return cached

        print(f"Reading corpus for co-occurrence matrix...")
        lines = self._read_lines_from_corpus()

        # Split into chunks for parallel processing
        num_workers = cpu_count() or 1
        chunk_size = max(100000, len(lines) // num_workers)
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        print(f"Building co-occurrence matrix from {len(lines)} lines using {num_workers} workers...")
        processor = CooccurrenceProcessor(target_words, self.word_to_idx, self.window_size, self.vocabulary_size)

        with Pool(processes=num_workers) as pool:
            results = pool.map(processor.process_chunk, enumerate(chunks))

        # Merge co-occurrence counts from all chunks
        print("Merging co-occurrence counts from all chunks...")
        word_vectors = {word: np.zeros(self.vocabulary_size, dtype=np.float32)
                        for word in target_words}

        for chunk_vectors in results:
            for word, vec in chunk_vectors.items():
                word_vectors[word] += vec

        print(f"Co-occurrence matrix built for {len(target_words)} target words")

        # Save to cache
        self._save_to_cache('cooccurrence', word_vectors)

        return word_vectors

    def get_embeddings(self,
                       words: List[str],
                       word_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get normalized embedding matrix for a list of words.

        Args:
            words: List of words to get embeddings for
            word_vectors: Dictionary mapping words to their co-occurrence vectors

        Returns:
            Numpy array of shape (len(words), vocabulary_size) with L2-normalized embeddings
        """
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

    def _read_lines_from_corpus(self) -> List[str]:
        """
        Read lines from corpus up to max_lines.

        Returns:
            List of lines from the corpus file
        """
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines if self.max_lines is None else lines[:self.max_lines]

    def _get_cache_path(self, name: str) -> str:
        """
        Get cache file path for a given cache name.

        Args:
            name: Name identifier for the cache (e.g., 'vocab', 'cooccurrence')

        Returns:
            Full path to the cache file
        """
        if self.max_lines is not None:
            return os.path.join(self.cache_dir, f'{name}_{self.vocabulary_size}_{self.max_lines}_{self.window_size}.pkl')
        else:
            return os.path.join(self.cache_dir, f'{name}_{self.vocabulary_size}_{self.window_size}.pkl')

    def _load_from_cache(self, name: str):
        """
        Load data from cache if available.

        Args:
            name: Name identifier for the cache (e.g., 'vocab', 'cooccurrence')

        Returns:
            Cached data if found, None otherwise
        """
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(name)
        if os.path.exists(cache_path):
            print(f"Loading {name} from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, name: str, data) -> None:
        """
        Save data to cache.

        Args:
            name: Name identifier for the cache (e.g., 'vocab', 'cooccurrence')
            data: Data to save to the cache file

        Returns:
            None
        """
        if not self.use_cache:
            return
        cache_path = self._get_cache_path(name)
        print(f"Saving {name} to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)


class VocabularyProcessor:
    """Helper class for parallel vocabulary building."""

    def process_chunk(self, chunk_data: Tuple[int, List[str]]) -> Counter:
        """
        Process a chunk of lines to count word frequencies.

        Args:
            chunk_data: Tuple of (chunk_index, list_of_lines)

        Returns:
            Counter with word frequencies for this chunk
        """
        chunk_idx, chunk = chunk_data
        word_counts = Counter()

        for i, line in enumerate(chunk):
            tokens = line.lower().split()
            word_counts.update(tokens)

            if i % 100000 == 0 and i > 0:
                print(f"Chunk {chunk_idx}: Processed {i} lines...")

        print(f"Chunk {chunk_idx}: Finished processing {len(chunk)} lines")
        return word_counts


class CooccurrenceProcessor:
    """Helper class for parallel co-occurrence matrix building."""

    def __init__(self,
                 target_words: Set[str],
                 word_to_idx: Dict[str, int],
                 window_size: int,
                 vocabulary_size: int):
        """
        Initialize the CooccurrenceProcessor.

        Args:
            target_words: Set of words to build co-occurrence vectors for
            word_to_idx: Mapping from vocabulary words to indices
            window_size: Context window size for co-occurrence
            vocabulary_size: Size of the vocabulary (vector dimension)
        """
        self.target_words = target_words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        self.vocabulary_size = vocabulary_size

    def process_chunk(self, chunk_data: Tuple[int, List[str]]) -> Dict[str, np.ndarray]:
        """
        Process a chunk of lines to build co-occurrence vectors.

        Args:
            chunk_data: Tuple of (chunk_index, list_of_lines)

        Returns:
            Dictionary mapping target words to their co-occurrence vectors for this chunk
        """
        chunk_idx, chunk = chunk_data
        word_vectors = {word: np.zeros(self.vocabulary_size, dtype=np.float32)
                        for word in self.target_words}

        for i, line in enumerate(chunk):
            tokens = line.lower().split()
            self._update_cooccurrence(tokens, word_vectors)

            if i % 100000 == 0 and i > 0:
                print(f"Chunk {chunk_idx}: Processed {i} lines...")

        print(f"Chunk {chunk_idx}: Finished processing {len(chunk)} lines")
        return word_vectors

    def _update_cooccurrence(self, tokens: List[str], word_vectors: Dict[str, np.ndarray]) -> None:
        """
        Update co-occurrence counts for a single line of tokens.

        Args:
            tokens: List of tokens from a single line
            word_vectors: Dictionary mapping target words to their co-occurrence vectors

        Returns:
            None (modifies word_vectors in place)
        """
        for i, token in enumerate(tokens):
            if token not in self.target_words:
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


class DenseEmbedder:
    """Loads and provides access to pre-trained word embeddings."""

    def __init__(self, model_name: str, use_cache: bool = False):
        """
        Initialize the DenseEmbedder.

        Args:
            model_name: Name of the pre-trained model to load (e.g., 'word2vec-google-news-300')
            use_cache: Whether to cache embeddings to disk

        Returns:
            None
        """
        self.model_name = model_name
        self.model = None
        self.use_cache = use_cache

        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self) -> None:
        """
        Load the pre-trained embedding model.

        Args:
            None

        Returns:
            None (sets self.model with the loaded model)
        """
        self.model = api.load(self.model_name)

    def get_train_test_embeddings(self,
                                  train_words: List[str],
                                  test_words: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for train and test words.

        Args:
            train_words: List of words from the training set
            test_words: List of words from the test set

        Returns:
            Tuple of (X_train, X_test) where each is a numpy array of shape
            (num_words, embedding_dim) containing word embeddings
        """
        # Try cache first
        cached = self._load_from_cache('dense_embeddings')
        if cached is not None:
            return cached['X_train'], cached['X_test']

        # Load model and build embeddings
        self.load_model()
        X_train = self.get_embeddings(train_words)
        X_test = self.get_embeddings(test_words)

        # Save to cache
        self._save_to_cache('dense_embeddings', {'X_train': X_train, 'X_test': X_test})

        return X_train, X_test

    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """
        Get embedding matrix for a list of words.

        Args:
            words: List of words to get embeddings for

        Returns:
            Numpy array of shape (len(words), embedding_dim) with word embeddings
        """
        dim = self.model.vector_size
        embeddings = []

        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(dim, dtype=np.float32))

        return np.array(embeddings)

    def _get_cache_path(self, name: str) -> str:
        """
        Get cache file path for a given cache name.

        Args:
            name: Name identifier for the cache

        Returns:
            Full path to the cache file
        """
        return os.path.join(self.cache_dir, f'{name}_{self.model_name}.pkl')

    def _load_from_cache(self, name: str):
        """
        Load data from cache if available.

        Args:
            name: Name identifier for the cache

        Returns:
            Cached data if found, None otherwise
        """
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(name)
        if os.path.exists(cache_path):
            print(f"Loading {name} from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, name: str, data) -> None:
        """
        Save data to cache.

        Args:
            name: Name identifier for the cache
            data: Data to save to the cache file

        Returns:
            None
        """
        if not self.use_cache:
            return
        cache_path = self._get_cache_path(name)
        print(f"Saving {name} to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)


class ValencePredictor:
    """Trains and evaluates a linear regression model for valence prediction."""

    def __init__(self):
        """
        Initialize the ValencePredictor.

        Args:
            None

        Returns:
            None
        """
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            None (fits the internal model)
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict valence scores.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Numpy array of predicted valence scores of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate MSE and Pearson correlation.

        Args:
            y_true: Ground truth valence scores
            y_pred: Predicted valence scores

        Returns:
            Tuple of (MSE, Pearson correlation coefficient)
        """
        mse = mean_squared_error(y_true, y_pred)
        corr, _ = pearsonr(y_true, y_pred)
        return mse, corr


class ValenceDataset:
    """Handles loading word-valence data from CSV files."""

    def __init__(self, file_path: str):
        """
        Initialize the ValenceDataset.

        Args:
            file_path: Path to the CSV file containing word-valence pairs

        Returns:
            None
        """
        self.words: List[str] = []
        self.scores: List[float] = []
        self.word_to_score: Dict[str, float] = {}
        self._load(file_path)

    def _load(self, file_path: str) -> None:
        """
        Load word-valence pairs from a CSV file.

        Args:
            file_path: Path to the CSV file containing word-valence pairs

        Returns:
            None (populates self.words, self.scores, and self.word_to_score)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    word = row[0].strip()
                    score = float(row[1])
                    self.words.append(word)
                    self.scores.append(score)
                    self.word_to_score[word] = score


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    # Load datasets
    train_data = ValenceDataset(train_file)
    test_data = ValenceDataset(test_file)

    # Get embeddings based on type
    if is_dense_embedding:
        embedder = DenseEmbedder(model_name='word2vec-google-news-300')
    else:
        embedder = SparseEmbedder(
            corpus_path=data_path,
            vocabulary_size=VOCABULARY_SIZE,
            window_size=WINDOW_SIZE,
        )
    X_train, X_test = embedder.get_train_test_embeddings(train_data.words, test_data.words)

    # Train and evaluate
    y_train = np.array(train_data.scores)
    y_test = np.array(test_data.scores)

    predictor = ValencePredictor()
    predictor.train(X_train, y_train)
    y_pred = predictor.predict(X_test)

    mse, corr = predictor.evaluate(y_test, y_pred)
    return mse, corr


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    mse, corr = predict_words_valence(
        config['train'],
        config['test'],
        config['wiki_data'],
        config["word_embedding_dense"])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time: .2f} seconds")

    print(f'test set evaluation results: MSE: {mse: .3f}, '
          f'Pearsons correlation: {corr: .3f}')
