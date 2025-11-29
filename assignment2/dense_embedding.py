import os
import pickle
from typing import List
import numpy as np
import gensim.downloader as api

class DenseEmbedding:
    """Loads and provides access to pre-trained word embeddings."""

    def __init__(self, model_name: str = 'word2vec-google-news-300', use_cache: bool = False):
        self.model_name = model_name
        self.model = None
        self.use_cache = use_cache

        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self) -> None:
        """Load the pre-trained embedding model."""
        self.model = api.load(self.model_name)

    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Get embedding matrix for a list of words."""
        # Try cache first
        cached = self._load_from_cache('dense_embeddings')
        if cached is not None:
            return cached

        # Build embeddings
        dim = self.model.vector_size
        embeddings = []

        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(dim, dtype=np.float32))

        result = np.array(embeddings)
        # Save to cache
        self._save_to_cache(name='dense_embeddings', data=result)

        return result

    def _get_cache_path(self, name: str) -> str:
        """Get cache file path for a given cache name."""
        return os.path.join(self.cache_dir, f'{name}_{self.model_name}.pkl')

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
