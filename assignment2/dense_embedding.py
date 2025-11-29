"""
DenseEmbedding: Loads pre-trained word embeddings using gensim.
"""
from typing import List
import numpy as np
import gensim.downloader as api


class DenseEmbedding:
    """Loads and provides access to pre-trained word embeddings."""

    def __init__(self, model_name: str = 'word2vec-google-news-300'):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """Load the pre-trained embedding model."""
        self.model = api.load(self.model_name)

    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Get embedding matrix for a list of words."""
        dim = self.model.vector_size
        embeddings = []

        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(dim, dtype=np.float32))

        return np.array(embeddings)
