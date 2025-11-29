import json
import time
import numpy as np

from valence_dataset import ValenceDataset
from sparse_embedding import SparseEmbedding
from dense_embedding import DenseEmbedding
from valence_predictor import ValencePredictor


VOCABULARY_SIZE = 10000
WINDOW_SIZE = 2  # experiment with different values (2-5 work well)


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    # Load datasets
    train_data = ValenceDataset(train_file)
    test_data = ValenceDataset(test_file)

    # Get embeddings based on type
    if is_dense_embedding:
        X_train, X_test = _get_dense_embeddings(train_data.words, test_data.words)
    else:
        X_train, X_test = _get_sparse_embeddings(train_data.words, test_data.words, data_path)

    # Train and evaluate
    y_train = np.array(train_data.scores)
    y_test = np.array(test_data.scores)

    predictor = ValencePredictor()
    predictor.train(X_train, y_train)
    y_pred = predictor.predict(X_test)

    mse, corr = predictor.evaluate(y_test, y_pred)
    return mse, corr


def _get_dense_embeddings(train_words, test_words):
    """Get dense embeddings using pre-trained word vectors."""
    # TODO: use_cache should be False when submitting the task
    embedding = DenseEmbedding(model_name='word2vec-google-news-300', use_cache=True)
    embedding.load_model()

    X_train = embedding.get_embeddings(train_words)
    X_test = embedding.get_embeddings(test_words)
    return X_train, X_test


def _get_sparse_embeddings(train_words, test_words, corpus_path):
    """Get sparse embeddings using co-occurrence matrix."""
    # TODO: use_cache should be False when submitting the task
    embedding = SparseEmbedding(
        corpus_path=corpus_path,
        vocabulary_size=VOCABULARY_SIZE,
        window_size=WINDOW_SIZE,
        use_cache=True
    )

    # Build vocabulary from corpus
    print("Building vocabulary...")
    embedding.build_vocabulary()

    # Build co-occurrence matrix for train + test words
    all_words = set(train_words) | set(test_words)
    print(f"Building co-occurrence matrix for {len(all_words)} words...")
    word_vectors = embedding.build_cooccurrence_matrix(all_words)

    X_train = embedding.get_embeddings(train_words, word_vectors)
    X_test = embedding.get_embeddings(test_words, word_vectors)
    return X_train, X_test


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
