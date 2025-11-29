import json
import time
import numpy as np

from valence_dataset import ValenceDataset
from sparse_embedder import SparseEmbedder
from dense_embedder import DenseEmbedder
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
    # TODO: use_cache should be False when submitting the task
    use_cache = True
    if is_dense_embedding:
        embedding = DenseEmbedder(model_name='word2vec-google-news-300', use_cache=use_cache)
        X_train, X_test = embedding.get_train_test_embeddings(train_data.words, test_data.words)
    else:
        embedding = SparseEmbedder(
            corpus_path=data_path,
            vocabulary_size=VOCABULARY_SIZE,
            window_size=WINDOW_SIZE,
            max_lines=8_000_000,
            use_cache=use_cache
        )
        X_train, X_test = embedding.get_train_test_embeddings(train_data.words, test_data.words)

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
