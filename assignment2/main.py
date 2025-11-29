import json
import time


VOCABULARY_SIZE = 10000
WINDOW_SIZE = 0  # todo: find the window size that works best for you


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    # todo: implement this function

    return 0, 0  #mse, corr


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
