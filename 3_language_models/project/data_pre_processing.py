import operator
import numpy as np
import pandas as pd


def get_n_gram_to_freq_map(file, n):
    n_gram_freq = {}
    for line in file:
        line_arr = line.split()
        for i in range(0, len(line_arr) - (n-1)):
            n_gram = ' '.join(line_arr[i:i+n])
            if n_gram_freq.get(n_gram) is None:
                n_gram_freq[n_gram] = 1
            else:
                n_gram_freq[n_gram] += 1
    return n_gram_freq


def get_n_gram_to_id_map(n_grams):
    # the input n_grams must be a set of unique n_grams
    ids = {}
    id_counter = 0
    for n_gram in n_grams:
        ids[n_gram] = id_counter
        id_counter += 1
    return ids


def get_data_matrix(vocab_to_id_map, n_grams):
    n = 1
    for n_gram in n_grams.keys():
        n = len(n_gram.strip().split())
        break
    data_matrix = np.ndarray(shape=(len(n_grams), n))
    i = 0
    for n_gram in n_grams.keys():
        arr = n_gram.strip().split()
        for j in range(len(arr)):
            data_matrix[i, j] = vocab_to_id_map[arr[j]]
        i += 1
    return data_matrix


def sort_by_values(dict):
    return sorted(dict.items(), key=operator.itemgetter(1))


def get_non_top_k_words(k, word_freq):
    sorted_words = sort_by_values(word_freq)
    non_top_k_words = []
    for i in range(len(sorted_words) - k):
        non_top_k_words.append(sorted_words[i][0])
    return non_top_k_words


def get_non_vocab_words(vocabulary, source_words):
    non_vocab_words = []
    for word in source_words:
        if word not in vocabulary:
            non_vocab_words.append(word)
    return non_vocab_words


def replace_words(words, file, replacement):
    for i in range(len(file)):
        for word in words:
            temp = file[i].strip().split()
            for j in range(len(temp)):
                if temp[j] == word:
                    temp[j] = replacement
            file[i] = ' '.join(temp)

    return file


def insert_at_ends_of_sent(start_str, end_str, file):
    for i in range(len(file)):
        file[i] = ' '.join((start_str, file[i], end_str))
    return file


def pre_process_training_data(path):
    with open(path) as f:
        file = f.readlines()

    # convert all words to lowercase
    for i in range(len(file)):
        file[i] = file[i].lower()

    # get word frequencies
    word_freq = get_n_gram_to_freq_map(file=file, n=1)

    # get the non top-7997 words.
    non_top_words = get_non_top_k_words(k=7997, word_freq=word_freq)

    # replace non top-7997 with UNK
    file = replace_words(words=non_top_words, file=file, replacement="UNK")

    # insert STOP and END
    file = insert_at_ends_of_sent(start_str="START", end_str='END', file=file)

    # assign word ids
    word_ids = get_n_gram_to_id_map(get_n_gram_to_freq_map(file=file, n=1).keys())
    return word_ids, file


def pre_process_validation_data(path, vocabulary):
    with open(path) as f:
        file = f.readlines()

    # convert all words to lower case
    for i in range(len(file)):
        file[i] = file[i].lower()

    # get word frequencies
    word_freq = get_n_gram_to_freq_map(file=file, n=1)

    # get words not in the vocabulary
    non_vocab_words = get_non_vocab_words(vocabulary=vocabulary.keys(), source_words=word_freq.keys())

    # replace words not in the vocabulary with UNK
    file = replace_words(words=non_vocab_words, file=file, replacement="UNK")

    # insert STOP and END
    file = insert_at_ends_of_sent(start_str="START", end_str='END', file=file)

    return file


def save_as_hdf5(path, ds_name, ds):
    hdf = pd.HDFStore(path)
    hdf[ds_name] = ds
    hdf.close()

if __name__ == '__main__':
    # paths
    trn_data_path = 'project/data/train.txt'
    vldn_data_path = 'project/data/val.txt'
    data_path = 'project/data/'

    # pre-process the training data
    vocab_to_id_map, trn_data_processed = pre_process_training_data(path=trn_data_path)

    # generate n-grams for training data
    trn_n_gram_to_freq_map = get_n_gram_to_freq_map(file=trn_data_processed, n=4)
    trn_n_gram_to_id_map = get_n_gram_to_id_map(n_grams=trn_n_gram_to_freq_map.keys())
    trn_combined = get_data_matrix(vocab_to_id_map=vocab_to_id_map, n_grams=trn_n_gram_to_id_map)

    # save the training data structures
    save_as_hdf5(path=data_path + 'vocab_to_id.hdf5', ds_name='vocab_to_id',
                 ds=pd.Series(vocab_to_id_map))
    save_as_hdf5(path=data_path + 'trn_data_processed.hdf5',
                 ds_name='trn_data_processed', ds=pd.DataFrame(trn_data_processed))
    save_as_hdf5(path=data_path + 'trn_n_gram_to_freq.hdf5', ds_name='trn_n_gram_to_freq',
                 ds=pd.Series(trn_n_gram_to_freq_map))
    save_as_hdf5(path=data_path + 'trn_n_gram_to_id.hdf5', ds_name='trn_n_gram_to_id',
                 ds=pd.Series(trn_n_gram_to_id_map))
    save_as_hdf5(path=data_path + 'trn_combined.hdf5', ds_name='trn_combined',
                 ds=pd.DataFrame(trn_combined))

    # pre-process validation data using trn vocab
    vldn_data_processed = pre_process_validation_data(path=vldn_data_path, vocabulary=vocab_to_id_map)

    # generate n-grams for validation data
    vldn_n_gram_to_freq_map = get_n_gram_to_freq_map(file=vldn_data_processed, n=4)
    vldn_n_gram_to_id_map = get_n_gram_to_id_map(n_grams=vldn_n_gram_to_freq_map.keys())
    vldn_combined = get_data_matrix(vocab_to_id_map=vocab_to_id_map, n_grams=vldn_n_gram_to_id_map)

    # save the validation data structures
    save_as_hdf5(path=data_path + 'vldn_data_processed.hdf5',
                 ds_name='vldn_data_processed', ds=pd.DataFrame(vldn_data_processed))
    save_as_hdf5(path=data_path + 'vldn_n_gram_to_freq.hdf5', ds_name='vldn_n_gram_to_freq',
                 ds=pd.Series(vldn_n_gram_to_freq_map))
    save_as_hdf5(path=data_path + 'vldn_n_gram_to_id.hdf5', ds_name='vldn_n_gram_to_id',
                 ds=pd.Series(vldn_n_gram_to_id_map))
    save_as_hdf5(path=data_path + 'vldn_combined.hdf5', ds_name='vldn_combined',
                 ds=pd.DataFrame(vldn_combined))
