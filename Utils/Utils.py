import numpy as np
from DataLoader.DataLoader import *
import torch
import time


def read_file(config):
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)
    if os.path.isdir(config.save_pkl_path):
        tra_word = read_pkl(config.train_data_pkl)
        dev_word = read_pkl(config.dev_data_pkl)
        test_word = read_pkl(config.test_data_pkl)
        tra_label = read_pkl(config.train_label_pkl)
        dev_label = read_pkl(config.dev_label_pkl)
        test_label = read_pkl(config.test_label_pkl)
        return tra_word, dev_word, test_word, tra_label, dev_label, test_label
    else:
        os.makedirs(config.save_pkl_path)
        tra_word, tra_label = read_data(config.train_data)
        dev_word, dev_label = read_data(config.dev_data)
        test_word, test_label = read_data(config.test_data)
        if config.train_data_pkl:
            pickle.dump(tra_word, open(config.train_data_pkl, 'wb'))
        if config.dev_data_pkl:
            pickle.dump(dev_word, open(config.dev_data_pkl, 'wb'))
        if config.test_data_pkl:
            pickle.dump(test_word, open(config.test_data_pkl, 'wb'))
        if config.train_label_pkl:
            pickle.dump(tra_label, open(config.train_label_pkl, 'wb'))
        if config.dev_label_pkl:
            pickle.dump(dev_label, open(config.dev_label_pkl, 'wb'))
        if config.test_label_pkl:
            pickle.dump(test_label, open(config.test_label_pkl, 'wb'))
        return tra_word, dev_word, test_word, tra_label, dev_label, test_label


def create_embedding(src_vocab, config):
    embedding_dim = 0
    embedding_num = 0
    find_count = 0
    embedding = np.zeros((src_vocab.getsize, 1), dtype='float64')
    with open(config.embedding_file, encoding='utf-8') as f:
        for vec in f.readlines():
            vec = vec.strip()
            vec = vec.split()
            if embedding_num == 0:
                embedding_dim = len(vec) - 1
                embedding = np.zeros((src_vocab.getsize, embedding_dim), dtype='float64')
            if vec[0] in src_vocab.id2word_lower:
                find_count += 1
                vector = np.array(vec[1:], dtype='float64')
                embedding[src_vocab.w2i_lower(vec[0])] = vector
                embedding[src_vocab.UNK] += vector
            embedding_num += 1
        not_find = src_vocab.getsize - find_count
        oov_ration = float(not_find / src_vocab.getsize)
        embedding[src_vocab.UNK] = embedding[src_vocab.UNK] / find_count
        embedding = embedding / np.std(embedding)
        print('Total word:', str(embedding_num))
        print('The dim of pre_embedding:' + str(embedding_dim) + '\n')
        print('oov ratio is: {:.4f}'.format(oov_ration))
        return embedding


def pair_data_variable(batch, src_vocab, tar_vocab, config):
    length = []
    max_data_length = len(batch[0][0])
    batch_size = len(batch)
    src_matrix = np.zeros((batch_size, max_data_length))
    tar_matrix = np.zeros((batch_size, max_data_length))
    for idx, instance in enumerate(batch):
        length.append(len(instance[0]))
        sentence = src_vocab.w2i(instance[0])
        for jdx, value in enumerate(sentence):
            src_matrix[idx][jdx] = value
        tag = tar_vocab.w2i(instance[1])
        for kdx, values in enumerate(tag):
            tar_matrix[idx][kdx] = values
    src_matrix = torch.from_numpy(src_matrix).long()
    tar_matrix = torch.from_numpy(tar_matrix).long()
    if config.use_cuda:
        src_matrix = src_matrix.cuda()
        tar_matrix = tar_matrix.cuda()
    return [src_matrix, tar_matrix, length]


def create_batch(src, batch_size, src_vocab, tar_vocab, config, shuffle=False):
    print('create_batch is start')
    start_time = time.time()
    batch_data = []
    data_size = len(src)
    if shuffle:
        np.random.shuffle(src)
    src_ids = sorted(range(data_size), key=lambda src_id: len(src[src_id][0]), reverse=True)
    src = [src[src_id] for src_id in src_ids]

    unit = []
    instances = []
    for instance in src:
        instances.append(instance)
        if len(instances) == batch_size:
            unit.append(instances)
            instances = []

    if len(instances) > 0:
        unit.append(instances)

    for batch in unit:
        one_data = pair_data_variable(batch, src_vocab, tar_vocab, config)
        batch_data.append([one_data, batch])

    print('the create_batch all use:{:.2f} S'.format(time.time() - start_time))
    return batch_data


