import os
from configparser import ConfigParser


class Configer(object):

    def __init__(self, config_file):

        config = ConfigParser()
        config.read(config_file)
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ':', v)
        self._config = config
        self.config_file = config_file
        config.write(open(config_file, 'w+'))

    def add_paramter(self, section, keys, values):
        if self._config.has_section(section):
            print('The section is already exist!!!')
        else:
            self._config.add_section(section)
        if self._config.has_option(section, keys):
            print('The option is already exist!!!')
            self._config.set(section, keys, values)
        else:
            self._config.set(section, keys, values)
        self._config.write(open(self.config_file, 'w'))

    # file
    @property
    def train_data(self):
        return self._config.get('file', 'train_data')

    @property
    def dev_data(self):
        return self._config.get('file', 'dev_data')

    @property
    def test_data(self):
        return self._config.get('file', 'test_data')

    @property
    def embedding_file(self):
        return self._config.get('file', 'embedding_file')

    @property
    def result_dev_file(self):
        return self._config.get('file', 'result_dev_file')

    @property
    def result_test_file(self):
        return self._config.get('file', 'result_test_file')

    # save
    @property
    def save_dir(self):
        return self._config.get('save', 'save_dir')

    @property
    def save_pkl_path(self):
        return self._config.get('save', 'save_pkl_path')

    @property
    def save_model_path(self):
        return self._config.get('save', 'save_model_path')

    @property
    def model_pkl(self):
        return self._config.get('save', 'model_pkl')

    @property
    def train_data_pkl(self):
        return self._config.get('save', 'train_data_pkl')

    @property
    def train_label_pkl(self):
        return self._config.get('save', 'train_label_pkl')

    @property
    def dev_data_pkl(self):
        return self._config.get('save', 'dev_data_pkl')

    @property
    def dev_label_pkl(self):
        return self._config.get('save', 'dev_label_pkl')

    @property
    def test_data_pkl(self):
        return self._config.get('save', 'test_data_pkl')

    @property
    def test_label_pkl(self):
        return self._config.get('save', 'test_label_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('save', 'embedding_pkl')

    # model
    @property
    def which_model(self):
        return self._config.get('model', 'which_model')

    @property
    def pre_word_embedding(self):
        return self._config.getboolean('model', 'pre_word_embedding')

    @property
    def hidden_size(self):
        return self._config.getint('model', 'hidden_size')

    @property
    def dropout(self):
        return self._config.getfloat('model', 'dropout')

    @property
    def embedding_num(self):
        return self._config.getint('model', 'embedding_num')

    @property
    def learning_algorithm(self):
        return self._config.get('model', 'learning_algorithm')

    @property
    def lr(self):
        return self._config.getfloat('model', 'lr')

    @property
    def weight_decay(self):
        return self._config.getfloat('model', 'weight_decay')

    @property
    def lr_rate_decay(self):
        return self._config.getfloat('model', 'lr_rate_decay')

    @property
    def epoch(self):
        return self._config.getint('model', 'epoch')

    @property
    def use_lr_decay(self):
        return self._config.getboolean('model', 'use_lr_decay')

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean('model', 'clip_max_norm_use')

    # train
    @property
    def use_cuda(self):
        return self._config.getboolean('train', 'use_cuda')

    @property
    def vocab_size(self):
        return self._config.getint('train', 'vocab_size')

    @property
    def label_size(self):
        return self._config.getint('train', 'label_size')

    @property
    def test_interval(self):
        return self._config.getint('train', 'test_interval')

    @property
    def batch_size(self):
        return self._config.getint('train', 'batch_size')

    @property
    def use_crf(self):
        return self._config.getboolean('train', 'use_crf')



