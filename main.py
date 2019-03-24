import random
import argparse

from Config.Configer import Configer
from DataLoader.Vocab import VocabSrc, VocabTar
from Models.Lstm import *
from Train.Train import *


if __name__ == '__main__':

    #  random
    torch.manual_seed(666)
    torch.cuda.manual_seed(888)
    np.random.seed(666)
    random.seed(666)

    #  gpu
    gpu = torch.cuda.is_available()
    if gpu:
        print('The gpu is available!!!')
    else:
        print('The trainning will use cpu!!!')
    print('CuDNN', torch.backends.cudnn.enabled)

    #  parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./Config/configer.ini')
    args = parser.parse_args()
    config = Configer(args.config_file)
    if gpu is True:
        config.add_paramter('train', 'use_cuda', 'True')

    # data_loader
    tra_word, dev_word, test_word, tra_label, dev_label, test_label = read_file(config)

    # vocab
    src_vocab = VocabSrc(tra_word, dev_word, test_word, config)
    tar_vocab = VocabTar(tra_label, dev_label, test_label, config)

    label_kind = tar_vocab.getsize

    # embedding
    if not os.path.exists(config.embedding_pkl):
        config.add_paramter('model', 'embedding_num', str(src_vocab.getsize))
        embedding = create_embedding(src_vocab, config)
        pickle.dump(embedding, open(config.embedding_pkl, 'wb'))
    else:
        embedding = read_pkl(config.embedding_pkl)

    # model
    model = ''
    if config.which_model == 'lstm':
        model = LSTM(config, embedding, label_kind)
    else:
        print('Please choose true model!!!')

    if config.use_cuda:
        model.cuda()

    # train
    train(model, src_vocab, tar_vocab, tra_word, dev_word, test_word, config)
    #predict_dev(model, src_vocab, tar_vocab, tra_word, config)
    #predict_test(model, src_vocab, tar_vocab, dev_word, config)
