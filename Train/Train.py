import time
import torch.nn as nn
from torch import optim

from Utils.Utils import *
from Train.eval import *
from Train.utils import *
import torch.nn.functional as F


def decay_learning_rate(config, optimizer, epoch):
    lr = config.lr / (1 + config.lr_rate_decay * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train(model, src_vocab, tar_vocab, tra_data, dev_data, test_data, config):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = ''
    if config.learning_algorithm == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'sdg':
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        print('Invalid optimizer method: ' + config.learning_algorithm)

    batch_data = create_batch(tra_data, config.batch_size, src_vocab, tar_vocab, config, shuffle=True)

    # Get start!!!
    best_f1 = -1
    best_epoch = -1
    for epoch in range(0, config.epoch):
        epoch_start_time = time.time()
        batch_iter = 0
        model.train()
        print('The epoch is :', str(epoch))
        if config.use_lr_decay:
            optimizer = decay_learning_rate(config, optimizer, epoch)
            print("now lr is {}".format(optimizer.param_groups[0].get("lr")), '\n')
        for batch in batch_data:
            start_time = time.time()
            feather = batch[0][0]
            target = batch[0][1]
            length = batch[0][2]
            optimizer.zero_grad()
            logit = model(feather, length)
            if config.use_crf:
                loss = model.crf.loss_log(logit, target, length)
            else:
                logit = logit.view(logit.size()[0] * logit.size()[1], logit.size()[2])
                target = target.view(target.size()[0] * target.size()[1])
                # loss = F.cross_entropy(logit, target, reduction='mean')
                loss = F.cross_entropy(logit, target, reduction='sum')
            if config.clip_max_norm_use:
                nn.utils.clip_grad_norm_(parameters, max_norm=5)
            loss.backward()
            optimizer.step()
            during_time = float(time.time() - start_time)
            print('Epoch:{}, batch_iter:{}, time:{:.2f}, loss:{:.6f}'
                  .format(epoch, batch_iter, during_time, loss.item()))
            batch_iter += 1
        epoch_time = float(time.time() - epoch_start_time)
        print("epoch_time is:", epoch_time)
        f1, current_epoch = predict_dev(model, src_vocab, tar_vocab, tra_data, epoch, best_f1, config)
        if f1 != -1:
            best_f1 = f1
        if current_epoch != -1:
            best_epoch = current_epoch
        if epoch - best_epoch > 100:
            predict_test(model, src_vocab, tar_vocab, test_data, config)
            print('early stop!!!')
            exit()


def predict_dev(model, src_vocab, tar_vocab, data, epoch, best_f1, config):
    model.eval()
    predict_path = []
    best_dev_epoch = -1
    best_dev_f1 = -1
    dev_eval = Eval()
    eval_PRF = EvalPRF()
    #dev_eval.clear_PRF()
    batch_data = create_batch(data, config.batch_size, src_vocab, tar_vocab, config)
    epoch_start_time = time.time()
    for batch in batch_data:
        feather = batch[0][0]
        target = batch[0][1]
        length = batch[0][2]
        logit = model(feather, length)
        path = model.crf.viterbi_decode(logit, length, tar_vocab)
        predict_path.append(path)
    for p_labels, g_labels in zip(predict_path, batch_data):
        for p_label, g_label in zip(p_labels, g_labels[1]):
            eval_PRF.evalPRF(predict_labels=p_label, gold_labels=g_label[1][1:-1], evall=dev_eval)
    p, r, f = dev_eval.getFscore()
    print(
        "dev: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(p, r, f))
    if f > best_f1:
        best_dev_f1 = f
        best_dev_epoch = epoch
        if not os.path.isdir(config.save_model_path):
            os.makedirs(config.save_model_path)
        if os.path.isfile(config.model_pkl):
            os.remove(config.model_pkl)
        output = open(config.model_pkl, mode="wb")
        torch.save(model, output)
        output.close()
        once_time = float(time.time() - epoch_start_time)
        print('dev_time is:', once_time)
    return best_dev_f1, best_dev_epoch


# def predict_dev(model, src_vocab, tar_vocab, data, config):
#     model.eval()
#     batch_data = create_batch(data, config.batch_size, src_vocab, tar_vocab, config)
#     if os.path.exists(config.result_dev_file):
#         os.remove(config.result_dev_file)
#     writer = open(config.result_dev_file, encoding='utf-8', mode='w')
#     epoch_start_time = time.time()
#     for batch in batch_data:
#         feather = batch[0][0]
#         target = batch[0][1]
#         length = batch[0][2]
#         logit = model(feather, length)
#         path = model.crf.viterbi_decode(logit, length, tar_vocab)
#         for idx in range(len(batch[1])):
#             word = batch[1][idx][0][1:-1]
#             tag = batch[1][idx][1][1:-1]
#             single_path = path[idx]
#             if len(word) != len(tag) != single_path:
#                 print('the predict_path is error!')
#                 break
#             else:
#                 for jdx in range(len(word)):
#                     writer.write(word[jdx] + " " + tag[jdx] + " " + single_path[jdx] + "\n")
#                 writer.write("\n")
#     once_time = float(time.time() - epoch_start_time)
#     print('dev_time is:', once_time)
#     writer.close()


def predict_test(model, src_vocab, tar_vocab, data, config):
    model = torch.load(config.model_pkl)
    batch_data = create_batch(data, config.batch_size, src_vocab, tar_vocab, config)
    if os.path.exists(config.result_test_file):
        os.remove(config.result_test_file)
    writer = open(config.result_test_file, encoding='utf-8', mode='w')
    epoch_start_time = time.time()
    for batch in batch_data:
        feather = batch[0][0]
        target = batch[0][1]
        length = batch[0][2]
        logit = model(feather, length)
        path = model.crf.viterbi_decode(logit, length, tar_vocab)
        for idx in range(len(batch[1])):
            word = batch[1][idx][0][1:-1]
            tag = batch[1][idx][1][1:-1]
            single_path = path[idx]
            if len(word) != len(tag) != single_path:
                print('the predict_path is error!')
                break
            else:
                for jdx in range(len(word)):
                    writer.write(word[jdx] + " " + tag[jdx] + " " + single_path[jdx] + "\n")
                writer.write("\n")
    once_time = float(time.time() - epoch_start_time)
    print('test_time is:', once_time)
    writer.close()
