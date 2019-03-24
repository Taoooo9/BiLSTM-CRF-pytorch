import torch
import numpy as np
import torch.nn as nn
from torch import autograd


class CRF(nn.Module):

    def __init__(self, label_kind, config):
        super(CRF, self).__init__()
        self.config = config
        self.label_num = label_kind
        if config.use_cuda is True:
            self.transition = torch.randn((self.label_num, self.label_num), requires_grad=True).cuda()
        else:
            self.transition = torch.randn((self.label_num, self.label_num), requires_grad=True)
        self.transition.data[1, :] = -15
        self.transition.data[:, 2] = -15

    @staticmethod
    def make_sen_mask(length, batch_size):
        mask = torch.zeros((batch_size, length[0]), dtype=torch.uint8)
        for i in range(batch_size):
            mask[i][0:length[i]] = 1
        return mask

    def make_t_mask(self, label_idx, batch_size):
        sen_length = label_idx.size()[1] - 1
        transition_t = torch.unsqueeze(self.transition, 0)
        transition_t = torch.unsqueeze(transition_t, 0)
        transition_t = transition_t.expand(batch_size, sen_length, self.label_num, self.label_num)
        if self.config.use_cuda is True:
            t_mask = torch.zeros((batch_size, sen_length, self.label_num, self.label_num), dtype=torch.uint8).cuda()
        else:
            t_mask = torch.zeros((batch_size, sen_length, self.label_num, self.label_num), dtype=torch.uint8)
        for i in range(batch_size):
            idx_start = label_idx[i][0]
            for idx, label in enumerate(label_idx[i][1:]):
                t_mask[i][idx][idx_start][label] = 1
                idx_start = label
        return transition_t, t_mask

    def make_e_mask(self, target, batch_size):
        sen_length = target.size()[1] - 2
        if self.config.use_cuda is True:
            e_mask = torch.zeros((batch_size, sen_length, self.label_num), dtype=torch.uint8).cuda()
        else:
            e_mask = torch.zeros((batch_size, sen_length, self.label_num), dtype=torch.uint8)
        for idx in range(batch_size):
            for i, j in enumerate(target[idx][1:-1]):
                e_mask[idx][i][j] = 1
        return e_mask

    def select_max(self, vec, batch_size, label_num):
        vec = torch.chunk(vec, batch_size * label_num, 1)
        vec = torch.cat(vec, 0)
        if self.config.use_cuda:
            vec = vec.cuda()
        max_score, _ = torch.max(vec, 1)
        max_scorer = max_score.view(batch_size * label_num, 1)
        max_score_broadcast = max_scorer.expand((batch_size * label_num, label_num)).reshape(1, -1)
        return max_score, max_score_broadcast

    def select_last_max(self, vec, batch_size, label_num):
        vec = torch.chunk(vec, batch_size, 1)
        vec = torch.cat(vec, 0)
        if self.config.use_cuda:
            vec = vec.cuda()
        max_score, _ = torch.max(vec, 1)
        max_scorer = max_score.view(batch_size, 1)
        max_score_broadcast = max_scorer.expand((batch_size, label_num)).reshape(1, -1)
        return max_score, max_score_broadcast

    def log_sum_exp(self, s, batch_size, label_num):
        if s.size()[1] == batch_size * label_num * label_num:
            max_score, max_score_broadcast = self.select_max(s, batch_size, label_num)
            a = torch.exp(s - max_score_broadcast)
            a = a.view(batch_size * label_num, -1)
        else:
            max_score, max_score_broadcast = self.select_last_max(s, batch_size, label_num)
            a = torch.exp(s - max_score_broadcast)
            a = a.view(batch_size, -1)
        s_temp = max_score + torch.log(torch.sum(a, 1))
        return s_temp

    def sentences_score(self, emit, target):
        batch_size = emit.size()[0]
        max_len = emit.size()[1]
        index = torch.linspace(1, max_len - 2, steps=max_len - 2).long()
        if self.config.use_cuda:
            index = index.cuda()
        new_emit = torch.index_select(emit, 1, index)
        transition, t_mask = self.make_t_mask(target, batch_size)
        e_mask = self.make_e_mask(target, batch_size)
        t_res = torch.masked_select(transition, t_mask)
        e_res = torch.masked_select(new_emit, e_mask)
        sen_score = torch.sum(t_res) + torch.sum(e_res)
        return sen_score

    def calc_sentences_scores(self, emit_scores, labels, length):
        """
        params:
            emit_scores: variable (seq_length, batch_size, label_nums)
            labels: variable (batch_size, seq_length)
            masks: variable (batch_size, seq_length)
        """

        seq_length = emit_scores.size()[1]
        batch_size = emit_scores.size()[0]
        masks = self.make_sen_mask(length, batch_size)
        emit_scores = emit_scores.transpose(0, 1)

        # ***** Part 2
        batch_length = torch.sum(masks, dim=1).long().unsqueeze(1)
        ends_index = torch.gather(labels, 1, (batch_length-1))

        # print(self.T[:, self.label2id['<pad>']].unsqueeze(0).view(1, self.label_nums))
        ends_transition = self.transition[:, 2].unsqueeze(0).expand(batch_size, self.label_num)
        ends_scores = torch.gather(ends_transition, 1, ends_index)
        ##### ends_scores: variable (batch_size, 1)


        # ***** Part 1
        # labels = Variable(torch.LongTensor(list(map(lambda t: [self.label2id['<start>']] + list(t), labels.data.tolist()))))
        labels = list(map(lambda t: [1] + list(t), labels.data.tolist()))
        ##### labels: list (batch_size, (seq_length+1))


        ##### labels_group: use lower dimension to map high dimension
        # labels_group = []
        # for label in labels:
        #     new = [label[id]*self.label_nums+label[id+1] for id in range(seq_length)]
        #     new = []
        #     for id in range(seq_length):
        #         new.append(label[id]*self.label_nums+label[id+1])
        #     labels_group.append(new)

        ##### optimize calculating the labels_group
        labels_group = [[label[id]*self.label_num+label[id+1] for id in range(seq_length)] for label in labels]
        labels_group = torch.tensor(labels_group).long()
        if self.config.use_cuda:
            labels_group = labels_group.cuda()
        ##### labels_group: variable (batch_size, seq_length)

        batch_words_num = batch_size * seq_length
        emit_scores_broadcast = emit_scores.contiguous().view(batch_words_num, -1).unsqueeze(1).view(batch_words_num, 1, self.label_num).expand(batch_words_num, self.label_num, self.label_num)
        trans_scores_broadcast = self.transition.unsqueeze(0).view(1, self.label_num, self.label_num).expand(batch_words_num, self.label_num, self.label_num)
        scores = emit_scores_broadcast + trans_scores_broadcast
        ##### scores: variable (batch_words_num, label_nums, label_nums)

        ##### error version
        ##### reasons: because before packing to 'batch_words_num' size, the size of emit_scores is (variable (seq_length, batch_size, label_nums)), in view of the problem of data storage, if you do it like this, you will achieve the different results with the correct version, although the different is small.
        # calc_total = torch.gather(scores.view(batch_size, seq_length, self.label_nums, self.label_nums).view(batch_size, seq_length, -1), 2, labels_group.view(batch_size, seq_length).unsqueeze(2).view(batch_size, seq_length, 1)).squeeze(2)
        ##### calc_total: variable (batch_size, seq_length)

        ##### correct version
        labels_group = labels_group.transpose(0, 1).contiguous()
        calc_total = torch.gather(scores.view(seq_length, batch_size, self.label_num, self.label_num).view(seq_length, batch_size, -1), 2, labels_group.view(seq_length, batch_size).unsqueeze(2).view(seq_length, batch_size, 1)).squeeze(2)

        ##### calc_total: variable (seq_length, batch_size)
        batch_scores = calc_total.masked_select(masks.transpose(0, 1))
        return batch_scores.sum() + ends_scores.sum()

    def encode_score(self, emit):
        batch_size = emit.size()[0]
        sentence_len = emit.size()[1] - 2
        label_num = emit.size()[2]
        mask = torch.tensor(self.make_s_row(batch_size, label_num))
        emit = torch.chunk(emit, batch_size, 0)
        emits = torch.squeeze(torch.cat(emit, 2), 0)
        one_t_row = self.transition[1][:].repeat(batch_size)
        if self.config.use_cuda is True:
            s_matrix = torch.zeros((sentence_len, label_num * batch_size), dtype=torch.float).cuda()
            mask = mask.cuda()
        else:
            s_matrix = torch.zeros((sentence_len, label_num * batch_size), dtype=torch.float)
        s_matrix[0][:] = emits[1][:] + one_t_row
        for idx in range(1, sentence_len):
            s_row = torch.take(s_matrix[idx-1][:], mask)
            e_row = self.make_e_row(emits[idx+1][:], batch_size, label_num)
            t_row = self.make_t_row(self.transition, batch_size, label_num)
            next_tag_var = s_row + e_row + t_row
            s_matrix[idx][:] = self.log_sum_exp(next_tag_var, batch_size, label_num)
        t_last_row = self.transition[:, 2].repeat((1, batch_size))
        last_tag_var = s_matrix[-1][:] + t_last_row
        s_end = self.log_sum_exp(last_tag_var, batch_size, label_num)
        s_end_sum = s_end.sum()
        return s_end_sum

    def viterbi_decode(self, feats, length, tar_vocab):
        best_way = []
        end_way = []
        batch_size = feats.size()[0]
        sentence_len = feats.size()[1] - 2
        label_num = feats.size()[2]
        last_best_way = torch.zeros((batch_size, sentence_len), dtype=torch.int)
        mask = torch.tensor(self.make_s_row(batch_size, label_num))
        feats = torch.chunk(feats, batch_size, 0)
        emits = torch.squeeze(torch.cat(feats, 2), 0)
        one_t_row = self.transition[1][:].repeat(batch_size)
        if self.config.use_cuda:
            s_matrix = torch.zeros((sentence_len, label_num * batch_size), dtype=torch.float).cuda()
            mask = mask.cuda()
            last_best_way = last_best_way.cuda()
        else:
            s_matrix = torch.zeros((sentence_len, label_num * batch_size), dtype=torch.float)
        s_matrix[0][:] = emits[1][:] + one_t_row
        for idx in range(1, sentence_len):
            s_row = torch.take(s_matrix[idx - 1][:], mask)
            e_row = self.make_e_row(emits[idx + 1][:], batch_size, label_num)
            t_row = self.make_t_row(self.transition, batch_size, label_num)
            next_tag_var = s_row + e_row + t_row
            s_temp, best_label_id = self.select_max_label(next_tag_var, batch_size, label_num)
            s_matrix[idx][:] = s_temp
            best_way.append(best_label_id)
        t_last_row = self.transition[:, 2].repeat((1, batch_size))
        next_tag_var = s_matrix[-1][:] + t_last_row
        s_temp, best_label_id = self.select_last_label(next_tag_var, batch_size)
        last_best_way[:, 0] = best_label_id
        for idx, row in enumerate(best_way[-1::-1], 1):
            row = torch.chunk(row, batch_size, 0)
            for jdx, content in enumerate(zip(best_label_id, row)):
                last_best_way[jdx][idx] = content[1][content[0]]
            best_label_id = last_best_way[:, idx]
        for i in range(batch_size):
            a = last_best_way[i][:].tolist()
            a.reverse()
            end_way.append(tar_vocab.i2w(a[0:length[i] - 2]))
        return end_way

    # def encode_score(self, emit, feather):
    #     batch_size = len(feather)
    #     sen_length = feather.size()[1] - 2
    #     if self.config.use_cuda is True:
    #         s_matrix = torch.zeros((batch_size, sen_length, self.label_num), dtype=torch.float).cuda()
    #         s_end_unit = torch.zeros(batch_size, dtype=torch.float).cuda()
    #     else:
    #         s_matrix = torch.zeros((batch_size, sen_length, self.label_num), dtype=torch.float)
    #         s_end_unit = torch.zeros(batch_size, dtype=torch.float)
    #     for idx in range(batch_size):
    #         s_matrix[idx][0][:] = emit[idx][1][:] + self.transition[1][:]
    #         #s_matrix[idx][0][:] = emit[idx][1][:]
    #         for i in range(1, sen_length):
    #             s_matrix[idx][i][:] = self.log_sum_exp(s_matrix[idx][i-1][:], self.transition, emit[idx][i+1][:])
    #         s_end = self.log_sum_exp(s_matrix[idx][-1][:], self.transition[:, 2])
    #         s_end_unit[idx] = s_end
    #     s_end_sum = s_end_unit.sum()
    #     return s_end_sum

    @staticmethod
    def to_scalar(var):
        return var.view(-1).data.tolist()[0]

    def arg_max(self, vec):
        _, idx = torch.max(vec, 1)
        return self.to_scalar(idx)

    @staticmethod
    def make_s_row(batch_size, label_num):
        mask = []
        unit = []
        for i in range(batch_size * label_num):
            unit.append(i)
            if len(unit) == label_num:
                for j in range(label_num):
                    mask.extend(unit)
                unit = []
        return mask

    def make_e_row(self, emit, batch_size, label_num):
        e_row = torch.chunk(emit, batch_size * label_num, 0)
        e_row = torch.tensor(e_row)
        e_row = torch.unsqueeze(e_row, 1)
        if self.config.use_cuda:
            e_row = e_row.cuda()
        e_row = e_row.expand((batch_size * label_num, label_num))
        e_row = torch.chunk(e_row, batch_size * label_num, 0)
        e_row = torch.cat(e_row, 1)
        return e_row

    @staticmethod
    def make_t_row(transition, batch_size, label_size):
        transition = torch.transpose(transition, 0, 1)
        transition = torch.chunk(transition, label_size, 0)
        transition = torch.cat(transition, 1)
        transition = transition.repeat((1, batch_size))
        return transition

    @staticmethod
    def select_max_label(next_tag_var, batch_size, label_num):
        next_tag_var = torch.chunk(next_tag_var, batch_size * label_num, 1)
        next_tag_var = torch.cat(next_tag_var, 0)
        s_temp, max_label = torch.max(next_tag_var, 1)
        return s_temp, max_label

    @staticmethod
    def select_last_label(next_tag_var, batch_size):
        next_tag_var = torch.chunk(next_tag_var, batch_size, 1)
        next_tag_var = torch.cat(next_tag_var, 0)
        s_temp, max_label = torch.max(next_tag_var, 1)
        return s_temp, max_label

    def loss_log(self, emit, target, length):
        #gold_score = self.calc_sentences_scores(emit, target, length)
        gold_score = self.sentences_score(emit, target)
        forward_score = self.encode_score(emit)
        print('forward_score, gold_score:', float(forward_score.data), float(gold_score.data))
        return forward_score - gold_score