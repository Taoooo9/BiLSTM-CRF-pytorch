import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

    def __init__(self, config, embedding, label_kind):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(config.embedding_num, 100, padding_idx=0)
        if config.pre_word_embedding is True:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.lstm = nn.LSTM(input_size=100, hidden_size=config.hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size * 2, config.label_size, bias=True)
        # nn.init.xavier_uniform_(self.lstm.all_weights[0][0])
        # nn.init.xavier_uniform_(self.lstm.all_weights[0][1])
        # nn.init.xavier_uniform_(self.lstm.all_weights[1][0])
        # nn.init.xavier_uniform_(self.lstm.all_weights[1][1])
        self.crf = CRF(label_kind, config)

    def forward(self, x, length):
        x = self.embedding(x)
        x = self.dropout(x)
        packed_words = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.lstm(packed_words)
        x, _ = pad_packed_sequence(x)
        x = torch.transpose(x, 0, 1)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit