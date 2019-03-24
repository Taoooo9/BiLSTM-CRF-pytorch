from collections import Counter

PAD, UNK = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'


class VocabSrc(object):
    def __init__(self, tra_word, dev_word, test_word, config):
        self.word2id = {}
        self.word2id_lower = {}
        self.id2word = [PAD_S, UNK_S]
        self.id2word_lower = [PAD_S, UNK_S]
        self.UNK = 1
        data = tra_word + dev_word + test_word
        word_counter = Counter()
        for line in data:
            for word in line[0]:
                word_counter[word] += 1
        most_word = [k for k, v in word_counter.most_common()]
        most_word_lower = [k.lower() for k, v in word_counter.most_common()]
        self.id2word = self.id2word + most_word
        self.id2word_lower = self.id2word_lower + most_word_lower
        for idx, word in enumerate(zip(self.id2word, self.id2word_lower)):
            self.word2id[word[0]] = idx
            self.word2id_lower[word[1]] = idx

    def i2w(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]

    def w2i(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word, UNK) for word in xx]
        return self.word2id.get(xx)

    def w2i_lower(self, xx):
        if isinstance(xx, list):
            return [self.word2id_lower.get(word, UNK) for word in xx]
        return self.word2id_lower.get(xx)

    @property
    def getsize(self):
        return len(self.id2word)


class VocabTar(object):
    def __init__(self, tra_label, dev_label, test_label, config):
        self.word2id = {}
        data = tra_label + dev_label + test_label
        label_counter = Counter()
        for line in data:
            for label in line:
                label_counter[label] += 1
        most_label = [k for k, v in label_counter.most_common()]
        self.id2word = most_label
        for idx, label in enumerate(self.id2word):
            self.word2id[label] = idx

    def w2i(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word) for word in xx]
        return self.word2id.get(xx)

    def i2w(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]

    @property
    def getsize(self):
        return len(self.word2id)

