import pickle
import os
import codecs
import re


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_data(file):
    src_list = []
    tar_list = []
    word_list = []
    label_list = []
    local_unit = []
    local_list = []
    local_num = []
    count = 0
    temp = ''
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip()
                line = line.split(' ')
                if len(word_list) == 0:
                    word_list.append('start')
                    label_list.append('start')
                word_src = normalize_word(line[0])
                word_list.append(word_src)
                label_list.append(line[-1])
                if line[-1] != 'O':
                    local_list.append(line[-1])
                    local_num.append(count)
                    count += 1
                    temp = line[-1]
                else:
                    if local_num:
                        local_list.append(local_num)
                        local_unit.append(local_list)
                    local_list = []
                    local_num = []
                    count += 1
            elif line == '\n':
                if temp != 'O':
                    if local_num:
                        local_list.append(local_num)
                        local_unit.append(local_list)
                    local_list = []
                    local_num = []
                    count += 1
                #src_list.append([word_list, local_unit])
                word_list.append('end')
                label_list.append('end')
                src_list.append([word_list, label_list])
                tar_list.append(label_list)
                word_list = []
                label_list = []
                local_unit = []
                count = 0
    return src_list, tar_list


def read_pkl(pkl):
    file_pkl = codecs.open(pkl, 'rb')
    return pickle.load(file_pkl)


if __name__ == '__main__':
    file = '../Data/text.txt'
    file2 = '../Data/text1.txt'
    file1 = '../Data/Conll2003_BMES/train.txt'
    unit, unit1 = read_data(file1)
    label_kind = []
    for line in unit1:
        for label in line:
            label_kind.append(label)
    kind = set(label_kind)
    print(kind, len(kind))
