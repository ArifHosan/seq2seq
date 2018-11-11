from __future__ import unicode_literals, print_function, division
import matplotlib.pyplot as plt
from io import open, StringIO
import unicodedata
import string
import re
import random
from io import BytesIO
from tensorflow.python.lib.io import file_io
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

plt.switch_backend('agg')
from Language import Lang

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?।])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    is_good_length = len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    return is_good_length


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(0)
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)


def save_model_param(model, path):
    torch.save(model.state_dict(), path)


def load_model_param(model, path):
    model.load_state_dict(torch.load(path))
    return model


def read_test():
    print("Reading test...")
    lines = open('data/test-data-final.txt', encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    return pairs


def check_if_unk(lang, sentence):
    for word in sentence.split(' '):
        if word not in lang.word2index:
            return word
    return ''


def read_dict():
    print("Reading dict...")
    lines = open('data/en_bn_2_col.tsv', encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    dict = {}
    for pair in pairs:
        if len(pair) != 2:
            continue
        if not pair[0] in dict:
            dict[pair[0]] = pair[1]
    return dict


word_dict = read_dict()
