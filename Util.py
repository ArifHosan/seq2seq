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
MAX_LENGTH = 30
TRAIN_PERCENTAGE = 80
TRAIN_SET = None
TEST_SET = None

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?।])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    lines = open('data/sust.txt', encoding='utf-8').read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    train_set, test_set = split_lang(pairs)
    print(len(train_set))
    print(len(test_set))
    pairs = train_set
    global TEST_SET, TRAIN_SET
    TEST_SET = test_set
    TRAIN_SET = train_set


    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    try:
        is_good_length = len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
        return is_good_length
    except:
        return False


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
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
    index_array = []
    for word in sentence.split(' '):
        try:
            index_array.append(lang.word2index[word])
        except:
            index_array.append(0)
    return index_array


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


def split_lang(sentence_pairs):
    random.shuffle(sentence_pairs)
    total_len = len(sentence_pairs)
    train_len = int(total_len * TRAIN_PERCENTAGE / 100.0)
    return sentence_pairs[:train_len],sentence_pairs[train_len:]


def get_test_set():
    return TEST_SET


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)


def save_model_param(model, path):
    torch.save(model.state_dict(), path)


def load_model_param(model, path):
    model.load_state_dict(torch.load(path))
    return model
