#!/usr/bin/env Python
# coding=utf-8
import json
from pprint import pprint

import codecs

data = json.load(codecs.open('data/BengaliDictionary.json', 'r', 'utf-8-sig'))

total_data = len(data)
with open('data/dictionary.txt', 'w') as a:
    for i in range(total_data):
        a.write(data[i]["en"] + "\t" + data[i]["bn"] + "\n")
        # en_syn_len = len(data[i]["en_syns"])
        # bn_syn_len = len(data[i]["bn_syns"])
        #
        # for j in range(en_syn_len):
        #
        #     for k in range(bn_syn_len):
        #         a.write(data[i]["en_syns"][j] + "\t" + data[i]["bn_syns"][k] + "\n")
