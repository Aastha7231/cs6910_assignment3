import pandas as pd
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
lang_1 = 'eng'
lang_2 = 'hin'
UNK_token = 3
PAD_token = 4
dir = 'aksharantar_sampled'

train_path = os.path.join(dir, lang_2, lang_2 + '_train.csv')
validation_path = os.path.join(dir, lang_2, lang_2 + '_valid.csv')
test_path = os.path.join(dir, lang_2, lang_2 + '_test.csv')


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.n_chars = 4
        self.index2char = {0: '<', 1: '>',2 : '?', 3:'.'}
        
    def addWord(self, word):
        i=0
        while(i< len(word)):
            self.addChar(word[i])
            i+=1

    def addChar(self, char):
        if char in self.char2index:
            self.char2count[char] += 1
        else:
            self.index2char[self.n_chars] = char
            self.char2count[char] = 1
            self.char2index[char] = self.n_chars
            self.n_chars += 1

def prepareData(dir, lang_1, lang_2):

    data = pd.read_csv(dir, sep=",", names=['input', 'output'])

    input_lang = Lang(lang_1)
    output_lang = Lang(lang_2)

    max_input_length = max([len(txt) for txt in data['input'].to_list()])

    output_list = data['output'].to_list()
    input_list = data['input'].to_list() 

    i=0
    pairs = []
    while(i < (len(input_list))):
        x=[input_list[i],output_list[i]]
        pairs.append(x)
        i+=1

    j=0
    while(j < len(pairs)):
        input_lang.addWord(pairs[j][0])
        output_lang.addWord(pairs[j][1])
        j+=1
    
    max_output_length = max([len(txt) for txt in data['output'].to_list()])

    print("Counted letters:")
    print(input_lang.name, max_input_length)
    print(output_lang.name, max_output_length)
    return input_lang, output_lang, pairs, max_input_length, max_output_length


input_lang, output_lang, pairs, max_input_length, max_target_length = prepareData(train_path, lang_1, lang_2)
val_input_lang, val_output_lang, val_pairs, max_input_length_val, max_target_length_val = prepareData(validation_path, lang_1, lang_2)
test_input_lang, test_output_lang, test_pairs, max_input_length_test, max_target_length_test = prepareData(test_path, lang_1, lang_2)
