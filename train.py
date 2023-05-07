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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, configuration):
        super(EncoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.bidirectional = configuration['bi_directional']
        self.batch_size = configuration['batch_size']
        self.hidden_size = configuration['hidden_size']

        self.dropout = nn.Dropout(configuration['drop_out'])
        self.embedding = nn.Embedding(input_size, self.embedding_size)

        self.cell_layer = None
        self.cell_type = configuration["cell_type"]
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
        if self.cell_type == 'GRU':
            self.cell_layer = nn.GRU(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
        if self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
 
    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input).view(1,self.batch_size, -1))
        output = embedded
        output, hidden = self.cell_layer(output, hidden)
        return output, hidden

    def initHidden(self , num_layers):
        if (self.bidirectional==False):
            res = torch.zeros(num_layers, self.batch_size, self.hidden_size)
        else:
            res = torch.zeros(num_layers*2, self.batch_size, self.hidden_size)
        if use_cuda : 
            return res.cuda()
        else :
            return res
        

class DecoderRNN(nn.Module):
    def __init__(self, configuration,  output_size):
        super(DecoderRNN, self).__init__()

        self.embedding_size = configuration['embedding_size']
        self.bidirectional = configuration['bi_directional']
        self.hidden_size = configuration['hidden_size']
        self.batch_size = configuration['batch_size']

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.dropout = nn.Dropout(configuration['drop_out'])

        self.cell_layer = None
        self.cell_type = configuration["cell_type"]
        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        if self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        if self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        
        if (self.bidirectional==False):
            self.out = nn.Linear(self.hidden_size , output_size)
        else:
            self.out = nn.Linear(self.hidden_size*2 , output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        
        output = self.dropout(self.embedding(input).view(1,self.batch_size, -1))
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
        
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        if (self.bidirectional==False):
            res = torch.zeros(self.num_layers_decoder , self.batch_size, self.hidden_size)
        else:
            res = torch.zeros(self.num_layers_decoder*2 , self.batch_size, self.hidden_size)
        if use_cuda : 
            return res.cuda()
        else :
            return res