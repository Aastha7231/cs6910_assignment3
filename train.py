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
        

def indexesFromWord(lang, word):
    index_list = []
    i=0
    while(i < len(word)):
        if word[i] in lang.char2index.keys():
            index_list.append(lang.char2index[word[i]])
        else:
            index_list.append(UNK_token)
        i += 1
    return index_list


def variableFromSentence(lang, word, max_length):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    i=0
    while(i < len(pairs)):
        input_variable = variableFromSentence(input_lang, pairs[i][0], max_length)
        output_variable = variableFromSentence(output_lang, pairs[i][1], max_length)
        res.append((input_variable, output_variable))
        i+=1
    return res

def train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length):
    batch_size = configuration['batch_size']

    encoder_hidden = encoder.initHidden(configuration['num_layers_encoder'])

    if configuration["cell_type"] == "LSTM":
        encoder_cell_state = encoder.initHidden(configuration['num_layers_encoder'])
        encoder_hidden = (encoder_hidden, encoder_cell_state)

    input_tensor = Variable(input_tensor.transpose(0, 1))
    output_tensor = Variable(output_tensor.transpose(0, 1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss ,i = 0, 0

    input_length = input_tensor.size(0)
    output_length = output_tensor.size(0)

    while(i < (input_length)):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        i+=1

    decoder_input = Variable(torch.LongTensor([SOS_token]*batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing :
        i=0
        while(i < (output_length)) :
            decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            decoder_input = output_tensor[i]
            loss += criterion(decoder_output, output_tensor[i])
            i+=1

    else :
        j=0
        while(j < (output_length)) :
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, output_tensor[j])
            j+=1
            
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length


def evaluate(encoder, decoder, loader, configuration, criterion , max_length, output_lang):

    batch_size = configuration['batch_size']
    loss = 0
    total = 0
    correct = 0
    
    for batch_x, batch_y in loader:
        batch_loss = 0
        
        encoder_hidden = encoder.initHidden(configuration['num_layers_encoder'])
        if configuration["cell_type"] == "LSTM":
                    encoder_cell_state = encoder.initHidden(configuration['num_layers_encoder'])
                    encoder_hidden = (encoder_hidden, encoder_cell_state)

        input_variable = Variable(batch_x.transpose(0, 1))
        output_variable = Variable(batch_y.transpose(0, 1))
        
        input_length = input_variable.size()[0]
        target_length = output_variable.size()[0]

        output = torch.LongTensor(target_length, batch_size)

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        
        i=0
        while(i < (input_length)) :
            encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
            i+=1

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        j=0
        while(j < (target_length)) :
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            batch_loss += criterion(decoder_output, output_variable[j].squeeze())

            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))
            output[j] = torch.cat(tuple(topi))
            j+=1

        output = output.transpose(0,1)

        k=0
        while(k < (output.size()[0])):
            ignore = [SOS_token, EOS_token, PAD_token]
            sent = [output_lang.index2char[letter.item()] for letter in output[k] if letter not in ignore]
            y = [output_lang.index2char[letter.item()] for letter in batch_y[k] if letter not in ignore]
            # print(sent,' ',y)
            if sent == y:
                correct += 1
            total += 1
            k+=1
        accuracy = (correct/total)*100
        loss += batch_loss.item()/target_length
    return accuracy, loss

def trainIters(encoder, decoder, train_loader, val_loader, configuration, max_len, max_len_all, output_lang):

    encoder_optimizer = optim.NAdam(encoder.parameters(),lr=configuration['learning_rate'])
    decoder_optimizer = optim.NAdam(decoder.parameters(),lr=configuration['learning_rate'])

    criterion = nn.NLLLoss()
    
    ep = 10

    i=0
    while(i < (ep)) :
        print('ep : ',i)
        train_loss = 0
        print('training..')
        batch_no = 1
        for batchx, batchy in train_loader:
            loss = None

            if configuration['attention'] == False:
                loss = train(batchx, batchy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, configuration, max_len_all)
            
            train_loss += loss
            batch_no+=1

        print('train loss :', train_loss/len(train_loader))
        validation_accuracy , validation_loss= evaluate(encoder, decoder, val_loader, configuration, criterion, max_len, output_lang)
        print('validation loss :', validation_loss/len(val_loader))
        print("val_accuracy : ",validation_accuracy)
        i+=1
        wandb.log({'validation_loss': validation_loss/len(val_loader), 'validation_accuracy': validation_accuracy, 'train_loss': train_loss/len(train_loader)})
            
    # test_acc = evaluate(encoder, decoder, test_loader, configuration, criterion)
    # print("test_accuracy : ",test_acc)


def sweepfunction():
    config = None
    with wandb.init(config = config, entity = 'cs22m005') as run:
        config = wandb.config
        run.name='hl_'+str(config.hidden_size)+'_bs_'+str(config.batch_size)+'_ct_'+config.cell_type
        configuration = {
            "hidden_size" : config.hidden_size,
            "input_lang" : 'eng',
            "output_lang" : 'hin',
            "cell_type"   : config.cell_type,
            "num_layers_encoder" : config.num_layers_encoder ,
            "num_layers_decoder" : config.num_layers_encoder,
            "drop_out"    : config.drop_out, 
            "embedding_size" : config.embedding_size,
            "bi_directional" : config.bidirectional,
            "batch_size" : config.batch_size,
            "attention" : False ,
            "learning_rate" : config.learning_rate
    
        }
        
        train_path = os.path.join(dir, lang_2, lang_2 + '_train.csv')
        validation_path = os.path.join(dir, lang_2, lang_2 + '_valid.csv')
        test_path = os.path.join(dir, lang_2, lang_2 + '_test.csv')
        
        input_lang, output_lang, pairs, max_input_length, max_target_length = prepareData(train_path, lang_1, lang_2)
        val_input_lang, val_output_lang, val_pairs, max_input_length_val, max_target_length_val = prepareData(validation_path, lang_1, lang_2)
        test_input_lang, test_output_lang, test_pairs, max_input_length_test, max_target_length_test = prepareData(test_path, lang_1, lang_2)
        print(random.choice(pairs))

        max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]

        max_len_all = 0
        for i in range(len(max_list)):
            if(max_list[i] > max_len_all):
                max_len_all = max_list[i]

        max_len = max(max_input_length, max_target_length) + 2

        
        pairs = variablesFromPairs(input_lang, output_lang, pairs, max_len)
        val_pairs = variablesFromPairs(input_lang, output_lang, val_pairs, max_len_all)
        # test_pairs = variablesFromPairs(input_lang, output_lang, test_pairs, max_len_all)

        encoder1 = EncoderRNN(input_lang.n_chars, configuration)
        decoder1 = DecoderRNN(configuration, output_lang.n_chars)
        if use_cuda:
            encoder1=encoder1.cuda()
            decoder1=decoder1.cuda()

        train_loader = torch.utils.data.DataLoader(pairs, batch_size=configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size=configuration['batch_size'], shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_pairs, batch_size=configuration['batch_size'], shuffle=True)
        print("done")
        if configuration['attention'] == False :
            trainIters(encoder1, decoder1, train_loader, val_loader, configuration, max_len, max_len_all, output_lang)


# wandb.agent(sweep_id, sweepfunction, count = 50)

def final_run():
    
        configuration = {
            "hidden_size" : 512,
            "input_lang" : 'eng',
            "output_lang" : 'hin',
            "cell_type"   : 'LSTM',
            "num_layers_encoder" : 3 ,
            "num_layers_decoder" : 3,
            "drop_out"    : 0, 
            "embedding_size" : 64,
            "bi_directional" : True,
            "batch_size" : 128,
            "attention" : False ,
            "learning_rate" : 0.001,
    
        }
        
        train_path = os.path.join(dir, lang_2, lang_2 + '_train.csv')
        validation_path = os.path.join(dir, lang_2, lang_2 + '_valid.csv')
        test_path = os.path.join(dir, lang_2, lang_2 + '_test.csv')
        
        input_lang, output_lang, pairs, max_input_length, max_target_length = prepareData(train_path, lang_1, lang_2)
        val_input_lang, val_output_lang, val_pairs, max_input_length_val, max_target_length_val = prepareData(validation_path, lang_1, lang_2)
        test_input_lang, test_output_lang, test_pairs, max_input_length_test, max_target_length_test = prepareData(test_path, lang_1, lang_2)
        print(random.choice(pairs))

        max_list = [max_input_length, max_target_length, max_input_length_val, max_target_length_val, max_input_length_test, max_target_length_test]

        max_len_all = 0
        for i in range(len(max_list)):
            if(max_list[i] > max_len_all):
                max_len_all = max_list[i]
        max_len_all+=1

        pairs = variablesFromPairs(input_lang, output_lang, pairs, max_len_all)
        val_pairs = variablesFromPairs(input_lang, output_lang, val_pairs, max_len_all)
        test_pairs = variablesFromPairs(input_lang, output_lang, test_pairs, max_len_all)

        encoder1 = EncoderRNN(input_lang.n_chars, configuration)
        decoder1 = DecoderRNN(configuration, output_lang.n_chars)
        
        if use_cuda:
            encoder1=encoder1.cuda()
            decoder1=decoder1.cuda()

        train_loader = torch.utils.data.DataLoader(pairs, batch_size = configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size = configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size = configuration['batch_size'], shuffle=True)
        print("done")
        if configuration['attention'] == False :
            trainIters(encoder1, decoder1, train_loader, val_loader,test_loader, configuration, max_len_all, output_lang)
        else:
            trainIters(encoder1, decoder1, train_loader, val_loader,test_loader, configuration, max_len_all, output_lang)

final_run()
