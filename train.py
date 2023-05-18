import pandas as pd
import os
import random
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import argparse
import csv

# parser=argparse.ArgumentParser()

# parser.add_argument('-wp',      '--wandb_project',      help='project name in wandb',                                                    type=str,       default='dlasg3'    )
# parser.add_argument('-we',      '--wandb_entity',       help='entity name in wandb',                                                     type=str,       default='cs22m005'  )
# parser.add_argument('-bd',      '--bidirectional',      help='bidirectional',                   choices=[True,False],                    type=bool,      default=True        )
# parser.add_argument('-at',      '--attention',          help='attention',                       choices=[True,False],                    type=bool,      default=False       )
# parser.add_argument('-b',       '--batch_size',         help='batch sizes',                     choices=[32,64,128],                     type=int,       default=128         )
# parser.add_argument('-lr',      '--learning_rate',      help='learning rates',                  choices=[1e-2,1e-3],                     type=float,     default=1e-3        )
# parser.add_argument('-sz',      '--hidden_size',        help='hidden layer size',               choices=[128,256,512],                   type=int,       default=512         )
# parser.add_argument('-il',      '--input_lang',         help='input language',                  choices=['eng'],                         type=str,       default='eng'       )
# parser.add_argument('-do',      '--drop_out',           help='drop out',                        choices=[0.0,0.2,0.3],                   type=float,     default=0           )
# parser.add_argument('-nle',     '--num_layers_en',      help='layers in encoder',               choices=[1,2,3],                         type=int,       default=3           )
# parser.add_argument('-nld',     '--num_layers_dec',     help='layers in decoder',               choices=[1,2,3],                         type=int,       default=3           )
# parser.add_argument('-es',      '--embedding_size',     help='embedding size',                  choices=[64,128,256],                    type=int,       default=64          )
# parser.add_argument('-ol',      '--output_lang',        help='output language',                 choices=['hin','tel'],                   type=str,       default='hin'       )
# parser.add_argument('-ct',      '--cell_type',          help='cell type',                       choices=['LSTM','GRU','RNN'],            type=str,       default='LSTM'      )

# args=parser.parse_args()

# project_name          = args.wandb_project
# entity_name           = args.wandb_entity
# num_layers_encoder    = args.num_layers_en
# num_layers_decoder    = args.num_layers_dec
# embedding_size        = args.embedding_size
# bidirectional         = args.bidirectional
# batch_size            = args.batch_size
# learning_rate         = args.learning_rate
# hidden_size           = args.hidden_size
# input_lang            = args.input_lang
# output_lang           = args.output_lang
# cell_type             = args.cell_type
# drop_out              = args.drop_out
# attention             = args.attention


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
lang_1 = 'eng'
lang_2 = 'hin'
UNK_token = 3
PAD_token = 4

dir = '/kaggle/input/dataset/aksharantar_sampled'

# sweep_config ={
#     'method':'bayes'
# }

# metric = {
#     'name' : 'validation_accuracy',
#     'goal' : 'maximize'
# }
# sweep_config['metric'] = metric

# parameters_dict={
#     'hidden_size':{
#         'values' : [128,256,512]
#     },
#     'learning_rate':{
#         'values' : [1e-2,1e-3]
#     },
#     'cell_type':{
#         'values' : ['LSTM','RNN','GRU']
#     },
#     'num_layers_encoder':{
#         'values' : [1,2,3]
#     },
#     'num_layers_decoder':{
#         'values' : [1,2,3]
#     },
#     'drop_out':{
#         'values' : [0.0,0.2,0.3]
#     },
#     'embedding_size':{
#         'values' : [64,128,256,512]
#     },
#     'batch_size':{
#         'values' : [32,64,128]
#     },
#     'bidirectional':{
#         'values' : [True,False]
#     }
# }
# sweep_config['parameters'] = parameters_dict

# sweep_id = wandb.sweep(sweep_config, project = 'dlasg3')


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
            self.cell_layer = nn.RNN(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
        if self.cell_type == 'GRU':
            self.cell_layer = nn.GRU(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
        if self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_encoder"], dropout = configuration['drop_out'], bidirectional = configuration['bi_directional'])
 
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
            self.cell_layer = nn.RNN(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        if self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        if self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(configuration['embedding_size'], configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"], bidirectional = configuration["bi_directional"])
        
        if (self.bidirectional==False):
            self.out = nn.Linear(configuration['hidden_size'] , output_size)
        else:
            self.out = nn.Linear(configuration['hidden_size']*2 , output_size)
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

class AttnDecoder(nn.Module):
    def __init__(self, configuration, output_size, max_lengthWord):
        super(AttnDecoder, self).__init__()
        
        self.hidden_size = configuration["hidden_size"]
        self.output_size = output_size
        self.embedding_size = configuration["embedding_size"]
        self.batch_size = configuration["batch_size"]
        self.max_lengthWord =max_lengthWord
        self.max_lengthTensor = self.max_lengthWord


        self.embedding = nn.Embedding(self.output_size , configuration["embedding_size"])
        self.attn = nn.Linear(configuration["embedding_size"] + configuration['hidden_size'], self.max_lengthWord+1)
        self.attn_combine = nn.Linear(configuration["embedding_size"] + configuration['hidden_size'], configuration["embedding_size"])
        self.dropout = nn.Dropout(configuration["drop_out"])
        
        
        self.cell_layer = None
        self.cell_type = configuration["cell_type"]

        if self.cell_type == 'RNN':
            self.cell_layer = nn.RNN(configuration["embedding_size"] , configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"])
        if self.cell_type == 'GRU':
            self.cell_layer =   nn.GRU(configuration["embedding_size"], configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"])
        if self.cell_type == 'LSTM':
            self.cell_layer = nn.LSTM(configuration["embedding_size"] , configuration['hidden_size'], num_layers = configuration["num_layers_decoder"], dropout = configuration["drop_out"])
        

        if configuration["bi_directional"] != True:
            self.out = nn.Linear(configuration['hidden_size'] , self.output_size)
        else:
            self.out = nn.Linear(configuration['hidden_size']*2 , self.output_size)
       

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
      
        if self.cell_type == "LSTM":
          
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        else:
          
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        
        attn_applied = torch.bmm(attn_weights.view(self.batch_size, 1, self.max_lengthWord+1),encoder_outputs).view(1, self.batch_size, -1)
       
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
   
        output = F.relu(output)
        output, hidden = self.cell_layer(output, hidden)
    
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        
        
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

teacher_forcing_ratio = 0.5


def train(input_tensor, output_tensor, encoder, decoder,decoder_attn, encoder_optimizer, decoder_optimizer, criterion, configuration, max_length):
    batch_size = configuration['batch_size']

    encoder_hidden = encoder.initHidden(configuration['num_layers_encoder'])

    if configuration["cell_type"] == "LSTM":
        encoder_cell_state = encoder.initHidden(configuration['num_layers_encoder'])
        encoder_hidden = (encoder_hidden, encoder_cell_state)

    input_tensor = Variable(input_tensor.transpose(0, 1))
    output_tensor = Variable(output_tensor.transpose(0, 1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if configuration['attention']== True:
        encoder_outputs = Variable(torch.zeros(max_length +1, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss ,i = 0, 0

    input_length = input_tensor.size(0)
    output_length = output_tensor.size(0)

    while(i < (input_length)):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        if configuration['attention']== True:
            encoder_outputs[i] += encoder_output[0,0]
        i+=1

    decoder_input = Variable(torch.LongTensor([SOS_token]*batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing :
        i=0
        while(i < (output_length)) :
            if configuration['attention']== False:
                decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden,decoder_attention = decoder_attn(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size, max_length+1 ,encoder.hidden_size))

            decoder_input = output_tensor[i]
            loss += criterion(decoder_output, output_tensor[i])
            i+=1

    else :
        j=0
        while(j < (output_length)) :
            if configuration['attention']== False:
                decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden,decoder_attention = decoder_attn(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size, max_length+1 ,encoder.hidden_size))

            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, output_tensor[j])
            j+=1
            
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length


def evaluate(encoder, decoder,decoder_attn, loader, configuration, criterion ,max_length,output_lang):

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
        
        if configuration['attention']== True:
            encoder_outputs = Variable(torch.zeros(max_length +1, batch_size, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        output = torch.LongTensor(target_length, batch_size)
        
        i=0
        while(i < (input_length)) :
            encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
            if configuration['attention']== True:
                encoder_outputs[i] = encoder_output[0,0]
            i+=1

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        j=0
        while(j < (target_length)) :
            if configuration['attention']== False:
                decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden, decoder_attention = decoder_attn(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size, max_length+1,encoder.hidden_size))
                

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

def get_word(word1, word2, word3):
    output=[]
    s1=''
    s2=''
    s3=''
    for x in word1:
        s1+=x
    for y in word2:
        s2+=y
    for z in word3:
        s3+=z
    output.append(s1)
    output.append(s2)
    output.append(s3)
    return output


def evaluate_testset(encoder, decoder,decoder_attn, loader, configuration, criterion ,max_length,output_lang, input_lang):

    batch_size = configuration['batch_size']
    loss = 0
    total = 0
    correct = 0
    output_words = []
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
        
        if configuration['attention']== True:
            encoder_outputs = Variable(torch.zeros(max_length +1, batch_size, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        output = torch.LongTensor(target_length, batch_size)
        
        i=0
        while(i < (input_length)) :
            encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
            if configuration['attention']== True:
                encoder_outputs[i] = encoder_output[0,0]
            i+=1

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        j=0
        while(j < (target_length)) :
            if configuration['attention']== False:
                decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden, decoder_attention = decoder_attn(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size, max_length+1,encoder.hidden_size))
                

            batch_loss += criterion(decoder_output, output_variable[j].squeeze())

            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))
            output[j] = torch.cat(tuple(topi))
            j+=1

        output = output.transpose(0,1)

        k=0
        while(k < (output.size()[0])):
            ignore = [SOS_token, EOS_token, PAD_token]
            x = [input_lang.index2char[letter.item()] for letter in batch_x[k] if letter not in ignore]
            sent = [output_lang.index2char[letter.item()] for letter in output[k] if letter not in ignore]
            y = [output_lang.index2char[letter.item()] for letter in batch_y[k] if letter not in ignore]
            out = get_word(x, sent, y)
            output_words.append(out)
            if sent == y:
                correct += 1
            total += 1
            k+=1
        accuracy = (correct/total)*100
        loss += batch_loss.item()/target_length
    
    return accuracy, loss, output_words

def test_accuracy(encoder, decoder,decoder_attn, test_loader, configuration, criterion,max_len_all, output_lang, input_lang):

    test_acc, test_loss, test_output = evaluate_testset(encoder, decoder,decoder_attn, test_loader, configuration, criterion,max_len_all,output_lang, input_lang)
    print("Test_accuracy : ",test_acc)
    print("Test_loss : ",test_loss/len(test_loader))
    print()
    print("--------------------***------------------------")
    fields = ['input','prediction','output']
    df=pd.DataFrame(test_output)
    
    if(configuration['attention']==True):
        df.to_csv('predictions_attention.csv', sep = ',', index= 'false', header = fields)
    else:
        df.to_csv('predictions_vanilla.csv', sep = ',', index= 'false', header = fields)
        
#     for k in range(len(test_output)):
#         print(test_output[k])


def trainIters(encoder, decoder,decoder_attn, train_loader, val_loader,test_loader, configuration, max_len_all, output_lang, input_lang):

    encoder_optimizer = optim.NAdam(encoder.parameters(),lr=configuration['learning_rate'])
    decoder_optimizer = optim.NAdam(decoder.parameters(),lr=configuration['learning_rate'])

    criterion = nn.NLLLoss()
    
    ep = 10

    i=0
    while(i < (ep)) :
        print("--------------------***------------------------")
        print('ep : ',i)
        train_loss = 0
        print('training the model..')
        batch_no = 1
        for batchx, batchy in train_loader:
            loss = None

            loss = train(batchx, batchy, encoder, decoder, decoder_attn, encoder_optimizer, decoder_optimizer, criterion, configuration, max_len_all)
            
            train_loss += loss
            batch_no+=1
        print('Train loss :', train_loss/len(train_loader))
        validation_accuracy , validation_loss = evaluate(encoder, decoder,decoder_attn, val_loader, configuration, criterion,max_len_all, output_lang)
        print('validation loss :', validation_loss/len(val_loader))
        print("val_accuracy : ",validation_accuracy)
        i+=1
#         wandb.log({'validation_loss': validation_loss/len(val_loader), 'validation_accuracy': validation_accuracy, 'train_loss': train_loss/len(train_loader)})
        print()
    print("--------------------***------------------------")
    test_accuracy(encoder, decoder, decoder_attn, test_loader, configuration, criterion, max_len_all, output_lang, input_lang)


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
            "bi_directional" : False,
            "batch_size" : config.batch_size,
            "attention" : True ,
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
        max_len_all+=1

        pairs = variablesFromPairs(test_input_lang, test_output_lang, pairs, max_len_all)
        val_pairs = variablesFromPairs(test_input_lang, test_output_lang, val_pairs, max_len_all)
        test_pairs = variablesFromPairs(test_input_lang, test_output_lang, test_pairs, max_len_all)

        encoder1 = EncoderRNN(test_input_lang.n_chars, configuration)
        if use_cuda:
            encoder1=encoder1.cuda()

        decoder1 = DecoderRNN(configuration, test_output_lang.n_chars)
        if use_cuda:
            decoder1=decoder1.cuda()

        decoder_attn = AttnDecoder(configuration, test_output_lang.n_chars, max_len_all)
        if use_cuda:
            decoder_attn = decoder_attn.cuda()


        train_loader = torch.utils.data.DataLoader(pairs, batch_size = configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size = configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size = configuration['batch_size'], shuffle=True)

        if configuration['attention'] == False :
            trainIters(encoder1, decoder1, decoder_attn, train_loader, val_loader,test_loader, configuration, max_len_all, test_output_lang, test_input_lang)
        else:
            trainIters(encoder1, decoder1, decoder_attn, train_loader, val_loader,test_loader, configuration, max_len_all,test_output_lang, test_input_lang)

# wandb.agent(sweep_id, sweepfunction, count = 50)

def final_run():
    
        configuration = {
            "hidden_size" : 512,
            "input_lang" : 'eng',
            "output_lang" : 'hin',
            "cell_type"   : 'LSTM',
            "num_layers_encoder" : 2 ,
            "num_layers_decoder" : 2,
            "drop_out"    : 0.2, 
            "embedding_size" : 128,
            "bi_directional" : True,
            "batch_size" : 32,
            "attention" : False ,
            "learning_rate" : 0.001,
    
        }
        # configuration = {

        #         'hidden_size'         : hidden_size,
        #         'input_lang'          : input_lang,
        #         'output_lang'         : output_lang,
        #         'cell_type'           : cell_type,
        #         'num_layers_encoder'  : num_layers_encoder,
        #         'num_layers_decoder'  : num_layers_encoder,
        #         'drop_out'            : drop_out, 
        #         'embedding_size'      : embedding_size,
        #         'bi_directional'      : bidirectional,
        #         'batch_size'          : batch_size,
        #         'attention'           : attention,
        #         'learning_rate'       : learning_rate,

        #     }


        
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

        pairs = variablesFromPairs(test_input_lang, test_output_lang, pairs, max_len_all)
        val_pairs = variablesFromPairs(test_input_lang, test_output_lang, val_pairs, max_len_all)
        test_pairs = variablesFromPairs(test_input_lang, test_output_lang, test_pairs, max_len_all)

        encoder1 = EncoderRNN(test_input_lang.n_chars, configuration)
        if use_cuda:
            encoder1=encoder1.cuda()

        decoder1 = DecoderRNN(configuration, test_output_lang.n_chars)
        if use_cuda:
            decoder1=decoder1.cuda()

        decoder_attn = AttnDecoder(configuration, test_output_lang.n_chars, max_len_all)
        if use_cuda:
            decoder_attn = decoder_attn.cuda()

        train_loader = torch.utils.data.DataLoader(pairs, batch_size = configuration['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_pairs, batch_size = configuration['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_pairs, batch_size = configuration['batch_size'], shuffle=True)

        if configuration['attention'] == False :
            trainIters(encoder1, decoder1,decoder_attn, train_loader, val_loader,test_loader, configuration, max_len_all, test_output_lang, test_input_lang)
        else:
            trainIters(encoder1, decoder1,decoder_attn, train_loader, val_loader,test_loader, configuration, max_len_all, test_output_lang, test_input_lang)

final_run()
