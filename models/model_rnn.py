#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils import Variable
from MCMG_utils.utils import Variable

#device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.embedding_zcc = nn.Embedding(voc_size, 512)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        #h_out = Variable(torch.zeros(h.size()))
        h_out = torch.zeros(h.size())
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out
    #'''
    def init_h_zcc(self,con_token):
        h_i0 = self.embedding_zcc(con_token)
        h_i1 = self.embedding_zcc(con_token)
        h_i2 = self.embedding_zcc(con_token)
        h_i = torch.stack((h_i0,h_i1,h_i2),dim =0)
        h_i = torch.sum(h_i,dim=2)
        return h_i
    #'''


    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc,device):
        self.rnn = MultiGRU(voc.vocab_size)
        #if torch.cuda.is_available():
        self.device = device
        self.rnn.to(self.device)
        self.voc = voc

    def likelihood(self, target,con_len):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        target = target.to(self.device)
        batch_size, seq_length = target.size()

        start_token = Variable(torch.zeros(batch_size, 1).long()).to(self.device)
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, con_len:-1]), 1).to(self.device)
        con_token = target[:, 0:con_len]
        '''
        print('target',target)  # 66 len
        print(target.size())
        print('x:',x)   # 62
        print(x.size())
        print('con',con_token) #
        exit()      
        '''

        '''
        h = self.rnn.init_h(batch_size)
        print('original_h',h.size()) # [3,128,512]
        print('con_token:',con_token)
        '''
        #h = self.rnn.init_h(batch_size)
        h = self.rnn.init_h_zcc(con_token).to(self.device)
        y_target = target[:, con_len:]
        log_probs = Variable(torch.zeros(batch_size)).to(self.device)
        # entropy = Variable(torch.zeros(batch_size))
        seq_length = seq_length - con_len
        #print('seq_length',seq_length)
        #print('x_size',x.size())
        for step in range(seq_length):
            #print('rnn step is',step)
            #print('x[:, step]',x[:, step].device)
            #print('h_size',h.device)
            logits, h = self.rnn(x[:, step], h)
            logits = logits.to(self.device)
            h = h.to(self.device)
            log_prob = F.log_softmax(logits)
            #prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, y_target[:,step],self.device)
            # entropy += -torch.sum((log_prob * prob), 1)
        return log_probs

    def sample(self, batch_size, max_length=140,token_list=[]):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        if len(token_list)>0:
            con_token = []
            for token in token_list:
                con_token.append(self.voc.vocab[token])
            con_token = torch.tensor(con_token).long().to(self.device)
            con_token = con_token.expand(batch_size,-1)
            h = self.rnn.init_h_zcc(con_token)
        else:
            h = self.rnn.init_h(batch_size)
        h = h.to(self.device)
        start_token = Variable(torch.zeros(batch_size).long()).to(self.device)
        start_token[:] = self.voc.vocab['GO']
        x = start_token


        sequences = []
        log_probs = Variable(torch.zeros(batch_size)).to(self.device)
        #print('batch is:',batch_size)

        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size)).to(self.device)
        if torch.cuda.is_available():
            finished = finished.to(self.device)

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            logits = logits.to(self.device)
            h = h.to(self.device)
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob, 1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x, self.device)
            # print(log_probs.shape)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data).to(self.device)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
        # a = sequences[1:3]
        # b = torch.cat(a, 1)
        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

def NLLLoss(inputs, targets,device):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).to(device)
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
