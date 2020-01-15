#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.2.24

'''
Seq2Seq with Attention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import random
import numpy as np
import pickle
import ipdb
import sys

from .layers import * 


class Encoder(nn.Module):
    
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5, pretrained=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.embed_size = embed_size
        self.n_layer = n_layers
        
        if pretrained:
            pretrained = f'{pretrained}/ipt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(self.input_size, self.embed_size, pretrained)
        else:
            self.embed = nn.Embedding(self.input_size, self.embed_size)

        self.input_dropout = nn.Dropout(p=dropout)
        
        self.rnn = nn.GRU(embed_size, 
                          hidden_size, 
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

        self.hidden_proj = nn.Linear(2 * n_layers * hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
            
        self.init_weight()
            
    def init_weight(self):
        # orthogonal init
        init.orthogonal_(self.rnn.weight_hh_l0)
        init.orthogonal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, src, inpt_lengths, hidden=None):
        # src: [seq, batch]
        embedded = self.embed(src)    # [seq, batch, embed]
        embedded = self.input_dropout(embedded)

        if not hidden:
            hidden = torch.randn(2 * self.n_layer, src.shape[-1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
        
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, 
                                                     enforce_sorted=False)
        # hidden: [2 * n_layer, batch, hidden]
        # output: [seq_len, batch, 2 * hidden_size]
        output, hidden = self.rnn(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # fix output
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        # fix hidden
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(hidden.shape[0], -1)
        hidden = self.bn(self.hidden_proj(hidden))    # [batch, *]
        hidden = torch.tanh(hidden)
        
        # [seq_len, batch, hidden_size], [batch, hidden]
        return output, hidden
    
    
class Decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, output_size, pretrained=None):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size
        
        if pretrained:
            pretrained = f'{pretrained}/opt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(output_size, embed_size, pretrained)
        else:
            self.embed = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size) 
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.init_weight()
        
    def init_weight(self):
        # orthogonal init
        init.orthogonal_(self.rnn.weight_hh_l0)
        init.orthogonal_(self.rnn.weight_ih_l0)
        
    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch]
        # last_hidden: [batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch, embed_size]
        
        # attn_weights: [batch, 1, timestep of encoder_outputs]
        attn_weights = self.attention(last_hidden, encoder_outputs)
            
        # context: [batch, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden.unsqueeze(0))
        output = output.squeeze(0)
        # context = context.squeeze(0)
        # [batch, hidden * 2]
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)    # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        
        # output: [batch, output_size]
        # hidden: [batch, hidden_size]
        hidden = hidden.squeeze(0)
        return output, hidden
    
    
class Seq2Seq(nn.Module):
    
    '''
    Compose the Encoder and Decoder into the Seq2Seq model
    '''
    
    def __init__(self, input_size, embed_size, output_size, 
                 utter_hidden, decoder_hidden,
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5, 
                 utter_n_layer=1, src_vocab=None, tgt_vocab=None,
                 pretrained=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden,
                               n_layers=utter_n_layer, dropout=dropout,
                               pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, 
                               pretrained=pretrained)
        self.teach_force = teach_force
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        
    def forward(self, src, tgt, lengths):
        # src: [lengths, batch], tgt: [lengths, batch], lengths: [batch]
        # ipdb.set_trace()
        batch_size, max_len = src.shape[1], tgt.shape[0]
        
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        # encoder_output: [seq_len, batch, hidden_size]
        # hidden: [1, batch, hidden_size]
        encoder_output, hidden = self.encoder(src, lengths)
        output = tgt[0, :]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                # output = torch.max(output, 1)[1]
                output = output.topk(1)[1].squeeze().detach()
        
        # [max_len, batch, output_size]
        return outputs
    
    def predict(self, src, maxlen, lengths, loss=True):
        batch_size = src.shape[1]
        outputs = torch.zeros(maxlen, batch_size)
        floss = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
            floss = floss.cuda()
            
        encoder_output, hidden = self.encoder(src, lengths)
        output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
        if torch.cuda.is_available():
            output = output.cuda()
        
        for t in range(1, maxlen):
            output, hidden = self.decoder(output, hidden, encoder_output)
            floss[t] = output
            # output = torch.max(output, 1)[1]    # [1]
            output = output.topk(1)[1].squeeze().detach()
            outputs[t] = output    # output: [1, output_size]
        
        if loss:
            return outputs, floss
        else:
            return outputs 


if __name__ == "__main__":
    pass
