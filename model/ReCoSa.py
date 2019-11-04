#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.20


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import ipdb

from .layers import * 


'''
ReCoSa model
'''


class Encoder(nn.Module):

    '''
    In the paper, the Encoder is the GRUCell without bidirection, num_layers can be controlled
    In the author's implement, the n_layer is 1
    '''

    def __init__(self, input_size, embed_size, hidden_size, 
                 n_layers=1, dropout=0.5, pretrained=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layer = n_layers

        if pretrained:
            pretrained = f"{pretrained}/ipt_bert_embedding.pkl"
            self.embed = PretrainedEmbedding(input_size, embedd_size, pretrained)
        else:
            self.embed = nn.Embedding(self.input_size, self.embed_size)
        self.input_dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.hidden_proj = nn.Linear(n_layers * hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(num_features=hidden_size)

        self.init_weight()

    def init_weight(self):
        init.orthogonal_(self.rnn.weight_hh_l0)
        init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, src, inpt_lengths, hidden=None):
        embedded = self.embed(src)
        embedded = self.input_dropout(embedded)

        #if not hidden:
        #    hidden = torch.randn(2 * self.n_layer, src.shape[-1], self.hidden_size)
        #    if torch.cuda.is_available():
        #        hidden = hidden.cuda()

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, 
                                                     enforce_sorted=False)

        # output: [seq_le, batch, hidden], hidden: [n_layer, batch, hidden_size]
        _, hidden = self.rnn(embedded, hidden)

        hidden = hidden.permute(1, 0, 2)    # [batch, n_layer, hidden_size]
        hidden = hidden.reshape(hidden.shape[0], -1)    # [batch, n_layer * hidden_size]
        hidden = self.bn(self.hidden_proj(hidden))      # [batch, hidden_size]
        hidden = torch.tanh(hidden)

        return hidden


class ReCoSa(nn.Module):

    '''
    ReCoSa implemented by the pytorch 1.2 Transformer

    encoding mask: padding mask
    decoding mask: sequence mask & padding mask
    '''

    def __init__(self, input_size, embed_size, utter_hidden_size, output_size, 
                 dropout=0.5, n_layers=1, sos=24744, pad=24745, pretrained=None):
        super(ReCoSa, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden_size, 
                               n_layers=n_layers, dropout=dropout)
        self.transformer = nn.Transformer(utter_hidden_size, utter_hidden_size)
        self.out = nn.Linear(embed_size, output_size)
        if pretrained:
            pretrained = f"{pretrained}/opt_bert_embedding.pkl"
            self.output_emb = PretrainedEmbedding(output_size, embed_size, pretrained)
        else:
            self.output_emb = nn.Embedding(output_size, embed_size)
        self.pos_emb = PositionEmbedding(embed_size, dropout=dropout) 

        self.embed_size = embed_size
        self.dropout = dropout
        self.output_size = output_size
        self.sos, self.tgt_pad = sos, pad

    def forward(self, src, tgt, lengths):
        # src: [turns, lengths, batch], tgt: [lengths, batch], lengths: [turns, batch]
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)

        # utterance encode
        turns = []
        for i in range(turn_size):
            hidden = self.encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        
        # position embedding
        turns = self.pos_emb(turns)

        # ReCoSa
        # turns: [turn_len, batch, hidden], tgt: [seq_len, batch]

        # output: [seq_len, batch, embed]
        # creating mask
        # tgt_mask: decoder sequence mask [maxlen, maxlen]
        # src_mask: donot need this mask attention matrix
        # src_key_padding_mask: donot need this mask attention matrix (because of context)
        # tgt_key_padding_mask: [batch_size, seq_len] 
        tgt_mask = self.transformer.generate_square_subsequent_mask(maxlen)
        tgt_key_padding_mask = tgt.transpose(0, 1) == self.tgt_pad    # [batch, seq_len]
        if torch.cuda.is_available():
            tgt_mask = tgt_mask.cuda()
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        
        # map the tgt
        tgt = self.output_emb(tgt)    # [seq_len, batch, embed]
        tgt = self.pos_emb(tgt)

        output = self.transformer(turns, tgt, 
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.out(output)    # [seq_len, batch, output]
        output = F.log_softmax(output, dim=2)

        return output

    def predict(self, src, maxlen, lengths):
        # src: [tunr, maxlen, batch], lengths: [turn, batch]
        turn_size, batch_size = len(src), src[0].size(1)
        # ipdb.set_trace()

        # utterance encode
        turns = []
        for i in range(turn_size):
            hidden = self.encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)

        # position embedding
        turns = self.pos_emb(turns)

        # ReCoSa predict
        tgt_mask = self.transformer.generate_square_subsequent_mask(maxlen)
        tgt = torch.zeros(maxlen, batch_size, dtype=torch.long).fill_(self.sos)  # [seq_len, batch]
        if torch.cuda.is_available():
            tgt = tgt.cuda()
            tgt_mask = tgt_mask.cuda()
        tgt = self.output_emb(tgt)    # [seq_len, batch, embed]
        tgt = self.pos_emb(tgt)
       
        preds = []
        for j in range(1, maxlen):
            tgt = self.transformer(turns, tgt, tgt_mask=tgt_mask)
            # ipdb.set_trace()
            tgt = self.out(tgt)   # [seq_len, batch, output]
            tgt = F.log_softmax(tgt, dim=2)    # [seq_len, batch, output_size]
            p = tgt.max(2)[1]    # [seq_len, batch]
            preds.append(p[j])
            tgt = self.output_emb(p)    # [seq_len, batch, embed]
            tgt = self.pos_emb(tgt)

        tgt = torch.stack(preds)    # [seq_len, batch]
        return tgt    # [seq_len, batch]


if __name__ == "__main__":
    pass
