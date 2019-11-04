#!/usr/bin/python3
# Authro: GMFTBY
# Time: 2019.9.24


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import ipdb
import pickle

from .layers import *

'''
MReCoSa: 
    1. Utterance-encoder: Bi-GRU
    2. Context-encoder: Multi-head attention
    3. Context-attention: Multi-head attention with decoder and encoder
    4. Decoder: GRU for generating the vocab from the vocabulary

details can be found in: https://arxiv.org/abs/1907.05339
'''

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
            self.embed = PretrainedEmbedding(input_size, embed_size, pretrained)
        else:
            self.embed = nn.Embedding(input_size, embed_size)

        self.input_dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.hidden_proj = nn.Linear(n_layers * hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, src, inpt_lengths, hidden=None):
        embedded = self.embed(src)
        embedded = self.input_dropout(embedded)

        if not hidden:
            hidden = torch.randn(self.n_layer, src.shape[-1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, 
                                                     enforce_sorted=False)
        _, hidden = self.rnn(embedded, hidden)

        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(hidden.shape[0], -1)
        hidden = self.bn(self.hidden_proj(hidden))
        hidden = torch.tanh(hidden)

        # [batch, hidden_size]
        return hidden


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

        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.init_weight()
        self.self_attention = nn.MultiheadAttention(hidden_size, 8)

    def init_weight(self):
        init.orthogonal_(self.rnn.weight_hh_l0)
        init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, inpt, last_hidden, context_encoder):
        # inpt: [batch], last_hidden: [batch, hidden_size]
        # context_encoder: [seq_len, batch, hidden]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch, embed_size]

        # attn_weight
        # context: [1, batch, embed], attn_weight: [batch, 1, src_seq_len]
        context, attn_weight = self.self_attention(last_hidden.unsqueeze(0), 
                                                   context_encoder, 
                                                   context_encoder)
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden.unsqueeze(0))
        output = output.squeeze(0)
        # context = context.squeeze(0)

        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)    # [batch, output_size]
        output = F.log_softmax(output, dim=1)

        hidden = hidden.squeeze(0)
        # output: [batch, output_size], hidden: [batch, hidden_size]
        return output, hidden


class MReCoSa(nn.Module):

    def __init__(self, input_size, embed_size, output_size, utter_hidden, 
                 decoder_hidden, teach_force=0.5, pad=1, sos=1, dropout=0.5, 
                 utter_n_layer=1, pretrained=None):
        super(MReCoSa, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, n_layers=utter_n_layer,
                               dropout=dropout, pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, pretrained=pretrained)
        self.teach_force = teach_force
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        self.pos_emb = PositionEmbedding(embed_size, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(embed_size, 8)

    def forward(self, src, tgt, lengths):
        # src: [turn, lengths, batch], tgt: [seq, batch], lengths: [turns, batch]
        turn_size, batch_size, max_len = len(src), tgt.size(1), tgt.size(0)

        # encoder
        turns = []
        for i in range(turn_size):
            hidden = self.encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        turns = self.pos_emb(turns)    # [turn_len, batch, hidden], hidden = [1, batch, hidden]

        # context multi-head attention
        # context: [seq, batch, hidden]
        context, attn_weight = self.self_attention(turns, turns, turns) 
        
        # decode with multi-head attention
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        output = tgt[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, context)
            outputs[t] = output
            is_teacher = np.random.rand() < self.teach_force
            top1 = torch.max(output, 1)[1]
            if is_teacher:
                output = tgt[t].clone().detach()
            else:
                output = top1

        # [seq, batch, output_size]
        return outputs
        
    def predict(self, src, maxlen, lengths, loss=False):
        turn_size, batch_size = len(src), src[0].size(1)

        turns = []
        for i in range(turn_size):
            hidden = self.encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        turns = self.pos_emb(turns)

        outputs = torch.zeros(maxlen, batch_size)
        output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
        floss = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
            output = output.cuda()
            floss = floss.cuda()

        # context multi-head attention
        context, attn_weight = self.self_attention(turns, turns, turns)

        for t in range(1, maxlen):
            output, hidden = self.decoder(output, hidden, context)
            floss[t] = output
            output = torch.max(output, 1)[1]
            outputs[t] = output

        if loss:
            return outputs, floss
        else:
            return outputs


if __name__ == "__main__":
    pass
