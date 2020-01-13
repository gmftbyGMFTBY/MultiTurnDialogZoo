#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import numpy as np
import ipdb

from .layers import *


'''
Dynamic-Static HRED model with the attention on context encoder

COLING 2018 Context-Sensitive Generation of Open-Domain Conversational Responses
@inproceedings{Zhang2018ContextSensitiveGO,
  title={Context-Sensitive Generation of Open-Domain Conversational Responses},
  author={Weinan Zhang and Yiming Cui and Yifa Wang and Qingfu Zhu and Lingzhi Li and Lianqiang Zhou and Ting Liu},
  booktitle={COLING},
  year={2018}
}
'''

class DSUtterance_encoder(nn.Module):

    '''
    Bidirectional GRU
    '''

    def __init__(self, input_size, embedding_size, 
                 hidden_size, dropout=0.5, n_layer=1, pretrained=None):
        super(DSUtterance_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer

        if pretrained:
            pretrained = f'{pretrained}/ipt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(input_size, embedding_size, pretrained)
        else:
            self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, 
                          dropout=dropout, bidirectional=True)
        # hidden_project
        self.hidden_proj = nn.Linear(n_layer * 2 * self.hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(num_features=hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.hidden_proj.weight)
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        # use pack_padded
        # inpt: [seq_len, batch], lengths: [batch_size]
        embedded = self.embed(inpt)    # [seq_len, batch, input_size]

        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), 
                                 self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)    
        # [n_layer * bidirection, batch, hidden_size]
        # hidden = hidden.reshape(hidden.shape[1], -1)
        # ipdb.set_trace()
        hidden = hidden.permute(1, 0, 2)    # [batch, n_layer * bidirectional, hidden_size]
        hidden = hidden.reshape(hidden.size(0), -1) # [batch, *]
        hidden = self.bn(self.hidden_proj(hidden))
        hidden = torch.tanh(hidden)   # [batch, hidden]
        return hidden


class DSContext_encoder(nn.Module):

    '''
    input_size is 2 * utterance_hidden_size
    '''

    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(DSContext_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.drop = nn.Dropout(p=dropout)

        self.attn = Attention(hidden_size)
        self.init_weight()

    def init_weight(self):
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)

    def forward(self, inpt, hidden=None):
        # inpt: [turn_len, batch, input_size]
        # hidden
        # ALSO RETURN THE STATIC ATTENTION
        if not hidden:
            hidden = torch.randn(1, inpt.shape[1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
        
        inpt = self.drop(inpt)
        # outpput: [Seq, batch, hidden_size]
        output, hidden = self.gru(inpt, hidden)
        
        # static attention
        static_attn = self.attn(output[0].unsqueeze(0), output)
        static_attn = static_attn.bmm(output.transpose(0, 1))
        static_attn = static_attn.transpose(0, 1)    # [1, batch, hidden]

        # hidden: [1, batch, hidden_size]
        hidden = hidden.squeeze(0)    # [batch, hidden_size]
        return static_attn, output, hidden
        


class DSDecoder(nn.Module):

    '''
    Max likelyhood for decoding the utterance
    input_size is the size of the input vocabulary

    Attention module should satisfy that the decoder_hidden size is the same as 
    the Context encoder hidden size
    '''

    def __init__(self, output_size, embed_size, hidden_size, pretrained=None):
        super(DSDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        if pretrained:
            pretrained = f'{pretrained}/opt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(self.output_size, self.embed_size, pretrained)
        else:
            self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # attention on context encoder
        self.attn = Attention(hidden_size)

        self.init_weight()

    def init_weight(self):
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)

    def forward(self, inpt, last_hidden, encoder_outputs, static_attn):
        # inpt: [batch_size], last_hidden: [1, batch, hidden_size]
        # static_attn: [1, batch, hidden_size]
        # encoder_outputs: [turn_len, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch_size, embed_size]
        last_hidden = last_hidden.squeeze(0)    # [batch, hidden]

        # [batch, 1, seq_len]
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)    # [1, batch, hidden], dynmaic attn
        
        # combine dynamic attn and static attn
        rnn_input = torch.cat([embedded, context, static_attn], 2)   # [1, batch, 3 * hidden]

        # output: [1, batch, hidden_size], hidden: [1, batch, hidden_size]
        output, hidden = self.gru(rnn_input, last_hidden.unsqueeze(0))
        output = output.squeeze(0)    # [batch, hidden_size]
        # context = context.squeeze(0)  # [batch, hidden]
        # output = torch.cat([output, context], 1)    # [batch, 2 * hidden]
        output = self.out(output)     # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden


class DSHRED(nn.Module):

    def __init__(self, embed_size, input_size, output_size, 
                 utter_hidden, context_hidden, decoder_hidden, 
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1,
                 pretrained=None):
        super(DSHRED, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = DSUtterance_encoder(input_size, embed_size, utter_hidden, 
                                               dropout=dropout, n_layer=utter_n_layer,
                                               pretrained=pretrained)
        self.context_encoder = DSContext_encoder(utter_hidden, context_hidden, 
                                               dropout=dropout) 
        self.decoder = DSDecoder(output_size, embed_size, decoder_hidden, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        # src: [turns, lengths, batch], tgt: [lengths, batch]
        # lengths: [turns, batch]
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        # utterance encoding
        turns = []
        for i in range(turn_size):
            # sbatch = src[i].transpose(0, 1)    # [seq_len, batch]
            hidden = self.utter_encoder(src[i], lengths[i])    # utter_hidden
            turns.append(hidden)
        turns = torch.stack(turns)    # [turn_len, batch, utter_hidden]

        # context encoding
        # output: [seq, batch, hidden], [batch, hidden]
        # static_attn: [1, batch, hidden]
        static_attn, context_output, hidden = self.context_encoder(turns)

        # decoding
        # tgt = tgt.transpose(0, 1)        # [seq_len, batch]
        hidden = hidden.unsqueeze(0)     # [1, batch, hidden_size]
        output = tgt[0, :]          # [batch]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output, static_attn)
                outputs[t] = output
                output = tgt[t].clone().detach()
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output, static_attn)
                outputs[t] = output
                output = torch.max(output, 1)[1]

        return outputs    # [maxlen, batch, vocab_size]


    def predict(self, src, maxlen, lengths, loss=False):
        # predict for test dataset, return outputs: [maxlen, batch_size]
        # src: [turn, max_len, batch_size], lengths: [turn, batch_size]
        turn_size, batch_size = len(src), src[0].size(1)
        outputs = torch.zeros(maxlen, batch_size)
        floss = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
            floss = floss.cuda()

        turns = []
        for i in range(turn_size):
            # sbatch = src[i].transpose(0, 1)
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)

        # context encoding
        # output: [seq, batch, hidden], [batch, hidden]
        # static_attn: [1, batch, hidden]
        static_attn, context_output, hidden = self.context_encoder(turns)
        
        hidden = hidden.unsqueeze(0)

        output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
        if torch.cuda.is_available():
            output = output.cuda()

        for i in range(1, maxlen):
            output, hidden = self.decoder(output, hidden, context_output, static_attn)
            floss[i] = output
            output = output.max(1)[1]
            outputs[i] = output
            
        if loss:
            return outputs, floss
        else:
            return outputs


if __name__ == "__main__":
    pass
