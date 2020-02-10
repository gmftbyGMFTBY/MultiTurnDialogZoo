#!/use/bin/python
# Author: GMFTBY
# Time: 2020.2.10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import numpy as np
import ipdb

from .layers import *


'''
Refer to: https://github.com/lipiji/dialogue-hred-vhred
'''


class Utterance_encoder(nn.Module):

    '''
    Bidirectional GRU
    '''

    def __init__(self, input_size, embedding_size, 
                 hidden_size, dropout=0.5, n_layer=1, pretrained=None):
        super(Utterance_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer

        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, 
                          dropout=(0 if n_layer == 1 else dropout), bidirectional=True)
        # hidden_project
        # self.hidden_proj = nn.Linear(n_layer * 2 * self.hidden_size, hidden_size)
        # self.bn = nn.BatchNorm1d(num_features=hidden_size)

        self.init_weight()

    def init_weight(self):
        # init.xavier_normal_(self.hidden_proj.weight)
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
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

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                                     enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)    
        # [n_layer * bidirection, batch, hidden_size]
        # hidden = hidden.reshape(hidden.shape[1], -1)
        # ipdb.set_trace()
        hidden = hidden.sum(axis=0)    # [4, batch, hidden] -> [batch, hidden]
        
        # hidden = hidden.permute(1, 0, 2)    # [batch, n_layer * bidirectional, hidden_size]
        # hidden = hidden.reshape(hidden.size(0), -1) # [batch, *]
        # hidden = self.bn(hidden)
        # hidden = self.hidden_proj(hidden)
        hidden = torch.tanh(hidden)   # [batch, hidden]
        return hidden


class Context_encoder(nn.Module):

    '''
    input_size is 2 * utterance_hidden_size
    '''

    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(Context_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        # self.drop = nn.Dropout(p=dropout)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, hidden=None):
        # inpt: [turn_len, batch, input_size]
        # hidden
        if not hidden:
            hidden = torch.randn(2, inpt.shape[1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
        
        # inpt = self.drop(inpt)
        output, hidden = self.gru(inpt, hidden)
        # [seq, batch, hidden]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        # hidden: [2, batch, hidden_size]
        # hidden = hidden.squeeze(0)
        hidden = torch.tanh(hidden)
        return output, hidden
    
    
class VariableLayer(nn.Module):
    
    '''
    VHRED
    '''
    
    def __init__(self, context_hidden, encoder_hidden, z_hidden):
        super(VariableLayer, self).__init__()
        self.context_hidden = context_hidden
        self.encoder_hidden = encoder_hidden
        self.z_hidden = z_hidden
        self.prior_h = nn.ModuleList([nn.Linear(context_hidden, context_hidden),
                                      nn.Linear(context_hidden, context_hidden)])
        self.prior_mu = nn.Linear(context_hidden, z_hidden)
        self.prior_var = nn.Linear(context_hidden, z_hidden)

        self.posterior_h = nn.ModuleList([nn.Linear(context_hidden+encoder_hidden, context_hidden), 
                                          nn.Linear(context_hidden, context_hidden)])
        self.posterior_mu = nn.Linear(context_hidden, z_hidden)
        self.posterior_var = nn.Linear(context_hidden, z_hidden)
        
    def prior(self, context_outputs):
        # context_outputs: [batch, context_hidden]
        h_prior = context_outputs
        for linear in self.prior_h:
            h_prior = torch.tanh(linear(h_prior))
        mu_prior = self.prior_mu(h_prior)
        var_prior = self.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior
    
    def posterior(self, context_outputs, encoder_hidden):
        # context_outputs: [batch, context_hidden]
        # encoder_hidden: [batch, encoder_hidden]
        h_posterior = torch.cat([context_outputs, encoder_hidden], 1)
        for linear in self.posterior_h:
            h_posterior = torch.tanh(linear(h_posterior))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior
    
    def kl_div(self, mu_1, var_1, mu_2, var_2):
        one = torch.FloatTensor([1.0])
        if torch.cuda.is_available():
            one = one.cuda()
        kl_div = torch.sum(0.5 * (torch.log(var2) - torch.log(var1)
                            + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)
        return kl_div
        
        
    def forward(self, context_outputs, encoder_hidden=None, train=True):
        # context_outputs: [batch, context_hidden]
        # Return: z_sent [batch, z_hidden]
        # Return: kl_div, scalar for calculating the loss
        mu_prior, var_prior = self.prior(context_outputs)
        eps = torch.randn((num_sentences, self.z_hidden))
        if torch.cuda.is_available():
            eps = eps.cuda()
            
        if train:
            mu_posterior, var_posterior = self.posterior(context_outputs, 
                                                         encoder_hidden)
            z_sent = mu_posterior + torch.sqrt(var_posterior) * eps
            kl_div = self.kl_div(mu_posterior, var_posterior, mu_prior, var_prior)
            kl_div = torch.sum(kl_div)
        else:
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            kl_div = None
        return z_sent, kl_div
    
    
    
class Decoder(nn.Module):

    '''
    Max likelyhood for decoding the utterance
    input_size is the size of the input vocabulary

    Attention module should satisfy that the decoder_hidden size is the same as 
    the Context encoder hidden size
    '''

    def __init__(self, output_size, embed_size, hidden_size, n_layer=2, dropout=0.5, pretrained=None):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size,
                          num_layers=n_layer, 
                          dropout=(0 if n_layer == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

        # attention on context encoder
        self.attn = Attention(hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch_size], last_hidden: [2, batch, hidden_size]
        # encoder_outputs: [turn_len, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch_size, embed_size]
        key = last_hidden.sum(axis=0)    # [batch, hidden_size]

        # [batch, 1, seq_len]
        attn_weights = self.attn(key, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)    # [1, batch, hidden]

        rnn_input = torch.cat([embedded, context], 2)   # [1, batch, embed+hidden]

        # output: [1, batch, 2*hidden_size], hidden: [2, batch, hidden_size]
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)    # [batch, hidden_size]
        # context = context.squeeze(0)  # [batch, hidden]
        # output = torch.cat([output, context], 1)    # [batch, 2 * hidden]
        output = self.out(output)     # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden
    
    
    
class VHRED(nn.Module):
    
    def __init__(self, embed_size, input_size, output_size, 
                 utter_hidden, context_hidden, decoder_hidden, 
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5,
                 utter_n_layer=1, z_hidden=100,
                 pretrained=None):
        super(VHRED, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder(input_size, 
                                               embed_size, 
                                               utter_hidden, 
                                               dropout=dropout,
                                               n_layer=utter_n_layer,
                                               pretrained=pretrained)
        self.context_encoder = Context_encoder(utter_hidden, context_hidden, 
                                               dropout=dropout) 
        self.decoder = Decoder(output_size, embed_size, decoder_hidden,
                               dropout=dropout, n_layer=utter_n_layer,
                               pretrained=pretrained)
        self.variablelayer = VariableLayer(context_hidden, utter_hidden, z_hidden)
        self.context2decoder = nn.Linear(context_hidden+z_hidden, context_hidden)
        
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
            # [4, batch, hidden]
            hidden = self.utter_encoder(src[i], lengths[i])    # utter_hidden
            turns.append(hidden)
        turns = torch.stack(turns)    # [turn_len, batch, utter_hidden]
        
        # encode the tgt for KL inference in VHRED
        tgt_lengths = []
        for i in range(batch_size):
            seq = tgt[:, i]
            counter = 0
            for j in seq:
                if j.item() == self.pad:
                    break
                counter += 1
            tgt_lengths.append(counter)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tgt_lengths = tgt_lengths.cuda()
        # [batch, utter_hidden]
        ipdb.set_trace()
        tgt_encoder_hidden = self.utter_encoder(tgt, tgt_lengths)

        # context encoding
        # output: [seq, batch, hidden], [2, batch, hidden]
        context_output, hidden = self.context_encoder(turns)
        
        # hidden + variable z 
        # z_sent: [batch, z_hidden]
        z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), 
                                            encoder_hidden=tgt_encoder_hidden, 
                                            train=True)
        z_sent = z_sent.repeat(2, 1, 1)    # [2, batch, z_hidden]
        hidden = torch.cat([hidden, z_sent], dim=2)    # [2, batch, z_hidden+hidden]
        hidden = torch.tanh(self.context2decoder(hidden))
        
        # decoding
        # tgt = tgt.transpose(0, 1)        # [seq_len, batch]
        # hidden = hidden.unsqueeze(0)     # [1, batch, hidden_size]
        output = tgt[0, :]          # [batch]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                # output = torch.max(output, 1)[1]
                output = output.topk(1)[1].squeeze().detach()
        return outputs, kl_div    # [maxlen, batch, vocab_size]

    def predict(self, src, maxlen, lengths, loss=False):
        # predict for test dataset, return outputs: [maxlen, batch_size]
        # src: [turn, max_len, batch_size], lengths: [turn, batch_size]
        with torch.no_grad():
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

            context_output, hidden = self.context_encoder(turns)
            # hidden = hidden.unsqueeze(0)
            
            # hidden + variable z 
            # z_sent: [batch, z_hidden]
            z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), 
                                                encoder_hidden=None, 
                                                train=False)
            z_sent = z_sent.repeat(2, 1, 1)    # [2, batch, z_hidden]
            hidden = torch.cat([hidden, z_sent], dim=2)
            hidden = torch.tanh(self.context2decoder(hidden))

            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output

            if loss:
                return outputs, floss
            else:
                return outputs




if __name__ == "__main__":
    pass