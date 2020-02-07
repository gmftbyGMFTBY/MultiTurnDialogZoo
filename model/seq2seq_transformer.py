#!/usr/bin/python3
# Author: GMFTBY
# Time: 2020.2.7

'''
Seq2Seq in Transformer, implemented by Pytorch's nn.Transformer
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


class Transformer(nn.Module):
    
    '''
    Refer to: 
     - https://github.com/andrewpeng02/transformer-translation
    '''
    
    def __init__(self, inpt_vocab_size, opt_vocab_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout, sos=0, pad=0):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(inpt_vocab_size, d_model)
        self.embed_tgt = nn.Embedding(opt_vocab_size, d_model)
        self.pos_enc = PositionEmbedding(d_model, dropout=dropout)
        self.inpt_vocab_size = inpt_vocab_size
        self.opt_vocab_size = opt_vocab_size
        self.pad, self.sos = pad, sos
        
        self.model = nn.Transformer(d_model, nhead, 
                                    num_encoder_layers, 
                                    num_decoder_layers, 
                                    dim_feedforward, 
                                    dropout)
        self.fc = nn.Linear(d_model, opt_vocab_size)
        self.init_weight()
        
    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)
        
    def forward(self, src, tgt, 
                src_key_padding_mask,
                tgt_key_padding_mask, 
                memory_key_padding_mask):
        # src, tgt: [seq, batch]
        tgt_mask = gen_nopeek_mask(tgt.shape[0])
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        
        # encoder and decoder in one line
        # input:
        # src: [seq, batch]
        # tgt: [seq, batch]
        # src_key_padding_mask: [batch, seq]
        # tgt_key_padding_mask: [batch, seq]
        # memory_key_padding_mask: [batch, seq]
        # output: [seq, batch, vocab]
        output = self.model(src, 
                            tgt, 
                            tgt_mask=tgt_mask, 
                            src_key_padding_mask=src_key_padding_mask, 
                            tgt_key_padding_mask=tgt_key_padding_mask, 
                            memory_key_padding_mask=memory_key_padding_mask)
        # [seq, batch, vocab_size]
        return F.log_softmax(self.fc(output), dim=-1)
    
    def predict(self, src, 
                src_key_padding_mask, 
                memory_key_padding_mask, 
                maxlen):
        # src: [seq, batch]
        with torch.no_grad():
            batch_size = src.shape[1]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.opt_vocab_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()
                
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()
            output = [output]
                
            src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
            for t in range(1, maxlen):
                # tgt: [seq, batch, vocab_size]
                # this part is slow druing inference
                tgt_mask = gen_nopeek_mask(t)
                soutput = torch.stack(output)
                soutput = self.pos_enc(self.embed_tgt(soutput) * math.sqrt(self.d_model))
                tgt = self.model(src, 
                                 soutput, 
                                 src_key_padding_mask=src_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 tgt_key_padding_mask=None,
                                 tgt_mask=tgt_mask)
                tgt = F.log_softmax(self.fc(tgt[-1]), dim=-1)    # [batch, vocab_size]
                floss[t] = tgt
                tgt = tgt.topk(1)[1].squeeze()    # [batch]
                outputs[t] = tgt
                output.append(tgt)
        return outputs, floss
                
            