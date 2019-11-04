#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.22

'''
Pure transformer for seq2seq modeling
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .layers import *


class transformer(nn.Module):

    def __init__(self, input_size, output_size, 
                 embed_size=512, nhead=8, n_layers=6,
                 src_pad=0, tgt_pad=0, tgt_sos=0, dropout=0.5, pretrained=None):
        super(transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.src_pad, self.tgt_pad, self.tgt_sos = src_pad, tgt_pad, tgt_sos
        self.dropout = dropout
        self.pos_emb = PositionEmbedding(self.embed_size, dropout=dropout)

        self.model = nn.Transformer(d_model=self.embed_size, nhead=nhead, 
                                    num_encoder_layers=n_layers,
                                    num_decoder_layers=n_layers, dropout=self.dropout)

        if pretrained:
            src_path = f'{pretrained}/ipt_bert_embedding.pkl'
            tgt_path = f'{pretrained}/opt_bert_embedding.pkl'
            self.inpt_emb = PretrainedEmbedding(self.input_size, self.embed_size, 
                                                src_path)
            self.opt_emb = PretrainedEmbedding(self.output_size, self.embed_size, 
                                               tgt_path)
        else:
            self.inpt_emb = nn.Embedding(self.input_size, self.embed_size)
            self.opt_emb = nn.Embedding(self.output_size, self.embed_size)
        
        self.out = nn.Linear(self.embed_size, self.output_size)
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, src, tgt, lengths):
        # src: [seq_len, batch], tgt: [seq_len, batch], lengths: [turn_len, batch]
        batch_size, maxlen = src.shape[-1], tgt.size(0)
        src_ = self.inpt_emb(src)    # [seq_len, batch, embed_size]
        src_ = self.pos_emb(src_)
        tgt_ = self.opt_emb(tgt)     # [seq_len, batch, embed_size]
        tgt_ = self.pos_emb(tgt_)

        # mask
        # tgt_mask: [seq_len, seq_len]
        tgt_mask = self.model.generate_square_subsequent_mask(maxlen)
        # src_key_padding_mask: [batch, seq_len]
        src_key_padding_mask = src.transpose(0, 1) == self.src_pad
        # tgt_key_padding_mask: [batch, seq_len]
        tgt_key_padding_mask = tgt.transpose(0, 1) == self.tgt_pad

        if torch.cuda.is_available():
            tgt_mask = tgt_mask.cuda()
            src_key_padding_mask = src_key_padding_mask.cuda()
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()

        # output: [seq_len, batch, embed]
        output = self.model(src_, tgt_, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.norm(self.out(output))
        output = F.log_softmax(output, -1)

        # output: [seq_len, batch, embed]
        return output

    def predict(self, src, maxlen, lengths):
        # src: [seq_len, batch], lengths: [turn_len, batch]
        batch_size = src.shape[-1]
        src_ = self.inpt_emb(src)
        src_ = self.pos_emb(src_)

        # mask
        # tgt_mask: [seq_len, seq_len]
        tgt_mask = self.model.generate_square_subsequent_mask(maxlen)
        # src_key_padding_mask: [batch, seq_len]
        src_key_padding_mask = src.transpose(0, 1) == self.src_pad
        # tgt_key_padding_mask: [batch, seq_len]
        tgt = torch.zeros(maxlen, batch_size, dtype=torch.long).fill_(self.tgt_sos)

        if torch.cuda.is_available():
            tgt = tgt.cuda()
            tgt_mask = tgt_mask.cuda()
            src_key_padding_mask = src_key_padding_mask.cuda()

        tgt = self.opt_emb(tgt)    # [seq_len, batch, embed]
        tgt = self.pos_emb(tgt)

        # pred
        preds = []
        for j in range(1, maxlen):
            tgt = self.model(src_, tgt, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask)
            tgt = self.norm(self.out(tgt))    # [seq_len, batch, output_size]
            tgt = F.log_softmax(tgt, dim=-1)
            p = tgt.max(2)[1]
            preds.append(p[j])
            tgt = self.opt_emb(p)    # [seq_len, batch, embed]
            tgt = self.pos_emb(tgt)

        tgt = torch.stack(preds)     # [seq_len, batch]
        return tgt


if __name__ == "__main__":
    pass
