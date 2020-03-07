#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.29


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import Data, DataLoader    # create the graph batch dynamically
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
import random
import math
from .layers import *
import ipdb


class Utterance_encoder_ggcn(nn.Module):
    
    '''
    Bidirectional GRU
    '''

    def __init__(self, input_size, embedding_size, 
                 hidden_size, dropout=0.5, n_layer=1, pretrained=False):
        super(Utterance_encoder_ggcn, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer

        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, 
                          dropout=dropout, bidirectional=True)
        # self.hidden_proj = nn.Linear(n_layer * 2 * self.hidden_size, hidden_size)
        # self.bn = nn.BatchNorm1d(num_features=hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        embedded = self.embed(inpt)
        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)
        hidden = hidden.sum(axis=0)
        hidden = torch.tanh(hidden)
        # hidden = hidden.permute(1, 0, 2)
        # hidden = hidden.reshape(hidden.size(0), -1)
        # hidden = self.bn(self.hidden_proj(hidden))
        # hidden = torch.tanh(hidden)

        return hidden    # [batch, hidden]

        
class GatedGCNContext(nn.Module):

    '''
    GCN Context encoder

    It should be noticed that PyG merges all the subgraph in the batch into a big graph
    which is a sparse block diagonal adjacency matrices.
    Refer: Mini-batches in 
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    Our implementation is the three layers GCN with the position embedding
    
    ========== Make sure the inpt_size == output_size ==========
    '''

    def __init__(self, inpt_size, output_size, user_embed_size, 
                 posemb_size, dropout=0.5, threshold=2):
        # inpt_size: utter_hidden_size + user_embed_size
        super(GatedGCNContext, self).__init__()
        # utter + user_embed + pos_embed
        size = inpt_size + user_embed_size + posemb_size
        self.threshold = threshold
        
        # GatedGCN
        self.kernel_rnn1 = nn.GRUCell(size, size)
        self.kernel_rnn2 = nn.GRUCell(size, size)
        self.conv1 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        self.conv2 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        self.conv3 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        # self.layer_norm1 = nn.LayerNorm(inpt_size)
        # self.layer_norm2 = nn.LayerNorm(inpt_size)
        self.layer_norm = nn.LayerNorm(inpt_size)

        # rnn for background
        self.rnn = nn.GRU(inpt_size + user_embed_size, inpt_size, bidirectional=True)

        self.linear1 = nn.Linear(inpt_size * 2, inpt_size)
        self.linear2 = nn.Linear(inpt_size * 2, output_size)
        self.drop = nn.Dropout(p=dropout)
        
        # 100 is far bigger than the max turn lengths (cornell and dailydialog datasets)
        self.posemb = nn.Embedding(100, posemb_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.kernel_rnn1.weight_hh)
        init.xavier_normal_(self.kernel_rnn1.weight_ih)
        self.kernel_rnn1.bias_ih.data.fill_(0.0)
        self.kernel_rnn1.bias_hh.data.fill_(0.0)
        init.xavier_normal_(self.kernel_rnn2.weight_hh)
        init.xavier_normal_(self.kernel_rnn2.weight_ih)
        self.kernel_rnn2.bias_ih.data.fill_(0.0)
        self.kernel_rnn2.bias_hh.data.fill_(0.0)
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def create_batch(self, gbatch, utter_hidden):
        '''create one graph batch
        :param: gbatch [batch_size, ([2, edge_num], [edge_num])]
        :param: utter_hidden [turn_len(node), batch, hidden_size]'''
        utter_hidden = utter_hidden.permute(1, 0, 2)    # [batch, node, hidden_size]
        batch_size = len(utter_hidden)
        data_list, weights = [], []
        for idx, example in enumerate(gbatch):
            edge_index, edge_w = example
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_w = torch.tensor(edge_w, dtype=torch.float)
            data_list.append(Data(x=utter_hidden[idx], edge_index=edge_index))
            weights.append(edge_w)
        # this special loader only have one batch
        loader = DataLoader(data_list, batch_size=batch_size)
        batch = list(loader)
        assert len(batch) == 1
        batch = batch[0]    # one big graph (mini-batch in PyG)
        weights = torch.cat(weights)

        return batch, weights

    def forward(self, gbatch, utter_hidden, ub):
        # utter_hidden: [turn_len, batch, inpt_size]
        # ub: [turn_len, batch, user_embed_size]
        # BiRNN First, rnn_x: [turn, batch, 2 * inpt_size]
        rnn_x, rnnh = self.rnn(torch.cat([utter_hidden, ub], dim=-1))
        rnn_x = torch.tanh(self.linear1(rnn_x))    # [turn, batch, inpt_size]
        turn_size = utter_hidden.size(0)
        rnnh = torch.tanh(rnnh.sum(axis=0))
        
        if turn_size <= self.threshold:
            return rnn_x, rnnh    # [turn, batch, inpt_size]
        
        batch, weights = self.create_batch(gbatch, rnn_x)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        # cat pos_embed: [node, posemb_size]
        batch_size = torch.max(batch).item() + 1

        # pos
        pos = []
        for i in range(batch_size):
            pos.append(torch.arange(turn_size, dtype=torch.long))
        pos = torch.cat(pos)
        
        ub = ub.reshape(-1, ub.size(-1))

        # load to GPU
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            batch = batch.cuda()
            weights = weights.cuda()
            pos = pos.cuda()    # [node]
        
        pos = self.posemb(pos)    # [node, pos_emb]
        
        # [node, pos_emb + inpt_size + user_embed_size]
        x = torch.cat([x, pos, ub], dim=1)
        # x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=weights)))
        x1 = torch.tanh(self.conv1(x, edge_index, edge_weight=weights))
        # x1 = self.layer_norm1(x1)
        x1_ = torch.cat([x1, pos, ub], dim=1)
        # x2 = F.relu(self.bn2(self.conv2(x1_, edge_index, edge_weight=weights)))
        x2 = torch.tanh(self.conv2(x1_, edge_index, edge_weight=weights))
        # x2 = self.layer_norm2(x2)
        x2_ = torch.cat([x2, pos, ub], dim=1)
        # x3 = F.relu(self.bn3(self.conv3(x2_, edge_index, edge_weight=weights)))
        x3 = torch.tanh(self.conv3(x2_, edge_index, edge_weight=weights))
        # x3 = self.layer_norm3(x3)

        # residual for overcoming over-smoothing, [nodes, inpt_size]
        # residual -> dropout -> layernorm
        x = x1 + x2 + x3
        x = self.layer_norm(self.drop(torch.tanh(x)))

        # [nodes/turn_len, output_size]
        # take apart to get the mini-batch
        x = torch.stack(x.chunk(batch_size, dim=0)).permute(1, 0, 2)    # [turn, batch, inpt_size]
        x = torch.cat([rnn_x, x], dim=2)    # [turn, batch, inpt_size * 2]
        x = torch.tanh(self.linear2(x))    # [turn, batch, output_size]
        return x, rnnh

    
class Decoder_ggcn(nn.Module):
    
    def __init__(self, output_size, embed_size, hidden_size, user_embed_size=10, n_layer=2, dropout=0.5, pretrained=None):
        super(Decoder_ggcn, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size,
                          num_layers=n_layer, dropout=(0 if n_layer == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attention(hidden_size)
        # self.attn = Graph_attention(hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, last_hidden, gcncontext):
        # inpt: [batch_size], last_hidden: [2, batch, hidden_size]
        # gcncontext: [turn_len, batch, hidden_size], user_de: [batch, 11]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch_size, embed_size]
        key = last_hidden.sum(axis=0)    # [batch, hidden]

        # attention on the gcncontext
        attn_weights = self.attn(key, gcncontext)
        context = attn_weights.bmm(gcncontext.transpose(0, 1))
        context = context.transpose(0, 1)    # [1, batch, hidden]

        rnn_inpt = torch.cat([embedded, context], 2)    # [1, batch, embed_size + hidden]

        output, hidden = self.gru(rnn_inpt, last_hidden)
        output = output.squeeze(0)      # [batch, hidden_size]
        # context = context.squeeze(0)    # [batch, hidden]
        # output = torch.cat([output, context], 1)    # [batch, hidden * 2]
        output = self.out(output)   # [batch, output_size]
        output = F.log_softmax(output, dim=1)

        # [batch, output_size], [1, batch, hidden_size]
        return output, hidden
    
    
class GatedGCN(nn.Module):
    
    '''
    When2Talk model
    1. utterance encoder
    2. GCN context encoder
    3. (optional) RNN Context encoder
    4. Attention RNN decoder
    '''
    
    def __init__(self, input_size, output_size, embed_size, utter_hidden_size, 
                 context_hidden_size, decoder_hidden_size, position_embed_size, 
                 teach_force=0.5, pad=0, sos=0, dropout=0.5, user_embed_size=10,
                 utter_n_layer=1, bn=False, context_threshold=2):
        super(GatedGCN, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder_ggcn(input_size, embed_size,
                                                    utter_hidden_size, 
                                                    dropout=dropout,
                                                    n_layer=utter_n_layer) 
        self.gcncontext = GatedGCNContext(utter_hidden_size,
                                          context_hidden_size,
                                          user_embed_size,
                                          position_embed_size, 
                                          dropout=dropout,
                                          threshold=context_threshold)
        self.decoder = Decoder_ggcn(output_size, embed_size, 
                                    decoder_hidden_size,
                                    n_layer=utter_n_layer,
                                    dropout=dropout) 
        
        # hidden project
        self.hidden_proj = nn.Linear(context_hidden_size + user_embed_size, 
                                     decoder_hidden_size)
        self.hidden_drop = nn.Dropout(p=dropout)
        
        # user embedding, 10 
        self.user_embed = nn.Embedding(2, user_embed_size)
        
    def forward(self, src, tgt, gbatch, subatch, tubatch, lengths):
        '''
        :param: src, [turns, lengths, bastch]
        :param: tgt, [lengths, batch]
        :param: gbatch, [batch, ([2, num_edges], [num_edges])]
        :param: subatch, [turn, batch]
        :param: tubatch, [batch]
        :param: lengths, [turns, batch]
        '''
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        subatch = self.user_embed(subatch)    # [turn, batch, 10]
        tubatch = self.user_embed(tubatch)    # [batch, 10]
        tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)    # [2, batch, 10]
        
        # utterance encoding
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)    # [turn_len, batch, utter_hidden]

        # GCN Context encoder
        # context_output: [turn, batch, output_size]
        context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
        # context_output = context_output.permute(1, 0, 2)    # [turn, batch, hidden]
        
        ghidden = context_output[-1]    # [batch, decoder_hidden]
        hidden = torch.stack([rnnh, ghidden])    # [2, batch, hidden]
        hidden = torch.cat([hidden, tubatch], 2)    # [2, batch, hidden+user_embed]
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))  # [2, batch, hidden]

        # decoding step
        # hidden = hidden.unsqueeze(0)     # [1, batch, hidden_size]
        output = tgt[0, :]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t].clone().detach()
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = torch.max(output, 1)[1]

        # de: [batch], outputs: [maxlen, batch, output_size]
        return outputs
    
    def predict(self, src, gbatch, subatch, tubatch, maxlen, lengths, loss=False):
        # similar with the forward function
        # src: [turn, maxlen, batch_size], lengths: [turn, batch_size]
        # subatch: [turn_len, batch], tubatch: [batch]
        # output: [maxlen, batch_size]
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()

            subatch = self.user_embed(subatch)    # [turn, batch, 10]
            tubatch = self.user_embed(tubatch)    # [batch, 10]
            tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)    # [2, batch, 10]

            # utterance encoding
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)     # [turn, batch, hidden]

            # GCN Context encoding
            # [batch, turn, hidden]
            context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
            # context_output = context_output.permute(1, 0, 2)    # [turn, batch, hidden]

            ghidden = context_output[-1]    # [batch, decoder_hidden]
            hidden = torch.stack([rnnh, ghidden])    # [2, batch, hidden]
            hidden = torch.cat([hidden, tubatch], 2)    # [batch, hidden+user_embed]
            hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))  # [batch, hidden]

            # hidden = hidden.unsqueeze(0)     # [1, batch, hidden]
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output

            # de: [batch], outputs: [maxlen, batch]
            if loss:
                return outputs, floss
            else:
                return outputs


if __name__ == "__main__":
    pass
