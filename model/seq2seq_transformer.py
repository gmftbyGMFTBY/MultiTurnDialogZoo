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
import types
import transformers

from .layers import * 


class Decoder(nn.Module):
    
    '''
    Add the multi-head attention for GRU
    '''
    
    def __init__(self, embed_size, hidden_size, output_size, 
                 n_layers=2, dropout=0.5, nhead=8):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size
        
        self.embed = nn.Embedding(output_size, embed_size)
        self.multi_head_attention = nn.ModuleList([Attention(hidden_size) for _ in range(nhead)])
        self.attention = Attention(hidden_size) 
        self.rnn = nn.GRU(hidden_size + embed_size, 
                          hidden_size,
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)
        self.ffn = nn.Linear(nhead*hidden_size, hidden_size)
        
        self.init_weight()
        
    def init_weight(self):
        # orthogonal inittor
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch]
        # last_hidden: [2, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch, embed_size]
        
        # attn_weights: [batch, 1, timestep of encoder_outputs]
        key = last_hidden.sum(axis=0)
        # calculate the attention
        context_collector = []
        for attention_head in self.multi_head_attention:
            attn_weights = attention_head(key, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            context = context.squeeze(1).transpose(0, 1)    # [hidden, batch]
            context_collector.append(context)    # [N, hidden, batch]
        context = torch.stack(context_collector).view(-1, context.shape[-1]).transpose(0, 1)    # [N, hidden, batch]
        # context = context.view(-1, context.shape[-1]).transpose(0, 1)    # [batch, N*hidden]
        context = torch.tanh(self.ffn(context)).unsqueeze(0)    # [1, batch, hidden]    
            
        # context: [batch, 1, hidden_size]
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context = context.transpose(0, 1)
        
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)
        # context = context.squeeze(0)
        # [batch, hidden * 2]
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)    # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        
        # output: [batch, output_size]
        # hidden: [2, batch, hidden_size]
        # hidden = hidden.squeeze(0)
        return output, hidden


class Transformer(nn.Module):
    
    '''
    Transformer encoder and GRU decoder
    
    Multi-head attention for GRU
    '''
    
    def __init__(self, input_vocab_size, opt_vocab_size, d_model, nhead,
                 num_encoder_layers, dim_feedforward, position_embed_size=300,
                 utter_n_layer=2, dropout=0.3, sos=0, pad=0, teach_force=1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.hidden_size = d_model
        self.embed_src = nn.Embedding(input_vocab_size, d_model)
        # position maxlen is 5000
        self.pos_enc = PositionEmbedding(d_model, dropout=dropout,
                                         max_len=position_embed_size)
        self.input_vocab_size = input_vocab_size
        self.utter_n_layer = utter_n_layer
        self.opt_vocab_size = opt_vocab_size
        self.pad, self.sos = pad, sos
        self.teach_force = teach_force
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,  activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_encoder_layers)
        
        self.decoder = Decoder(d_model, d_model, opt_vocab_size, 
                               n_layers=utter_n_layer, dropout=dropout, nhead=nhead)
        
    def generate_key_mask(self, x, lengths):
        # x: [seq, batch]
        # return: key mask [batch, seq]
        seq_length = x.shape[0]
        masks = []
        for sentence_l in lengths:
            masks.append([False for _ in range(sentence_l)] + [True for _ in range(seq_length - sentence_l)])
        masks = torch.tensor(masks)
        if torch.cuda.is_available():
            masks = masks.cuda()
        return masks
        
    def forward(self, src, tgt, lengths):
        # src: [seq, batch], tgt: [seq, batch], lengths: [batch]
        batch_size, max_len = src.shape[1], tgt.shape[0]
        src_key_padding_mask = self.generate_key_mask(src, lengths)
        
        outputs = torch.zeros(max_len, batch_size, self.opt_vocab_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
            
        # src: [seq, batch, d_model]
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        # memory: [seq, batch, d_model]
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # hidden: [2, batch, d_model]
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        output = tgt[0, :]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        
        # [max_len, batch, output_size]
        return outputs
        
        
    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            src_key_padding_mask = self.generate_key_mask(src, lengths)
            
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.opt_vocab_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()

            # src: [seq, batch, d_model]
            src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
            # memory: [seq, batch, d_model]
            memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
            # hidden: [2, batch, d_model]
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
                
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, memory)
                floss[t] = output
                # output = torch.max(output, 1)[1]    # [1]
                output = output.topk(1)[1].squeeze()
                outputs[t] = output    # output: [1, output_size]

            if loss:
                return outputs, floss
            else:
                return outputs 

        
        

'''
class Transformer(nn.Module):
    
    # Refer to: 
    #  - https://github.com/andrewpeng02/transformer-translation
    
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
'''

'''
def bert_for_masked_lm_forward(self, input_ids, encoder_hidden, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            masked_lm_labels=None):
    outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        encoder_hidden=encoder_hidden,  # NOTE: add this line
                        token_type_ids=token_type_ids,
                        position_ids=position_ids, 
                        head_mask=head_mask)

    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)

    outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
    if masked_lm_labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        outputs = (masked_lm_loss,) + outputs

    return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


def bert_model_forward(self, input_ids, encoder_hidden, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * self.config.num_hidden_layers

    embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(embedding_output,
                                    encoder_hidden,  # NOTE: add this line
                                    extended_attention_mask,
                                    head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def bert_encoder_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE: add `encoder_hidden` to next line
        layer_outputs = layer_module(hidden_states, encoder_hidden, attention_mask, head_mask[i])
        hidden_states = layer_outputs[0]

        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def bert_layer_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
    attention_output = attention_outputs[0]
    # NOTE: add the 2 line blow
    attention_outputs = self.cross_atten(attention_output, encoder_hidden, attention_mask, head_mask)
    attention_output = attention_outputs[0]
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
    return outputs


def cross_atten_block_forward(self, input_tensor, encoder_hidden, attention_mask=None, head_mask=None):
    self_outputs = self.self(input_tensor, encoder_hidden, attention_mask, head_mask)  # NOTE: add `encoder_hidden`
    attention_output = self.output(self_outputs[0], input_tensor)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


def cross_bert_self_atten_forward(self, hidden_states, encoder_hidden, attention_mask=None, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(encoder_hidden)  # NOTE: change `hidden_states` to `encoder_hidden`
    mixed_value_layer = self.value(encoder_hidden)  # NOTE: change `hidden_states` to `encoder_hidden`

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # NOTE: Commented to remove attention mask (target to source)
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs


def mask_bert_self_atten_forward(self, hidden_states, attention_mask=None, head_mask=None):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # NOTE: Triangle mask (target to target)
    target_len = attention_scores.size(-1)
    mask = torch.tril(torch.ones(target_len, target_len))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    mask = mask.to(next(self.parameters()).device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    attention_scores = attention_scores + mask
    # NOTE: Commented
    # if attention_mask is not None:
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    return outputs



class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.masked_lm = transformers.BertForMaskedLM(config)
        for layer in self.masked_lm.bert.encoder.layer:
            layer.cross_atten = transformers.modeling_bert.BertAttention(self.masked_lm.config)
            layer.cross_atten.load_state_dict(layer.attention.state_dict()) 

    def forward(self, input_ids, encoder_hidden):
        self.bind_methods()
        return self.masked_lm(input_ids, encoder_hidden)[0]

    def bind_methods(self):
        r"""Change forward method to add `encoder_hidden`.

        Architecture:
            (masked_lm): BertForMaskedLM              [change forward]
            (bert): BertModel                         [change forward]
                (embeddings): BertEmbeddings
                (encoder): BertEncoder                [change forward] 
                (layer): ModuleList
                    (0): BertLayer                    [change forward]
                    (attention): BertAttention        [change forward] [triangle mask]
                        (self): BertSelfAttention
                        (output): BertSelfOutput
                    (cross_atten): BertAttention      [change forward] [add model]
                        (self): BertSelfAttention     [change forward]
                        (output): BertSelfOutput
                    (intermediate): BertIntermediate
                    (output): BertOutput
                (pooler): BertPooler
            (cls): BertOnlyMLMHead
                (predictions): BertLMPredictionHead
                (transform): BertPredictionHeadTransform
                (decoder): Linear
        """
        self.masked_lm.forward = types.MethodType(bert_for_masked_lm_forward, self.masked_lm)
        self.masked_lm.bert.forward = types.MethodType(bert_model_forward, self.masked_lm.bert)
        self.masked_lm.bert.encoder.forward = types.MethodType(bert_encoder_forward, self.masked_lm.bert.encoder)
        for layer in self.masked_lm.bert.encoder.layer:
            layer.forward = types.MethodType(bert_layer_forward, layer)
            layer.cross_atten.forward = types.MethodType(cross_atten_block_forward, layer.cross_atten)
            layer.cross_atten.self.forward = types.MethodType(cross_bert_self_atten_forward, layer.cross_atten.self)
            layer.attention.self.forward = types.MethodType(mask_bert_self_atten_forward, layer.attention.self)



class Transformer(nn.Module):
    
    def __init__(self):
        super(Transformer, self).__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained('config/vocab_en.txt')
        print(f'[!] transformer model, vocab size: {len(self.tokenizer)}')
        self.vocab_size = len(self.tokenizer)
        config = transformers.BertConfig()
        config.num_hidden_layers = 6
        self.encoder = transformers.BertModel(config)
        self.decoder = Decoder()
        self.teach_force = 1
        
        config = self.encoder.config
        self.decoder.masked_lm.cls.predictions.decoder = \
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.masked_lm.cls.predictions.decoder.weight.data.copy_(
            self.encoder.embeddings.word_embeddings.weight)
        self.decoder.masked_lm.cls.predictions.bias = \
            torch.nn.Parameter(torch.zeros(config.vocab_size))
        
    def get_token_type_ids(self, x):
        token_type_ids = torch.zeros_like(x)
        for i in range(x.size(0)):
            sep_index = (x[i] == self.tokenizer.sep_token_id).nonzero()
            sep_index = sep_index.squeeze(-1).tolist()
            sep_index.append(len(x[0]))
            sep_index.append(len(x[i]) - 1)
            for j in range(0, len(sep_index) // 2 * 2, 2):
                start, end = sep_index[j], sep_index[j + 1]
                token_type_ids[i, start+1:end+1] = 1
        return token_type_ids
        
    def forward(self, x, y):
        # x, y: [batch, seq_len]
        token_type_ids = self.get_token_type_ids(x)
        encoder_hidden = self.encoder(x, token_type_ids=token_type_ids)[0]
        # logits: [batch, seq, vocab]
        logits = self.decoder(y[:, :-1], encoder_hidden)
        return logits
    
    def predict(self, x, max_len):
        # x: [batch, seq]
        with torch.no_grad():
            token_type_ids = self.get_token_type_ids(x)
            encoder_hidden = self.encoder(x)[0]
            # token_ids: [batch, maxlen]
            token_ids = torch.empty(x.size(0), max_len, dtype=torch.int64)
            token_ids[:, 0].fill_(self.tokenizer.cls_token_id)  # begin
            token_ids[:, 1:].fill_(self.tokenizer.pad_token_id)  # <pad>
            token_ids = token_ids.to(next(self.parameters()).device)
            for i in range(max_len - 1):
                with torch.no_grad():
                    logits = self.decoder(token_ids, encoder_hidden)
                new_token_ids = logits[:, i].argmax(dim=-1)
                token_ids[:, i + 1] = new_token_ids
        return token_ids
'''
            