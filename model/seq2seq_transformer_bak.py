import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import transformers
import numpy as np
import random
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import math
from .layers import *
import ipdb


class transformer_gpt2(nn.Module):
    
    '''
    GPT2 for seq2seq modeling
    '''
    
    def __init__(self, config_path):
        super(transformer_gpt2, self).__init__()
        self.tokenzier = BertTokenizer(vocab_file='config/vocab_en.txt')
        self.vocab_size = len(self.tokenzier)
        
        self.model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(self.vocab_size)
        
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        
    def forward(self, inpt):
        '''
        inpt: [seq, batch]
        '''
        inpt = inpt.transpose(0, 1)    # [batch, seq]
        ipdb.set_trace()
        opt = self.model.forward(input_ids=inpt)[0]    # [batch, seq, vocab]
        opt = F.log_softmax(opt, dim=-1)    # [batch, seq, vocab]
        return opt
    
    def predict(self, inpt, maxlen, loss=True):
        '''
        Different from the forward function, auto-regression
        inpt: [seq, batch]
        '''
        with torch.no_grad():
            inpt = inpt.transpose(0, 1)    # [batch, seq]
            batch_size = inpt.shape[0]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.vocab_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()
            
            for t in range(maxlen):
                opt = self.model.forward(input_ids=inpt)[0]    # [batch, seq, vocab]
                opt = opt[:, -1, :]    # [batch, vocab]
                next_token = F.log_softmax(opt, dim=-1)    # [batch, vocab]
                
                floss[t] = next_token
                
                next_token = next_token.topk(1)[1]    # [batch, 1]
                outputs[t] = next_token.squeeze()
                
                # inpt: [batch, seq], next_token: [batch, 1]
                # inpt: [batch, seq+1]
                inpt = torch.cat((inpt, next_token), dim=-1)
            if loss:
                return outputs, floss
            else:
                return outpus
    
    
if __name__ == "__main__":
    pass