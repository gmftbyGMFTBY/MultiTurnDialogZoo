import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import transformers
import numpy as np
import random
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer
import math
from .layers import *
import ipdb


class transformer(nn.Module):
    
    '''
    GPT2 for seq2seq modeling
    '''
    
    def __init__(self, ):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = len(tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(self.vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')