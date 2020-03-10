#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.16

'''
Translate the test dataset with the best trained model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import random
import ipdb
import math
import os

from utils import *
from data_loader import *
from metric.metric import *
from model.seq2seq_attention import Seq2Seq
from model.HRED import HRED
from model.HRAN import HRAN


def translate(**kwargs):
    # load the vocab
    tgt_vocab = load_pickle(kwargs['tgt_vocab'])
    src_vocab = load_pickle(kwargs['src_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    
    # load dataset
    if kwargs['hierarchical'] == 1:
        test_iter = get_batch_data(kwargs['src_test'], kwargs['tgt_test'],
                                   kwargs['src_vocab'], kwargs['tgt_vocab'],
                                   kwargs['batch_size'], kwargs['maxlen'],
                                   kwargs["tgt_maxlen"], ld=True)
    else:
        test_iter = get_batch_data_flatten(kwargs['src_test'], kwargs['tgt_test'],
                                           kwargs['src_vocab'], 
                                           kwargs['tgt_vocab'],
                                           kwargs['batch_size'], kwargs['maxlen'], 
                                           kwargs['tgt_maxlen'])

    # pretrained mode
    pretrained = None
    
    # load net
    if kwargs['model'] == 'HRED':
        net = HRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'HRAN':
        net = HRAN(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'Seq2Seq':
        net = Seq2Seq(len(src_w2idx), kwargs['embed_size'], len(tgt_w2idx), 
                      kwargs['utter_hidden' ], 
                      kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                      pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'],
                      dropout=kwargs['dropout'], 
                      utter_n_layer=kwargs['utter_n_layer'], 
                      pretrained=pretrained)
    else:
        raise Exception(f'[!] wrong model named {kwargs["model"]}')
        
    if torch.cuda.is_available():
        net.cuda()
        net.eval()

    print('Net:')
    print(net)
    print(f'[!] Parameters size: {sum(x.numel() for x in net.parameters())}')


    # load best model
    load_best_model(kwargs['dataset'], kwargs['model'], 
                    net, min_threshold=kwargs['min_threshold'],
                    max_threshold=kwargs["max_threshold"])
                        
    # calculate the loss
    criterion = nn.NLLLoss(ignore_index=tgt_w2idx['<pad>'])
    
    # translate
    with open(kwargs['pred'], 'w') as f:
        pbar = tqdm(test_iter)
        for batch in pbar:
            if kwargs['graph'] == 1:
                sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths = batch
            else:
                sbatch, tbatch, turn_lengths = batch

            batch_size = tbatch.shape[1]
            if kwargs['hierarchical']:
                turn_size = len(sbatch)
            
            src_pad, tgt_pad = src_w2idx['<pad>'], tgt_w2idx['<pad>']
            src_eos, tgt_eos = src_w2idx['<eos>'], tgt_w2idx['<eos>']
            
            # output: [maxlen, batch_size], sbatch: [turn, max_len, batch_size]
            output, _, attn = net.predict(sbatch, len(tbatch) + 10, turn_lengths,
                                          loss=True)
            if len(attn) == 2:
                attn, context_attn = attn
            
            for i in range(batch_size):
                ref = list(map(int, tbatch[:, i].tolist()))
                tgt = list(map(int, output[:, i].tolist()))    # [maxlen]
                if kwargs['hierarchical']:
                    src = [sbatch[j][:, i].tolist() for j in range(turn_size)]   # [turns, maxlen]
                else:
                    src = list(map(int, sbatch[:, i].tolist()))

                # filte the <pad>
                ref_endx = ref.index(tgt_pad) if tgt_pad in ref else len(ref)
                ref_endx_ = ref.index(tgt_eos) if tgt_eos in ref else len(ref)
                ref_endx = min(ref_endx, ref_endx_)
                ref = ref[1:ref_endx]
                ref = ' '.join(num2seq(ref, tgt_idx2w))
                ref = ref.replace('<sos>', '').strip()
                ref = ref.replace('< user1 >', '').strip()
                ref = ref.replace('< user0 >', '').strip()

                tgt_endx = tgt.index(tgt_pad) if tgt_pad in tgt else len(tgt)
                tgt_endx_ = tgt.index(tgt_eos) if tgt_eos in tgt else len(tgt)
                tgt_endx = min(tgt_endx, tgt_endx_)
                tgt = tgt[1:tgt_endx]
                tgt = ' '.join(num2seq(tgt, tgt_idx2w))
                tgt = tgt.replace('<sos>', '').strip()
                tgt = tgt.replace('< user1 >', '').strip()
                tgt = tgt.replace('< user0 >', '').strip()

                if kwargs['hierarchical']:
                    source = []
                    for item in src:
                        item_endx = item.index(src_pad) if src_pad in item else len(item)
                        item_endx_ = item.index(src_eos) if src_eos in item else len(item)
                        item_endx = min(item_endx, item_endx_)
                        item = item[1:item_endx]
                        item = num2seq(item, src_idx2w)
                        source.append(' '.join(item))
                    src = ' __eou__ '.join(source)
                else:
                    src_endx = src.index(src_pad) if src_pad in src else len(src)
                    src_endx_ = src.index(src_eos) if src_eos in src else len(src)
                    sec_endx = min(src_endx, src_endx_)
                    src = src[1:src_endx]
                    src = ' '.join(num2seq(src, src_idx2w))
                    
                f.write(f'- src: {src}\n')
                f.write(f'- ref: {ref}\n')
                f.write(f'- tgt: {tgt}\n\n')
                    
                if kwargs['attn_data'][0] in src:
                    if kwargs['model'] == 'HRED':
                        src = ' '.join(list(map(str, range(attn.shape[1]))))
                    path = f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/attn.pkl'
                    print(f'[!] find the sentences, save the attn({attn.shape}) to {path}. ABORT.')
                    with open(path, 'wb') as fk:
                        pickle.dump({'attn': attn, 'src': src, 'tgt': tgt}, fk)
                        
                    try:
                        path = f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/context_attn.pkl'
                        # ipdb.set_trace()
                        with open(path, 'wb') as fk:
                            pickle.dump({'attn': context_attn, 'src': ' '.join(list(map(str, range(context_attn.shape[1])))), 'tgt': tgt}, fk)
                    except:
                        pass
                    exit()
                        
    print(f'[!] write the translate result into {kwargs["pred"]}')
        
        
dataset, model = 'dstc7', 'HRAN'
if model in ['HRAN', 'HRED']:
    hierarchical = 1
else:
    hierarchical = 0
    
args = {'src_test': f'./data/{dataset}/src-test.txt',
        'tgt_test': f'./data/{dataset}/tgt-test.txt',
        'min_threshold': 0,
        'max_threshold': 100,
        'batch_size': 1,
        'model': model,
        'utter_n_layer': 2,
        'utter_hidden': 512,
        'context_hidden': 512,
        'decoder_hidden': 512,
        'seed': 30,
        'dropout': 0.3,
        'embed_size': 256,
        'd_model': 512,
        'nhead': 4,
        'num_encoder_layers': 8,
        'num_decoder_layers': 8,
        'dim_feedforward': 2048,
        'dataset': dataset,
        'src_vocab': f'./processed/{dataset}/iptvocab.pkl.bak',
        'tgt_vocab': f'./processed/{dataset}/optvocab.pkl.bak',
        'maxlen': 50,
        'pred': f'./processed/{dataset}/{model}/pure-pred.txt.bak',
        'hierarchical': hierarchical,
        'tgt_maxlen': 30,
        'graph': 0,
        'test_graph': f'./processed/{dataset}/test-graph.pkl',
        'position_embed_size': 30,
        'contextrnn': True,
        'plus': 0,
        'context_threshold': 2,
        'ppl': 'origin',
        'gat_heads': 8,
        'teach_force': 1}

# Seq2Seq vs. HRAN
args['attn_data'] = ['a person is standing a door with a plastic bag in her hand']

# Seq2Seq vs. HRED
# args['attn_data'] = ['a man is in the kitchen doing dishes with music heard in the background']
    
# set random seed
random.seed(args['seed'])
torch.manual_seed(args['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args['seed'])
         
with torch.no_grad():
    translate(**args)