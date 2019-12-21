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

from utils import *
from data_loader import *
from model.HRED import HRED
from model.seq2seq_attention import Seq2Seq
from model.WSeq import WSeq
from model.ReCoSa import ReCoSa
from model.seq2seq_transformer import transformer
from model.MReCoSa import MReCoSa
from model.MTGCN import MTGCN
from model.GCNRNN import GCNRNN
from model.GatedGCN import GatedGCN


def translate(**kwargs):
    # load the vocab
    tgt_vocab = load_pickle(kwargs['tgt_vocab'])
    src_vocab = load_pickle(kwargs['src_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    
    # load dataset
    if kwargs['hierarchical'] == 1:
        if kwargs['graph'] == 1:
            test_iter = get_batch_data_graph(kwargs['src_test'], kwargs['tgt_test'],
                                             kwargs['test_graph'], kwargs['src_vocab'],
                                             kwargs['tgt_vocab'], kwargs['batch_size'],
                                             kwargs['maxlen'], plus=kwargs["plus"])
        else:
            test_iter = get_batch_data(kwargs['src_test'], kwargs['tgt_test'],
                                       kwargs['src_vocab'], kwargs['tgt_vocab'],
                                       kwargs['batch_size'], kwargs['maxlen'],
                                       plus=kwargs["plus"])
    else:
        test_iter = get_batch_data_flatten(kwargs['src_test'], kwargs['tgt_test'],
                                           kwargs['src_vocab'], kwargs['tgt_vocab'],
                                           kwargs['batch_size'], kwargs['maxlen'])

    # pretrained mode
    if kwargs['pretrained'] == 'bert':
        pretrained = f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/'
    else:
        pretrained = None
    
    # load net
    if kwargs['model'] == 'HRED':
        net = HRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'], 
                   sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'], 
                   pretrained=pretrained)
    elif kwargs['model'] == 'WSeq':
        net = WSeq(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'], 
                   sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'ReCoSa':
        net = ReCoSa(len(src_w2idx), kwargs['d_model'], kwargs['d_model'], len(tgt_w2idx),
                     n_layers=kwargs['utter_n_layer'], sos=tgt_w2idx['<sos>'],
                     pad=tgt_w2idx['<pad>'], pretrained=pretrained)
    elif kwargs['model'] == 'MReCoSa':
        net = MReCoSa(len(src_w2idx), 512, len(tgt_w2idx), 
                      512, 512, pad=tgt_w2idx["<pad>"], sos=tgt_w2idx['<sos>'], 
                      utter_n_layer=kwargs['utter_n_layer'], pretrained=pretrained)
    elif kwargs['model'] == 'Transformer':
        net = transformer(len(src_w2idx), len(tgt_w2idx), embed_size=kwargs['embed_size'],
                          src_pad=src_w2idx['<pad>'], tgt_pad=tgt_w2idx['<pad>'], 
                          tgt_sos=tgt_w2idx['<sos>'], pretrained=pretrained)
    elif kwargs['model'] == 'Seq2Seq':
        net = Seq2Seq(len(src_w2idx), kwargs['embed_size'], 
                      len(tgt_w2idx), kwargs['utter_hidden'], 
                      kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'], 
                      sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'],
                      pretrained=pretrained)
    elif kwargs['model'] == 'MTGCN':
        net = MTGCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], kwargs['position_embed_size'],
                    pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                    utter_n_layer=kwargs['utter_n_layer'], 
                    context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'GCNRNN':
        net = GCNRNN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                     kwargs['utter_hidden'], kwargs['context_hidden'],
                     kwargs['decoder_hidden'], kwargs['position_embed_size'],
                     pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                     utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'GatedGCN':
        net = GatedGCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], kwargs['position_embed_size'],
                    pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                    utter_n_layer=kwargs['utter_n_layer'], 
                    context_threshold=kwargs['context_threshold'])
    else:
        raise Exception(f'[!] wrong model named {kwargs["model"]}')

    if torch.cuda.is_available():
        net.cuda()
        net.eval()

    print('Net:')
    print(net)

    # load best model
    load_best_model(kwargs['dataset'], kwargs['model'], 
                    net, min_threshold=kwargs['min_threshold'],
                    max_threshold=kwargs["max_threshold"])
                        
    # calculate the loss
    criterion = nn.NLLLoss(ignore_index=tgt_w2idx['<pad>'])
    total_loss, batch_num = 0.0, 0
    
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
            if kwargs['graph'] == 1:
                output, f_l = net.predict(sbatch, gbatch, subatch, tubatch, len(tbatch), turn_lengths,
                                          loss=True)
            else:
                output, f_l = net.predict(sbatch, len(tbatch), turn_lengths,
                                          loss=True)
                        
            # ipdb.set_trace()
            loss = criterion(f_l[1:].view(-1, len(tgt_w2idx)),
                             tbatch[1:].contiguous().view(-1))
            batch_num += 1
            total_loss += loss.item()
            
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
                        
    l = round(total_loss / batch_num, 4)
    print(f'[!] write the translate result into {kwargs["pred"]}')
    
    if kwargs['ppl'] == 'origin':
        print(f'[!] loss: {l}, N-gram PPL: {round(math.exp(l), 4)}', file=open(f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/ppl.txt', 'a'))
    elif kwargs['ppl'] == 'ngram':
        # n-gram ppl
        lm = load_pickle(f'./data/{kwargs["dataset"]}/lm.pkl')
        ref_data, tgt_data = read_pred_file(kwargs["pred"])
        print(f'[!] ========== begin to calcualte the n-gram ppl ==========')
        ref_ppl = ngram_ppl(lm, ref_data)
        ppl = ngram_ppl(lm, tgt_data)
        print(f'[!] Ref N-gram ppl: {round(ref_ppl, 4)}, N-gram PPL: {round(ppl, 4)}', file=open(f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/ppl.txt', 'a'))
    else:
        raise Exception(f'[!] make sure the mode for ppl calculating is origin or ngram, but {kwargs["ppl"]} is given.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate script')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--min_threshold', type=int, default=0, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=20, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--utter_n_layer', type=int, default=1, help='layer of encoder')
    parser.add_argument('--utter_hidden', type=int, default=150, 
                        help='utterance encoder hidden size')
    parser.add_argument('--context_hidden', type=int, default=150, 
                        help='context encoder hidden size')
    parser.add_argument('--decoder_hidden', type=int, default=150, 
                        help='decoder hidden size')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200, 
                        help='embedding layer size')
    parser.add_argument('--dataset', type=str, default='dailydialog', 
                        help='dataset for training')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50, help='the maxlen of the utterance')
    parser.add_argument('--pred', type=str, default=None, 
                        help='the csv file save the output')
    parser.add_argument('--hierarchical', type=int, default=1, help='whether hierarchical architecture')
    parser.add_argument('--d_model', type=int, default=512, help='d_model for transformer')
    parser.add_argument('--tgt_maxlen', type=int, default=50, help='target sequence maxlen')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained mode')
    parser.add_argument('--contextrnn', dest='contextrnn', action='store_true')
    parser.add_argument('--no-contextrnn', dest='contextrnn', action='store_false')
    parser.add_argument('--position_embed_size', type=int, default=30)
    parser.add_argument('--test_graph', type=str, default=None)
    parser.add_argument('--graph', type=int, default=0)
    parser.add_argument('--plus', type=int, default=2)
    parser.add_argument('--context_threshold', type=int, default=3, help='low turns filter')
    parser.add_argument('--ppl', type=str, default='origin', help='origin: e^{loss} ppl; ngram: supported by NLTK. More details can be found in README.')

    args = parser.parse_args()
    
    # show the parameters
    print('Parameters:')
    print(args)
    
    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # translate
    args_dict = vars(args)
    translate(**args_dict)
    
