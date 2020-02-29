#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.15

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
import torch.optim as optim
import random
import numpy as np
import argparse
import math
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import ipdb
import transformers
import gensim

from utils import *
from data_loader import *
from metric.metric import *
from model.seq2seq_attention import Seq2Seq
from model.seq2seq_multi_head_attention import Seq2Seq_Multi_Head
from model.seq2seq_transformer import Transformer
from model.HRED import HRED
from model.VHRED import VHRED
from model.KgCVAE import KgCVAE
from model.HRAN import HRAN
from model.HRAN_ablation import HRAN_ablation
from model.WSeq import WSeq
from model.WSeq_RA import WSeq_RA
from model.DSHRED import DSHRED
from model.DSHRED_RA import DSHRED_RA
from model.MReCoSa import MReCoSa
from model.MReCoSa_RA import MReCoSa_RA

try:
    from model.MTGCN import MTGCN
    from model.MTGAT import MTGAT
    from model.GatedGCN import GatedGCN
    from model.layers import *
except:
    print(f'[!] cannot find the module "torch_geometric", ignore it')


def train(train_iter, net, optimizer, vocab_size, pad, 
          grad_clip=10, graph=False, debug=False, 
          transformer_decode=False, kl_mult=0, 
          kl_annealing_iter=20000):
    # choose nll_loss for training the objective function
    net.train()
    total_loss, batch_num = 0.0, 0
    criterion = nn.NLLLoss(ignore_index=pad)

    pbar = tqdm(train_iter)
    for idx, batch in enumerate(pbar):
        # [turn, length, batch], [seq_len, batch] / [seq_len, batch], [seq_len, batch]
        if graph:
            sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch
        batch_size = tbatch.shape[1]
        if batch_size == 1:
            # batchnorm will throw error when batch_size is 1
            continue

        optimizer.zero_grad()

        # [seq_len, batch, vocab_size]
        if graph:
            output = net(sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths)
        else:
            output = net(sbatch, tbatch, turn_lengths)
        
        if type(output) == tuple:
            # VHRED model, KL divergence add to the loss
            if len(output) == 2:
                output, kl_div = output
                bow_loss = None
            elif len(output) == 3:
                output, kl_div, bow_loss = output
            else:
                raise Exception('[!] wrong')
        else:
            kl_div, bow_loss = None, None
        loss = criterion(output[1:].view(-1, vocab_size),
                         tbatch[1:].contiguous().view(-1))
        if kl_div:
            loss += kl_mult * kl_div
        if bow_loss:
            loss += bow_loss
            
        # add train loss to the tensorfboard
        # writer.add_scalar(f'{writer_str}-Loss/train-{epoch}', loss, idx)

        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1

        pbar.set_description(f'batch {batch_num}, training loss: {round(loss.item(), 4)}')
        
        # VHRED
        kl_mult = min(kl_mult + 1.0 / kl_annealing_iter, 1.0)

        if debug:
            # show the output result, output: [length, batch, vocab_size]
            ipdb.set_trace()
            utterance = output[:, 0, :].squeeze(1)    # [length, vocab_size]
            word_idx = utterance.data.max(1)[1]       # [length]

    # return avg loss
    return round(total_loss / batch_num, 4), kl_mult


def validation(data_iter, net, vocab_size, pad, 
               graph=False, transformer_decode=False, debug=False):
    net.eval()
    batch_num, total_loss = 0, 0.0
    criterion = nn.NLLLoss(ignore_index=pad)

    pbar = tqdm(data_iter)

    for idx, batch in enumerate(pbar):
        if graph:
            sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch
        batch_size = tbatch.shape[1]
        if batch_size == 1:
            continue

        if graph:
            output = net(sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths)
        else:
            output = net(sbatch, tbatch, turn_lengths)
                
        if type(output) == tuple:
            # VHRED model, KL divergence add to the loss
            if len(output) == 2:
                output, _ = output
            elif len(output) == 3:
                output, _, _ = output
            else:
                raise Exception('[!] wrong')
        loss = criterion(output[1:].view(-1, vocab_size),
                         tbatch[1:].contiguous().view(-1))
                
        total_loss += loss.item()
        batch_num += 1
        
        pbar.set_description(f'batch {idx}, dev loss: {round(loss.item(), 4)}')

        if debug:
            # show the output result, output: [length, batch, vocab_size]
            ipdb.set_trace()
            utterance = output[:, 0, :].squeeze(1)    # [length, vocab_size]
            word_idx = utterance.data.max(1)[1]       # [length]

    return round(total_loss / batch_num, 4)


def translate(data_iter, net, **kwargs):
    '''
    PPL calculating refer to: https://github.com/hsgodhia/hred
    '''
    net.eval()
    tgt_vocab = load_pickle(kwargs['tgt_vocab'])
    src_vocab = load_pickle(kwargs['src_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    # calculate the loss
    criterion = nn.NLLLoss(ignore_index=tgt_w2idx['<pad>'])
    total_loss, batch_num = 0.0, 0
    
    # translate, which is the same as the translate.py
    with open(kwargs['pred'], 'w') as f:
        pbar = tqdm(data_iter)
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
                output, _ = net.predict(sbatch, gbatch, 
                                          subatch, tubatch, 
                                          len(tbatch), turn_lengths,
                                          loss=True)
            else:
                output, _ = net.predict(sbatch, len(tbatch), turn_lengths,
                                        loss=True)
                
            # true working ppl by using teach_force
            with torch.no_grad():
                if kwargs['graph'] == 1:
                    f_l = net(sbatch, tbatch, gbatch, 
                              subatch, tubatch, turn_lengths)
                else:
                    f_l = net(sbatch, tbatch, turn_lengths)
                    if type(f_l) == tuple:
                        f_l = f_l[0]
            # teach_force over
            
            loss = criterion(f_l[1:].view(-1, len(tgt_w2idx)),
                             tbatch[1:].contiguous().view(-1))
            batch_num += 1
            total_loss += loss.item()
                
            # ipdb.set_trace()
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
                    src_endx = min(src_endx, src_endx_)
                    src = src[1:src_endx]
                    src = ' '.join(num2seq(src, src_idx2w))

                f.write(f'- src: {src}\n')
                f.write(f'- ref: {ref}\n')
                f.write(f'- tgt: {tgt}\n\n')
                        
    l = round(total_loss / batch_num, 4)
    print(f'[!] write the translate result into {kwargs["pred"]}')
    print(f'[!] test loss: {l}, test ppl: {round(math.exp(l), 4)}', 
          file=open(f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/ppl.txt', 'a'))
    
    return math.exp(l)

    
def write_into_tb(pred_path, writer, writer_str, epoch, ppl, bleu_mode, model, dataset):
    # obtain the performance
    print(f'[!] measure the performance and write into tensorboard')
    with open(pred_path) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            line = line.lower()    # lower the case
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())

    assert len(ref) == len(tgt)

    # ROUGE
    rouge_sum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, counter = 0, 0, 0, 0, 0, 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        rouge_sum += cal_ROUGE(rr, cc)
        # rouge_sum += 0.01
        counter += 1
    
    # BlEU
    refs, tgts = [' '.join(i) for i in ref], [' '.join(i) for i in tgt]
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_BLEU(refs, tgts)
        
    if bleu_mode == 'perl':
        bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_BLEU_perl(dataset, model)
    # elif bleu_mode == 'nlg-eval':
    # elif bleu_mode == 'nltk':
        

    # Distinct-1, Distinct-2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    rdistinct_1, rdistinct_2 = cal_Distinct(references)
    
    # Embedding-based metric: Embedding Average (EA), Vector Extrema (VX), Greedy Matching (GM)
    # load the dict
    # with open('./data/glove_embedding.pkl', 'rb') as f:
    #     dic = pickle.load(f)
    dic = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    print('[!] load the GoogleNews 300 word2vector by gensim over')
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        ea_sum += cal_embedding_average(rr, cc, dic)
        vx_sum += cal_vector_extrema(rr, cc, dic)
        gm_sum += cal_greedy_matching_matrix(rr, cc, dic)
        counterp += 1
        
    # write into the tensorboard
    writer.add_scalar(f'{writer_str}-Performance/PPL', ppl, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-1', bleu1_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-2', bleu2_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-3', bleu3_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-4', bleu4_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/ROUGE', rouge_sum / counter, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Distinct-1', distinct_1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Distinct-2', distinct_2, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-1', rdistinct_1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-2', rdistinct_2, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Embedding-Average', ea_sum / counterp, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Vector-Extrema', vx_sum / counterp, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Greedy-Matching', gm_sum / counterp, epoch)
    
    # write now
    writer.flush()


def main(**kwargs):
    # tensorboard 
    writer = SummaryWriter(log_dir=f'./tblogs/{kwargs["dataset"]}/{kwargs["model"]}')

    # load vocab
    src_vocab, tgt_vocab = load_pickle(kwargs['src_vocab']), load_pickle(kwargs['tgt_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    print(f'[!] load vocab over, src/tgt vocab size: {len(src_idx2w)}, {len(tgt_w2idx)}')

    # pretrained path
    pretrained = None

    # create the net
    if kwargs['model'] == 'HRED':
        net = HRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'VHRED':
        net = VHRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                    pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                    utter_n_layer=kwargs['utter_n_layer'], 
                    dropout=kwargs['dropout'],
                    z_hidden=kwargs['z_hidden'],
                    pretrained=pretrained)
    elif kwargs['model'] == 'KgCVAE':
        net = KgCVAE(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                    pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'],
                    eos=tgt_w2idx['<eos>'], unk=tgt_w2idx['<unk>'],
                    utter_n_layer=kwargs['utter_n_layer'], 
                    dropout=kwargs['dropout'],
                    z_hidden=kwargs['z_hidden'],
                    pretrained=pretrained)
    elif kwargs['model'] == 'HRAN':
        net = HRAN(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'HRAN-ablation':
        net = HRAN_ablation(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'DSHRED':
        net = DSHRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                     kwargs['utter_hidden'], kwargs['context_hidden'],
                     kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                     pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                     utter_n_layer=kwargs['utter_n_layer'], 
                     dropout=kwargs['dropout'],
                     pretrained=pretrained)
    elif kwargs['model'] == 'DSHRED_RA':
        net = DSHRED_RA(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                     kwargs['utter_hidden'], kwargs['context_hidden'],
                     kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                     pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                     utter_n_layer=kwargs['utter_n_layer'], 
                     dropout=kwargs['dropout'],
                     pretrained=pretrained)
    elif kwargs['model'] == 'WSeq':
        net = WSeq(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'WSeq_RA':
        net = WSeq_RA(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], 
                   utter_n_layer=kwargs['utter_n_layer'], 
                   dropout=kwargs['dropout'],
                   pretrained=pretrained)
    elif kwargs['model'] == 'Transformer':
        net = Transformer(len(src_w2idx), len(tgt_w2idx), kwargs['d_model'], 
                          kwargs['nhead'], kwargs['num_encoder_layers'], 
                          kwargs['dim_feedforward'], 
                          utter_n_layer=kwargs['utter_n_layer'],
                          dropout=kwargs['dropout'], sos=tgt_w2idx['<sos>'], 
                          pad=tgt_w2idx['<pad>'], teach_force=kwargs['teach_force'],
                          position_embed_size=kwargs['position_embed_size'])
    elif kwargs['model'] == 'MReCoSa':
        net = MReCoSa(len(src_w2idx), 512, len(tgt_w2idx), 512, 512,
                      teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'],
                      sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                      utter_n_layer=kwargs['utter_n_layer'], 
                      pretrained=pretrained)
    elif kwargs['model'] == 'MReCoSa_RA':
        net = MReCoSa_RA(len(src_w2idx), 512, len(tgt_w2idx), 512, 512,
                      teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'],
                      sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                      utter_n_layer=kwargs['utter_n_layer'], 
                      pretrained=pretrained)
    elif kwargs['model'] == 'Seq2Seq':
        net = Seq2Seq(len(src_w2idx), kwargs['embed_size'], len(tgt_w2idx), 
                      kwargs['utter_hidden' ], 
                      kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                      pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'],
                      dropout=kwargs['dropout'], 
                      utter_n_layer=kwargs['utter_n_layer'], 
                      pretrained=pretrained)
    elif kwargs['model'] == 'Seq2Seq_MHA':
        net = Seq2Seq_Multi_Head(len(src_w2idx), kwargs['embed_size'], 
                                 len(tgt_w2idx), kwargs['utter_hidden' ], 
                                 kwargs['decoder_hidden'],
                                 teach_force=kwargs['teach_force'],
                                 pad=tgt_w2idx['<pad>'], 
                                 sos=tgt_w2idx['<sos>'],
                                 dropout=kwargs['dropout'], 
                                 utter_n_layer=kwargs['utter_n_layer'], 
                                 pretrained=pretrained,
                                 nhead=kwargs['nhead'])
    elif kwargs['model'] == 'MTGCN':
        net = MTGCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'], 
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                    teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'], 
                    sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                    utter_n_layer=kwargs['utter_n_layer'],
                    context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'MTGAT':
        net = MTGAT(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'], 
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                    teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'], 
                    sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                    utter_n_layer=kwargs['utter_n_layer'],
                    context_threshold=kwargs['context_threshold'],
                    heads=kwargs['gat_heads'])
    elif kwargs['model'] == 'GatedGCN':
        net = GatedGCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'], 
                    kwargs['utter_hidden'], kwargs['context_hidden'],
                    kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                    teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'], 
                    sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                    utter_n_layer=kwargs['utter_n_layer'],
                    context_threshold=kwargs['context_threshold'])
    else:
        raise Exception(f'[!] wrong model named {kwargs["model"]}')

    if torch.cuda.is_available():
        net.cuda()

    print('[!] Net:')
    print(net)

    print(f'[!] Parameters size: {sum(x.numel() for x in net.parameters())}')
    print(f'[!] Optimizer Adam')
    optimizer = optim.Adam(net.parameters(), lr=kwargs['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                               mode='min',
                                               factor=kwargs['lr_gamma'],
                                               patience=kwargs['patience'],
                                               verbose=True,
                                               cooldown=0,
                                               min_lr=kwargs['lr_mini'])
    pbar = tqdm(range(1, kwargs['epochs'] + 1))
    training_loss, validation_loss = [], []
    min_loss = np.inf
    patience = 0
    best_val_loss = None
    teacher_force_ratio = kwargs['teach_force']    # default 1
    teacher_force_ratio_counter = kwargs['dynamic_tfr_counter']
    # holder = teacher_force_ratio_counter
    holder = 0
    kl_mult = 0.0    # VHRED only

    # train
    for epoch in pbar:
        # prepare dataset
        if kwargs['hierarchical'] == 1:
            if kwargs['graph'] == 1:
                train_iter = get_batch_data_graph(kwargs['src_train'], 
                                                  kwargs['tgt_train'], 
                                                  kwargs['train_graph'],
                                                  kwargs['src_vocab'], 
                                                  kwargs['tgt_vocab'],
                                                  kwargs['batch_size'], 
                                                  kwargs['maxlen'],
                                                  kwargs['tgt_maxlen'])
                test_iter = get_batch_data_graph(kwargs['src_test'], 
                                                 kwargs['tgt_test'], 
                                                 kwargs['test_graph'],
                                                 kwargs['src_vocab'], 
                                                 kwargs['tgt_vocab'],
                                                 kwargs['batch_size'], 
                                                 kwargs['maxlen'],
                                                 kwargs['tgt_maxlen'])
                dev_iter = get_batch_data_graph(kwargs['src_dev'], 
                                                kwargs['tgt_dev'], 
                                                kwargs['dev_graph'],
                                                kwargs['src_vocab'], 
                                                kwargs['tgt_vocab'],
                                                kwargs['batch_size'], 
                                                kwargs['maxlen'],
                                                kwargs['tgt_maxlen'])
            else:
                if kwargs['model'] in ['VHRED','KgCVAE']:
                    ld = False
                else:
                    ld = True
                train_iter = get_batch_data(kwargs['src_train'], 
                                            kwargs['tgt_train'],
                                            kwargs['src_vocab'], 
                                            kwargs['tgt_vocab'], 
                                            kwargs['batch_size'], 
                                            kwargs['maxlen'],
                                            kwargs['tgt_maxlen'],
                                            ld=ld)
                test_iter = get_batch_data(kwargs['src_test'], 
                                           kwargs['tgt_test'],
                                           kwargs['src_vocab'], 
                                           kwargs['tgt_vocab'],
                                           kwargs['batch_size'],
                                           kwargs['maxlen'],
                                           kwargs['tgt_maxlen'],
                                           ld=ld)
                dev_iter = get_batch_data(kwargs['src_dev'], 
                                          kwargs['tgt_dev'],
                                          kwargs['src_vocab'], 
                                          kwargs['tgt_vocab'],
                                          kwargs['batch_size'],
                                          kwargs['maxlen'],
                                          kwargs['tgt_maxlen'],
                                          ld=ld)
        else:
            train_iter = get_batch_data_flatten(kwargs['src_train'],
                                                    kwargs['tgt_train'],
                                                    kwargs['src_vocab'],
                                                    kwargs['tgt_vocab'],
                                                    kwargs['batch_size'],
                                                    kwargs['maxlen'],
                                                    kwargs['tgt_maxlen'])
            test_iter = get_batch_data_flatten(kwargs['src_test'], 
                                                   kwargs['tgt_test'],
                                                   kwargs['src_vocab'], 
                                                   kwargs['tgt_vocab'],
                                                   kwargs['batch_size'], 
                                                   kwargs['maxlen'],
                                                   kwargs['tgt_maxlen'])
            dev_iter = get_batch_data_flatten(kwargs['src_dev'], 
                                                  kwargs['tgt_dev'],
                                                  kwargs['src_vocab'],
                                                  kwargs['tgt_vocab'],
                                                  kwargs['batch_size'],
                                                  kwargs['maxlen'],
                                                  kwargs['tgt_maxlen'])

                
        # ========== train session begin ==========
        writer_str = f'{kwargs["dataset"]}'
        train_loss = train(train_iter, net, optimizer, len(tgt_w2idx),
                           tgt_w2idx['<pad>'], 
                           grad_clip=kwargs['grad_clip'], debug=kwargs['debug'],
                           transformer_decode=kwargs['transformer_decode'],
                           graph=kwargs['graph']==1, kl_mult=kl_mult, 
                           kl_annealing_iter=kwargs['kl_annealing_iter'])
        if type(train_loss) == tuple:
            # VHRED
            train_loss, kl_mult = train_loss
        with torch.no_grad():
            val_loss = validation(dev_iter, net, len(tgt_w2idx), 
                                  tgt_w2idx['<pad>'],
                                  transformer_decode=kwargs['transformer_decode'],
                                  graph=kwargs['graph']==1)
        
        # add loss scalar to tensorboard
        # and write the lr schedule, and teach force
        writer.add_scalar(f'{writer_str}-Loss/train', train_loss, epoch)
        writer.add_scalar(f'{writer_str}-Loss/dev', val_loss, epoch)
        writer.add_scalar(f'{writer_str}-Loss/lr', 
                          optimizer.state_dict()['param_groups'][0]['lr'], 
                          epoch)
        writer.add_scalar(f'{writer_str}-Loss/teach', net.teach_force,
                          epoch)
        
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
                          
        # checkpoint state
        # try not save all the checkpoints
        if epoch > int(kwargs['epochs'] * 0.8):
            optim_state = optimizer.state_dict()
            state = {'net': net.state_dict(), 'opt': optim_state, 
                     'epoch': epoch, 'patience': patience}
            torch.save(state, 
                           f'./ckpt/{kwargs["dataset"]}/{kwargs["model"]}/vloss_{val_loss}_epoch_{epoch}.pt')
        
        # translate on test dataset
        with torch.no_grad():
            ppl = translate(test_iter, net, **kwargs)
        
        # write the performance into the tensorboard
        write_into_tb(kwargs['pred'], writer, writer_str, epoch, ppl, kwargs['bleu'], kwargs['model'], kwargs['dataset'])
        
        pbar.set_description(f'Epoch: {epoch}, tfr: {round(teacher_force_ratio, 4)}, loss(train/dev): {train_loss}/{val_loss}, ppl(dev/test): {round(math.exp(val_loss), 4)}/{round(ppl, 4)}, patience: {patience}/{kwargs["patience"]}')
        
        # dynamic teach_force_ratio
        if epoch > kwargs["dynamic_tfr"]:
            if holder == 0:
                teacher_force_ratio -= kwargs["dynamic_tfr_weight"]
                teacher_force_ratio = max(kwargs['dynamic_tfr_threshold'], teacher_force_ratio)
                holder = teacher_force_ratio_counter
            else:
                holder -= 1
            net.teach_force = teacher_force_ratio
        
        # lr schedule change, monitor the evaluation loss
        scheduler.step(val_loss)
        

    pbar.close()
    writer.close()
    print(f'[!] Done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--src_train', type=str, default=None, help='src train file')
    parser.add_argument('--tgt_train', type=str, default=None, help='src train file')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--src_dev', type=str, default=None, help='src dev file')
    parser.add_argument('--tgt_dev', type=str, default=None, help='tgt dev file')
    parser.add_argument('--min_threshold', type=int, default=0, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=20, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--utter_hidden', type=int, default=150, 
                        help='utterance encoder hidden size')
    parser.add_argument('--teach_force', type=float, default=0.5, 
                        help='teach force ratio')
    parser.add_argument('--context_hidden', type=int, default=150, 
                        help='context encoder hidden size')
    parser.add_argument('--decoder_hidden', type=int, default=150, 
                        help='decoder hidden size')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200, 
                        help='embedding layer size')
    parser.add_argument('--patience', type=int, default=5, 
                        help='patience for early stop')
    parser.add_argument('--dataset', type=str, default='dailydialog', 
                        help='dataset for training')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='grad clip')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50, 
                        help='the maxlen of the utterance')
    parser.add_argument('--tgt_maxlen', type=int, default=50, 
                        help='the maxlen of the responses')
    parser.add_argument('--utter_n_layer', type=int, default=1, 
                        help='layers of the utterance encoder')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--hierarchical', type=int, default=1, 
                        help='Whether hierarchical architecture')
    parser.add_argument('--transformer_decode', type=int, default=0,
                        help='transformer decoder need a different training process')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='d_model for transformer')
    parser.add_argument('--nhead', type=int, default=8, 
                        help='head number for transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--warmup_step', type=int, default=4000, help='warm up steps')
    parser.add_argument('--contextrnn', dest='contextrnn', action='store_true')
    parser.add_argument('--no-contextrnn', dest='contextrnn', action='store_false')
    parser.add_argument('--position_embed_size', type=int, default=30)
    parser.add_argument('--graph', type=int, default=0)
    parser.add_argument('--train_graph', type=str, default=None, 
                        help='train graph data path')
    parser.add_argument('--test_graph', type=str, default=None, 
                        help='test graph data path')
    parser.add_argument('--dev_graph', type=str, default=None, 
                        help='dev graph data path')
    parser.add_argument('--context_threshold', type=int, default=3, 
                        help='low turns filter')
    parser.add_argument('--pred', type=str, default=None, 
                        help='the file save the output')
    parser.add_argument('--dynamic_tfr', type=int, default=20, 
                        help='begin to use the dynamic teacher forcing ratio, each ratio minus the tfr_weight')
    parser.add_argument('--dynamic_tfr_weight', type=float, default=0.05)
    parser.add_argument('--dynamic_tfr_counter', type=int, default=5)
    parser.add_argument('--dynamic_tfr_threshold', type=float, default=0.3)
    parser.add_argument('--bleu', type=str, default='nltk', help='nltk or perl')
    parser.add_argument('--lr_mini', type=float, default=1e-6, help='minial lr (threshold)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='lr schedule gamma factor')
    parser.add_argument('--gat_heads', type=int, default=5, help='heads of GAT layer')
    parser.add_argument('--z_hidden', type=int, default=100, help='z_hidden for VHRED')
    parser.add_argument('--kl_annealing_iter', type=int, default=20000, help='KL annealing for VHRED')

    args = parser.parse_args()

    # show the parameters and write into file
    print('[!] Parameters:')
    print(args)
    with open(f'./processed/{args.dataset}/{args.model}/metadata.txt', 'w') as f:
        print(args, file=f)

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # main function
    args_dict = vars(args)
    main(**args_dict)
