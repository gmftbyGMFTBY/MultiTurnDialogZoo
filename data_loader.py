#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import *
import ipdb
import random
import nltk


# ========== Remember to run the preprocess function for transformers ==========
'''
class GPT2Dataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)
    

def collate_fn(batch):
    """
    padding for transformers model
    """
    pad_id = 0    # for bert tokenizer, the pad id is 0
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    b = torch.tensor(input_ids, dtype=torch.long)
    if torch.cuda.is_available():
        b = b.cuda()
    return b
    
    
def get_batch_data_transformer(path, batch_size):
    # read the file
    with open(path) as f:
        data_list = []
        for line in f.readlines():
            line = line.strip()
            data_list.append(line)
    # create the dataset
    dataset = GPT2Dataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader
'''
# ========== ==========
    

def load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen):
    '''
    Used by vanilla seq2seq with attention and transformer
    '''
    # sort by the lengths
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    # sub function
    def load_(filename, w2idx, src=True):
        with open(filename) as f:
            dataset = []
            for line in tqdm(f.readlines()):
                line = clean(line)
                # if '<user0>' in line: user_c = '<user0>'
                # elif '<user1>' in line: user_c = '<user1>'
                line = line.replace('<user0>', 'user0')
                line = line.replace('<user1>', 'user1')
                line = [w2idx['<sos>']] + [w2idx.get(w, w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [w2idx['<eos>']]
                if len(line) > maxlen:
                    if src:
                        line = [w2idx['<sos>']] + line[-maxlen:]
                    else:
                        line = line[:maxlen] + [w2idx['<eos>']]
                dataset.append(line)
        return dataset

    src_dataset = load_(src, src_w2idx, src=True)    # [datasize, lengths]
    tgt_dataset = load_(tgt, tgt_w2idx, src=False)    # [datasize, lengths]
    print(f'[!] load dataset over')

    return src_dataset, tgt_dataset


def get_batch_data(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, plus=0):
    # batch and convert to tensor for training
    # batch according to the turns
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    src_dataset, _, tgt_dataset, _ = load_data(src, tgt, src_vocab, tgt_vocab, maxlen)
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        # print(fidx, bidx)

        # batch, [batch, turns, lengths], [batch, lengths]
        # shuffle
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        
        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue
        
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]

        # convert to [turns, batch, lengths], [batch, lengths]
        sbatch = transformer_list(sbatch)
        bs, ts = len(sbatch[0]), len(sbatch)

        # pad src by turns
        # create the lengths: [turns, batch] for sbatch
        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        # pad tgt, [batch, turns, lengths]
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # convert to tensor, change to cuda version tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()

        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths


def get_batch_data_flatten(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen):
    # flatten batch data for unHRED-based models (Seq2Seq)
    # return long context for predicting response
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    # [datasize, lengths], [datasize, lengths]
    src_dataset, tgt_dataset = load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen)

    turns = [len(i) for i in src_dataset]
    turnsidx = np.argsort(turns)

    # sort by the lengths
    src_dataset = [src_dataset[i] for i in turnsidx]
    tgt_dataset = [tgt_dataset[i] for i in turnsidx]

    # generate the batch
    turns = [len(i) for i in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        
        bs = len(sbatch)

        # pad sbatch and tbatch
        turn_lengths = [len(sbatch[i]) for i in range(bs)]
        pad_sequence(src_w2idx['<pad>'], sbatch, bs)
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # [seq_len, batch]
        sbatch = torch.tensor(sbatch, dtype=torch.long).transpose(0, 1)
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths
        
        
def get_batch_data_flatten_tf(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen):
    '''
    1. No turn_lengths because of transformer
    2. src_key_padding_mask and tgt_key_padding_mask
    '''
    # for transformer, return
    # src_mask and tgt_mask
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    # [datasize, lengths], [datasize, lengths]
    src_dataset, tgt_dataset = load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen)

    turns = [len(i) for i in src_dataset]
    turnsidx = np.argsort(turns)

    # sort by the lengths
    src_dataset = [src_dataset[i] for i in turnsidx]
    tgt_dataset = [tgt_dataset[i] for i in turnsidx]

    # generate the batch
    turns = [len(i) for i in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        
        bs = len(sbatch)

        # pad sbatch and tbatch, [batch, seq]
        pad_sequence(src_w2idx['<pad>'], sbatch, bs)
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # obtain the mask
        src_key_padding_mask, tgt_key_padding_mask = [], []
        for src_seq, tgt_seq in zip(sbatch, tbatch):
            src_key_padding_mask.append([True if i == src_w2idx['<pad>'] else False for i in src_seq])
            tgt_key_padding_mask.append([True if i == tgt_w2idx['<pad>'] else False for i in tgt_seq])
        
        # [seq_len, batch]
        sbatch = torch.tensor(sbatch, dtype=torch.long).transpose(0, 1)
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        src_key_padding_mask = torch.tensor(src_key_padding_mask)
        tgt_key_padding_mask = torch.tensor(tgt_key_padding_mask)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()
            src_key_padding_mask = src_key_padding_mask.cuda()
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()

        fidx = bidx

        yield sbatch, tbatch, src_key_padding_mask, tgt_key_padding_mask
        
        
def get_batch_data_graph(src, tgt, graph, src_vocab, tgt_vocab, 
                         batch_size, maxlen, plus=0):
    '''get batch data of hierarchical and graph mode
    return data:
    - sbatch: [turn, batch, length]
    - tbatch: [batch, length]
    - gbatch: [batch, ([2, num_edge], [num_edge])]
    - turn_lengths: [batch]
    '''
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    src_dataset, src_user, tgt_dataset, tgt_user = load_data(src, tgt, src_vocab, tgt_vocab, maxlen)
    graph = load_pickle(graph)
    
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)
    
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    graph = [graph[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        
        sbatch, tbatch, gbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx], graph[fidx:bidx]
        subatch, tubatch = src_user[fidx:bidx], tgt_user[fidx:bidx]
        
        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue
            
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]   # [batch, turns, lengths]
        tbatch = [tbatch[idx] for idx in shuffleidx]   # [batch, lengths]
        gbatch = [gbatch[idx] for idx in shuffleidx]   # [batch, ([2, edges_num], [edges_num]),]
        
        sbatch = transformer_list(sbatch)    # [turns, batch, lengths]
        bs, ts = len(sbatch[0]), len(sbatch)
        
        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # convert to tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)    # [turns, seq_len, batch]
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)    # [seq_len, batch]
        subatch = torch.tensor(subatch, dtype=torch.long).transpose(0, 1)   # [turns, batch]
        tubatch = torch.tensor(tubatch, dtype=torch.long)    # [batch]
        
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)     # [batch]
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            turn_lengths = turn_lengths.cuda()
            subatch = subatch.cuda()
            tubatch = tubatch.cuda()
            
        fidx = bidx
        yield sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths



if __name__ == "__main__":
    batch_num = 0
    src_w2idx, src_idx2w = load_pickle('./processed/dailydialog/iptvocab.pkl')
    tgt_w2idx, tgt_idx2w = load_pickle('./processed/dailydialog/optvocab.pkl')
    torch.cuda.set_device(2)
    for sbatch, tbatch, turn_lengths in get_batch_data_flatten('./data/dailydialog/src-train.txt', 
            './data/dailydialog/tgt-train.txt',
                                         './processed/dailydialog/iptvocab.pkl',
                                         './processed/dailydialog/optvocab.pkl',
                                         32, 50):
        ipdb.set_trace()
        print(len(sbatch), tbatch.shape, turn_lengths.shape)
        # if len(sbatch) == 3:
        #     ipdb.set_trace()
        batch_num += 1
    print('Batch_num:', batch_num)
