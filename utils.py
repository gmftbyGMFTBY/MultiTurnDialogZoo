#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14

'''
utils function for training the model
'''

import numpy as np
import argparse
from collections import Counter
import pickle
import os
import torch
import nltk
import ipdb
import random
from tqdm import tqdm
from scipy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from model.layers import NoamOpt


# ========== jaccard, cosine + tf, cosine + tf-idf, GloVe ========== #
# ========== refer to: https://blog.csdn.net/asd991936157/article/details/77011206 ========== #
def jaccard_similarity(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    vectors = np.array([s1, s2])
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator


def cosine_similarity_tf(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    return np.dot(s1, s2) / (norm(s1) * norm(s2))


def cosine_similarity_tfidf(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    return np.dot(s1, s2) / (norm(s2) * norm(s2))


def load_glove_embedding(path, dimension=300):
    if os.path.exists('./data/glove_embedding.pkl'):
        print(f'[!] load from the preprocessed embeddings ./data/glove_embedding.pkl')
        return load_pickle('./data/glove_embedding.pkl')
    with open(path) as f:
        vocab = {}
        for line in f.readlines():
            line = line.split()
            assert len(line) > 300
            vector = np.array(list(map(float, line[-300:])), dtype=np.float)    # [300]
            vocab[line[0]] = vector
    vocab['<unk>'] = np.random.rand(dimension)
    print(f'[!] load GloVe word embedding from {path}')
    with open('./data/glove_embedding.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    return vocab


def sent2glove(vocab, sent):
    s = np.zeros(vocab['<unk>'].shape, dtype=np.float)
    for word in nltk.word_tokenize(sent):
        # ipdb.set_trace()
        vector = vocab.get(word, vocab['<unk>'])
        s += vector
    return s

# ================================================================================= #
    

def load_best_model(dataset, model, net, min_threshold, max_threshold):
    path = f'./ckpt/{dataset}/{model}/'
    best_loss, best_file, best_epoch = np.inf, None, -1

    for file in os.listdir(path):
        _, val_loss, _, epoch = file.split('_')
        epoch = epoch.split('.')[0]
        val_loss, epoch = float(val_loss), int(epoch)

        if min_threshold <= epoch <= max_threshold and epoch > best_epoch:
            best_file = file
            best_epoch = epoch

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold ({min_threshold}, {max_threshold})')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception('[!] No saved model')
        

def cos_similarity(gr, ge):
    # word embedding
    return np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))


def num2seq(src, idx2w):
    # number to word sequence, src: [maxlen]
    return [idx2w[int(i)] for i in src]


def transformer_list(obj):
    # transformer [batch, turns, lengths] into [turns, batch, lengths]
    # turns are all the same for each batch
    turns = []
    batch_size, turn_size = len(obj), len(obj[0])
    for i in range(turn_size):
        turns.append([obj[j][i] for j in range(batch_size)])    # [batch, lengths]
    return turns


def pad_sequence(pad, batch, bs):
    maxlen = max([len(batch[i]) for i in range(bs)])
    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))


def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def generate_vocab(files, vocab, cutoff=50000):
    # training and validation files, input vocab and output vocab file
    words = []
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                words.extend(nltk.word_tokenize(line))
    words = Counter(words)
    print(f'[!] whole vocab size: {len(words)}')
    words = words.most_common(cutoff)
    # special token, start of dialog (sod) and end of dialog (eod)
    words.extend([('<sod>', 1), ('<eod>', 1), 
                  ('<sos>', 1), ('<eos>', 1), 
                  ('<unk>', 1), ('<pad>', 1),])
    w2idx = {item[0]:idx for idx, item in enumerate(words)}
    idx2w = [item[0] for item in words]
    with open(vocab, 'wb') as f:
        pickle.dump((w2idx, idx2w), f)
    print(f'[!] Save the vocab into {vocab}, vocab_size: {len(w2idx)}')


def generate_bert_embedding(vocab, path):
    bc = BertClient()
    w2idx, idx2w = vocab
    words = [word for word in w2idx]
    emb = bc.encode(words)    # [vocab_size, 768], ndarray

    # save into the processed folder
    with open(path, 'wb') as f:
        pickle.dump(emb, f)

    print(f'[!] write the bert embedding into {path}')
    
    
def create_the_graph(turns, vocab, weights=[1, 0.1], threshold=0.4, bidir=False):
    '''create the weighted directed graph of one conversation
    sequenutial edge, user connected edge, [BERT/PMI] edge
    param: turns: [turns(user, utterance)]
    param: weights: [sequential_w, user_w]
    output: [2, num_edges], [num_edges]'''
    edges = {}
    s_w, u_w = weights
    # sequential edges, (turn_len - 1)
    turn_len = len(turns)
    se, ue, pe = 0, 0, 0
    for i in range(turn_len - 1):
        edges[(i, i + 1)] = [s_w]
        se += 1
        
    # user edge
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                if 'user0' in turns[i]:
                    useri = 'user0'
                elif 'user1' in turns[i]:
                    useri = 'user1'
                if 'user0' in turns[j]:
                    userj = 'user0'
                elif 'user1' in turns[j]:
                    userj = 'user1'
                if useri == userj:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(u_w)
                    else:
                        edges[(i, j)] = [u_w]
                    ue += 1
    
    # distance
    utterances = []
    for utterance in turns:
        utterance = utterance.replace('user0', '').strip()
        utterance = utterance.replace('user1', '').strip()
        if utterance:
            utterances.append(utterance)
        else:
            utterances.append('<unk>')
            
    # ========== TFIDF, Counter, GloVe embedding ========== #
    count_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
    count_vectors = count_vectorizer.fit_transform(utterances).toarray()    # [datasize, word_size]
    # print(f'[!] over the count fit_transform, shape {count_vectors.shape}')
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    tfidf_vectors = tfidf_vectorizer.fit_transform(utterances).toarray()    # [datasize, word_size]
    # print(f'[!] over the tfidf fit_transform, shape: {tfidf_vectors.shape}')
        
    # add the edges accorading to the TFIDF and Counter information
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                utter1, utter2 = count_vectors[i], count_vectors[j]
                # jaccard
                jaccard = jaccard_similarity(utter1, utter2)
                # cosine + tf
                cosine_tf = cosine_similarity_tf(utter1, utter2)
                # cosine + tfidf 
                cosine_tf_idf = cosine_similarity_tfidf(utter1, utter2)
                # glove embedding
                utter1 = sent2glove(vocab, utterances[i])
                utter2 = sent2glove(vocab, utterances[j])
                glove = cos_similarity(utter1, utter2)
                
                weight = max([jaccard, cosine_tf, cosine_tf_idf, glove])
                
                if weight >= threshold:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(weight * u_w)
                    else:
                        edges[(i, j)] = [weight * u_w]
                    pe += 1

    # clean the edges
    e, w = [[], []], []
    for src, tgt in edges.keys():
        e[0].append(src)
        e[1].append(tgt)
        w.append(max(edges[(src, tgt)]))
        
        if bidir and src != tgt:
            e[0].append(tgt)
            e[1].append(src)
            w.append(max(edges[(src, tgt)]))

    return (e, w), se, ue, pe


def load_data(src, tgt, src_vocab, tgt_vocab, maxlen):
    # convert dataset into src: [datasize, turns, lengths]
    # convert dataset into tgt: [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src_user, tgt_user = [], []
    user_vocab = ['user0', 'user1']
    
    # src 
    with open(src) as f:
        src_dataset = []
        for line in f.readlines():
            utterances = line.split('__eou__')
            turn = []
            srcu = []
            for utterance in utterances:
                if '<user0>' in utterance: user_c, user_cr = '<user0>', 'user0'
                elif '<user1>' in utterance: user_c, user_cr = '<user1>', 'user1'
                utterance = utterance.replace(user_c, user_cr).strip()
                line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]
                if len(line) > maxlen:
                    line = [src_w2idx['<sos>'], line[1]] + line[-maxlen:]
                turn.append(line)
                srcu.append(user_vocab.index(user_cr))
            src_dataset.append(turn)
            src_user.append(srcu)

    # tgt
    with open(tgt) as f:
        tgt_dataset = []
        for line in f.readlines():
            if '<user0>' in line: user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line: user_c, user_cr = '<user1>', 'user1'
            line = line.replace(user_c, user_cr).strip()
            line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [tgt_w2idx['<eos>']]
            if len(line) > maxlen:
                line = [tgt_w2idx['<sos>'], line[1]] + line[-maxlen:]
            tgt_dataset.append(line)
            tgt_user.append(user_vocab.index(user_cr))
 
    # src_user: [datasize, turn], tgt_user: [datasize]
    return src_dataset, src_user, tgt_dataset, tgt_user


def generate_graph(dialogs, path, threshold=0.75, bidir=False):
    # dialogs: [datasize, turns]
    # return: [datasize, (2, num_edges)/ (num_edges)]
    # **make sure the bert-as-service is running**
    edges = []
    se, ue, pe = 0, 0, 0
    print(f'[!] prepare to load the GloVe 300 embedding from /home/lt/data/File/wordembedding/glove/glove.6B.300d.txt (you can change this path)')
    vocab = load_glove_embedding('/home/lt/data/File/wordembedding/glove/glove.6B.300d.txt')
    for dialog in tqdm(dialogs):
        edge, ses, ueu, pep = create_the_graph(dialog, vocab, threshold=threshold,
                                               bidir=bidir)
        se += ses
        ue += ueu
        pe += pep
        edges.append(edge)

    with open(path, 'wb') as f:
        pickle.dump(edges, f)

    print(f'[!] graph information is converted in {path}')
    print(f'[!] Avg se: {round(se / len(dialogs), 4)}; Avg ue: {round(ue / len(dialogs), 4)}; Avg pe: {round(pe / len(dialogs), 4)}')


def idx2sent(data, vocab):
    # turn the index to the sentence
    # data: [datasize, turn, length]
    # user: [datasize, turn]
    # return: [datasize, (user, turns)]
    _, idx2w = load_pickle(vocab)
    datasets = []
    for example in tqdm(data):
        # example: [turn, length], user: [turn]
        turns = []
        for turn in example:
            utterance = ' '.join([idx2w[w] for w in turn])
            utterance = utterance.replace('<sos>', '').replace('<eos>', '').strip()
            turns.append(utterance)
        datasets.append(turns)
    return datasets
    
    
def get_std_opt(model):
    # return the speical opt for transformer-based models
    return NoamOpt(model.embed_size, 2, 4000,
                   torch.optim.Adam(model.parameters(), 
                                    lr=0, 
                                    betas=(0.9, 0.98), 
                                    eps=1e-9))

# ========== stst of the graph ========== #
def analyse_graph(path, hops=3):
    '''
    This function analyzes the graph coverage stat of the graph in Dailydialog 
    and cornell dataset.
    Stat the context node coverage of each node in the conversation.
    :param: path, the path of the dataset graph file.
    '''
    def coverage(nodes, edges):
        # return the coverage information of each node
        # return list of tuple (context nodes, coverage nodes)
        # edges to dict
        e = {}
        for i, j in zip(edges[0], edges[1]):
            if i > j:
                continue
            if e.get(j, None):
                e[j].append(i)
            else:
                e[j] = [i]
        for key in e.keys():    # make the set
            e[key] = list(set(e[key]))
        collector = []
        for node in nodes:
            # context nodes
            context_nodes = list(range(0, node))
            if context_nodes:
                # ipdb.set_trace()
                # coverage nodes, BFS
                coverage_nodes, tools, tidx = [], [(node, 0)], 0
                while True:
                    try:
                        n, nidx = tools[tidx]
                    except:
                        break
                    if nidx < hops:
                        for src in e[n]:
                            if src not in tools:
                                tools.append((src, nidx + 1))
                            if src not in coverage_nodes:
                                coverage_nodes.append(src)
                    tidx += 1
                collector.append((len(context_nodes), len(coverage_nodes)))
        return collector
        
    graph = load_pickle(path)    # [datasize, ([2, num_edge], [num_edge])]
    avg_cover = []
    for idx, (edges, _) in enumerate(tqdm(graph)):
        # make sure the number of the nodes
        nodes = []
        for i, j in zip(edges[0], edges[1]):
            if i == j and i not in nodes:
                nodes.append(i)
        avg_cover.extend(coverage(nodes, edges))
        
    # ========== stat ========== #
    ratio = [i / j for i, j in avg_cover]
    print(f'[!] the avg graph coverage of the context is {round(np.mean(ratio), 4)}')
    
    
def Perturbations_test(src_test_in, src_test_out, mode=1):
    '''
    ACL 2019 Short paper:
    Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study
    
    ## Utterance-level
    1. Shuf: shuffles the sequence of utterances in the dialog history
    2. Rev:  reverses the order of utterances in the history (but maintains word order within each utterance)
    3.4. Drop: completely drops certain utterances (drop first / drop last)
    5. Truncate: that truncates the dialog history
    
    ## Word-level
    6. word-shuffle: randomly shuffles the words within an utterance
    7. reverse: reverses the ordering of words
    8. word-drop: drops 30% of the words uniformly
    9. noun-drop: drops all nouns
    10. verb-drop: drops all verbs
    '''
    
    # load the file
    with open(src_test_in) as f:
        corpus = []
        for line in f.readlines():
            line = line.strip()
            sentences = line.split('__eou__')
            sentences = [i.strip() for i in sentences]
            corpus.append(sentences)
    print(f'[!] load the data from {src_test_in}')
    
    print(f'[!] perburtation mode: {mode}')
    # perturbation
    new_corpus = []
    for i in corpus:
        if mode == 1:
            random.shuffle(i)
            new_corpus.append(i) 
        elif mode == 2:
            new_corpus.append(list(reversed(i)))
        elif mode == 3:
            if len(i) > 1:
                new_corpus.append(i[1:])
            else:
                new_corpus.append(i)
        elif mode == 4:
            if len(i) > 1:
                new_corpus.append(i[:-1])
            else:
                new_corpus.append(i)
        elif mode == 5:
            new_corpus.append([i[-1]])
        elif mode == 6:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                random.shuffle(words)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 7:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                s_.append(user + ' '.join(list(reversed(words))))
            new_corpus.append(s_)
        elif mode == 8:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                words = [w_ for w_ in words if random.random() > 0.3]
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 9:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                tagger = nltk.pos_tag(nltk.word_tokenize(s))
                words = []
                for w, t in tagger:
                    if t in ['NN', 'NNS', 'NNP', 'NNPS']:
                        continue
                    else:
                        words.append(w)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 10:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                tagger = nltk.pos_tag(nltk.word_tokenize(s))
                words = []
                for w, t in tagger:
                    if t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        continue
                    else:
                        words.append(w)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        else:
            raise Exception(f'[!] wrong mode: {mode}')
    
    # write the new source test file
    with open(src_test_out, 'w') as f:
        for i in new_corpus:
            i = ' __eou__ '.join(i)
            f.write(f'{i}\n')
    print(f'[!] write the new file into {src_test_out}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utils function')
    parser.add_argument('--mode', type=str, default='vocab', 
            help='how to run the utils.py, (vocab,)')
    parser.add_argument('--file', type=str, nargs='+', default=None, 
            help='file for generating the vocab')
    parser.add_argument('--vocab', type=str, default='',
            help='input or output vocabulary')
    parser.add_argument('--cutoff', type=int, default=50000,
            help='cutoff of the vocabulary')
    parser.add_argument('--pretrained', type=str, default=None,
            help='Pretrained embedding file')
    parser.add_argument('--graph', type=str, default='./processed/dailydialog/MTGCN/train-graph.pkl')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--src_vocab', type=str, default='./processed/dailydialog/MTGCN/iptvocab.pkl')
    parser.add_argument('--tgt_vocab', type=str, default='./processed/dailydialog/MTGCN/optvocab.pkl')
    parser.add_argument('--src', type=str, default='./data/dailydialog/src-train.pkl')
    parser.add_argument('--tgt', type=str, default='./data/dailydialog/tgt-train.pkl')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--bidir', dest='bidir', action='store_true')
    parser.add_argument('--no-bidir', dest='bidir', action='store_false')
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--perturbation_in', type=str, default=None)
    parser.add_argument('--perturbation_out', type=str, default=None)
    parser.add_argument('--perturbation_mode', type=int, default=1)
    args = parser.parse_args()

    mode = args.mode

    if mode == 'vocab':
        generate_vocab(args.file, args.vocab, cutoff=args.cutoff)
    elif mode == 'pretrained':
        with open(args.vocab, 'rb') as f:
            vocab = pickle.load(f)
        generate_bert_embedding(vocab, args.pretrained)
    elif mode == 'graph':
        # save the preprocessed data for generating graph
        src_dataset, src_user, tgt_dataset, tgt_user = load_data(args.src, args.tgt, args.src_vocab, args.tgt_vocab, args.maxlen)
        print(f'[!] load the cf mode dataset, prepare for preprocessing')
        ppdataset = idx2sent(src_dataset, args.src_vocab)
        print(f'[!] begin to create the graph')
        generate_graph(ppdataset, args.graph, threshold=args.threshold, bidir=args.bidir)
    elif mode == 'stat':
        analyse_graph(args.graph, hops=args.hops)
    elif mode == 'perturbation':
        if args.perturbation_in and args.perturbation_out:
            Perturbations_test(args.perturbation_in, args.perturbation_out, mode=args.perturbation_mode)
        else:
            print(f'[!] check the perturbation file path')
    else:
        print(f'[!] wrong mode to run the script')
