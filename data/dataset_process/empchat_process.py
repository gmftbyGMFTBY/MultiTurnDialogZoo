import re
import sys
import ipdb
import numpy as np


'''
Donwload the file: https://github.com/facebookresearch/EmpatheticDialogues
'''


def load_file(path):
    with open(path) as f:
        cache = f.readline().split(',')[0] 
        corpus, history = [], []
        user = 0
        for line in f.readlines():
            items = line.strip().split(',')
            utterance = f'<user{user}> ' + items[5].replace('_comma_', ',')
            if items[0] == cache:
                history.append(utterance)
            else:
                if history:
                    corpus.append(history)    # append the dialogue
                history = [utterance]
            user = 1 if user == 0 else 0
            cache = items[0]

    avg_turn = np.mean([len(i) for i in corpus])
    max_turn = max([len(i) for i in corpus])
    min_turn = min([len(i) for i in corpus])
    print(f'[!] find {len(corpus)} dialogue, turns(avg/max/min): {avg_turn}/{max_turn}/{min_turn}')
    return corpus

def write_file(mode, corpus):
    src_f = open(f'src-{mode}.txt', 'w')
    tgt_f = open(f'tgt-{mode}.txt', 'w')

    for dialog in corpus:
        for i in range(1, len(dialog)):
            src = ' __eou__ '.join(dialog[:i])
            src_f.write(f'{src}\n')
            tgt_f.write(f'{dialog[i]}\n')

    src_f.close()
    tgt_f.close()
    print(f'[!] write into {mode} file over ...')

            
if __name__  == '__main__':
    train_data = load_file('train.csv')
    test_data = load_file('test.csv')
    dev_data = load_file('valid.csv')

    write_file('train', train_data)
    write_file('test', test_data)
    write_file('dev', dev_data)
