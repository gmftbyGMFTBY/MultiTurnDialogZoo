import tqdm
import ipdb

'''
Download PersonaChat / Dailydialog / DSTC7_AVSD dataset from: https://github/com/PaddlePaddle/models/tree/75e463a22ef6cbd43f47917a62ee43d13a05831e/PaddleNLP/Research/Dialogue-PLATO
'''


def load_file(path):
    corpus = []
    with open(path) as f:
        for line in f.readlines():
            corpus.append(line.strip())
    return corpus


def write_file(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue.split('\t')
        if len(se) == 2:
            context, response = se
        elif len(se) == 3:
            knowledge, context, response = se
            context = f'{knowledge} __eou__ {context}'
        else:
            raise Exception('wrong')
        if '|' in response:
            response = response.split('|')[0].strip()
        else:
            response = response.strip()

        utterances = context.split('__eou__')
        utterances = ['<user0> ' + i.strip() if idx % 2 == 0 else '<user1> ' + i.strip() for idx, i in enumerate(utterances)]
        last_speaker = utterances[-1][:7]
        response = '<user1> ' + response if last_speaker == '<user0>' else '<user0> ' + response
        utterances = ' __eou__ '.join(utterances)
        src.append(utterances)
        tgt.append(response)

    src_path = f'{dataset}/src-{mode}.txt'
    tgt_path = f'{dataset}/tgt-{mode}.txt'
    with open(src_path, 'w') as f:
        for i in src:
            f.write(f'{i}\n')

    with open(tgt_path, 'w') as f:
        for i in tgt:
            f.write(f'{i}\n')


if __name__ == "__main__":
    import sys
    dataset, mode = sys.argv[1], sys.argv[2]
    corpus = load_file(f'{dataset}/dial.{mode}')
    write_file(corpus, dataset, mode)
