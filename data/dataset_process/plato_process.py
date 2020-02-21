import tqdm
import ipdb

'''
Download PersonaChat / Dailydialog / DSTC7_AVSD dataset from: https://github.com/PaddlePaddle/models/tree/75e463a22ef6cbd43f47917a62ee43d13a05831e/PaddleNLP/Research/Dialogue-PLATO

# DATASET: DailyDialog, PersonaChat, DSTC7_AVSD
# MODE: train, test, valid
# after processing, remember to rename the valid files to dev files (src-dev.txt, tgt-dev.txt)
python plato_process.py $DATASET $MODE
'''


def load_file(path):
    corpus = []
    with open(path) as f:
        for line in f.readlines():
            corpus.append(line.strip())
    return corpus


def write_file_dailydialog(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue,split('\t')
        assert len(se) == 2
        context, response = se
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
            
            
def write_file_personachat(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue.split('\t')
        assert len(se) == 3
        knowledge, context, response = se
        ku = knowledge.split('__eou__')
        ku = ['<user0> ' + i.strip() for i in ku]
        ku = ' __eou__ '.join(ku)
        cu = reversed(context.split('__eou__'))
        response = '<user0> ' + response
        
        speaker = '<user1> '
        fcu = []
        for i in cu:
            fcu.append(speaker + i.strip())
            speaker = '<user0> ' if speaker == '<user1> ' else '<user1> '
        fcu = ' __eou__ '.join(list(reversed(fcu)))
        src.append(ku + ' __eou__ ' + fcu)
        tgt.append(response)
        
    src_path = f'{dataset}/src-{mode}.txt'
    tgt_path = f'{dataset}/tgt-{mode}.txt'
    with open(src_path, 'w') as f:
        for i in src:
            f.write(f'{i}\n')

    with open(tgt_path, 'w') as f:
        for i in tgt:
            f.write(f'{i}\n')
            

def write_file_dstc7(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue.split('\t')
        # PersonaChat
        knowledge, context, response = se
        ku = knowledge.split('__eou__')
        ku = ['<user0> ' + i.strip() for i in ku]
        ku = ' __eou__ '.join(ku)
        cu = context.split('__eou__')
        
        speaker = '<user0> '
        fcu = []
        for i in cu:
            fcu.append(speaker + i.strip())
            speaker = '<user0> ' if speaker == '<user1> ' else '<user1> '
        fcu = ' __eou__ '.join(fcu)
            
        # train/dev is different from the test in DSTC7_AVSD dataset
        if '|' in response:
            response = speaker + response.split('|')[0].strip()
        else:
            response = speaker + response.strip()

        src.append(ku + ' __eou__ ' + fcu)
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
    if dataset == 'DailyDialog':
        write_file_dailydialog(corpus, dataset, mode)
    elif dataset == 'PersonaChat':
        write_file_personachat(corpus, dataset, mode)
    elif dataset == 'DSTC7_AVSD':
        write_file_dstc7(corpus, dataset, mode)
    else:
        raise Exception('[!] obtain the wrong dataset {dataset}')
