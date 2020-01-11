import sys
import os

'''
BLEU calcualted by the nltk is questionable for open-domain dialogue systems.
Some bad cases may be very long and dublicated, so the brifely penalty is harmful.
Try to calculate the BLEU score by the multi-bleu.perl script (moses).

Called by the train.py in the root folder
'''

dataset, model = sys.argv[1], sys.argv[2]
if dataset not in ['cornell', 'dailydialog', 'ubuntu', 'zh50']:
    raise Exception(f'[!] dataset must in cornell, dailydialog, ubuntu, zh50. But got {dataset}')
if model not in ['Seq2Seq', 'Transformer', 'HRED', 'WSeq', 'DSHRED', 'MReCoSa', 'MTGCN', 'GatedGCN', 'GatedGCN-no-sequential', 'GatedGCN-no-role', 'GatedGCN-no-correlation']:
    raise Exception(f'[!] model must in Seq2Seq, Transformer, HRED, DSHRED, WSeq, MReCoSa, MTGCN, GatedGCN. But got {model}')


pred_p = f'./processed/{dataset}/{model}/pred.txt'
ref_p = f'./processed/{dataset}/{model}/reference.txt'
tgt_p = f'./processed/{dataset}/{model}/output.txt'

with open(pred_p) as  f:
    tgt, ref = [], []
    for idx, line in enumerate(f.readlines()):
        if idx % 4 == 1:
            ref.append(line.strip()[7:].replace('user1', '').replace('user0', '').strip())
        elif idx % 4 == 2:
            tgt.append(line.strip()[7:].replace('user1', '').replace('user0', '').strip())

assert len(tgt) == len(ref)
with open(ref_p, 'w') as f:
    for i in ref:
        f.write(f'{i}\n')
with open(tgt_p, 'w') as f:
    for i in tgt:
        f.write(f'{i}\n')

# lc parameters ignore the case
os.system(f'./metric/multi-bleu.perl -lc {ref_p} < {tgt_p}')
