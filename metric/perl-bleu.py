import sys
import os

dataset, model = sys.argv[1], sys.argv[2]
if dataset not in ['cornell', 'dailydialog', 'ubuntu']:
    raise Exception(f'[!] dataset must in cornell, dailydialog, ubuntu. Got {dataset}')
if model not in ['HRED', 'WSeq', 'MReCoSa', 'MTGCN', 'GatedGCN']:
    raise Exception(f'[!] model must in HRED, WSeq, MReCoSa, MTGCN, GatedGCN. Got {model}')


pred_p = f'../processed/{dataset}/{model}/pred.txt'
ref_p = f'../processed/{dataset}/{model}/reference.txt'
tgt_p = f'../processed/{dataset}/{model}/output.txt'

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

os.system(f'./multi-bleu.perl {ref_p} < {tgt_p}')
