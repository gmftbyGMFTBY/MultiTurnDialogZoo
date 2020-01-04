import sys
import random

path = sys.argv[1]
src_path, tgt_path = f'src-{path}.txt', f'tgt-{path}.txt'
print(f'modified the file {path}')

src_corpus, tgt_corpus = [], []
with open(src_path) as f:
    for line in f.readlines():
        src_corpus.append(line.strip())

with open(tgt_path) as f:
    for line in f.readlines():
        tgt_corpus.append(line.strip())

assert len(src_corpus) == len(tgt_corpus)
idx = random.sample(range(len(src_corpus)), int(len(src_corpus) / 2) + 10)

src_corpus = [src_corpus[i] for i in idx]
tgt_corpus = [tgt_corpus[i] for i in idx]

with open(src_path, 'w') as f:
    for i in src_corpus:
        f.write(f'{i}\n')

with open(tgt_path, 'w') as f:
    for i in tgt_corpus:
        f.write(f'{i}\n')
