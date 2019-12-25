import sys
with open(f'{sys.argv[1]}/pred.txt') as  f:
    tgt, ref = [], []
    for idx, line in enumerate(f.readlines()):
        if idx % 4 == 1:
            ref.append(line.strip()[7:].replace('user1', '').replace('user0', '').strip())
        elif idx % 4 == 2:
            tgt.append(line.strip()[7:].replace('user1', '').replace('user0', '').strip())

assert len(tgt) == len(ref)
with open(f'{sys.argv[1]}/reference.txt', 'w') as f:
    for i in ref:
        f.write(f'{i}\n')
with open(f'{sys.argv[1]}/output.txt', 'w') as f:
    for i in tgt:
        f.write(f'{i}\n')
