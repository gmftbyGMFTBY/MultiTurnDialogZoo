#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.19

from metric.metric import * 
import argparse
import gensim
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--model', type=str, default='HRED', help='model name')
    parser.add_argument('--file', type=str, default=None, help='result file')
    args = parser.parse_args()

    with open(args.file) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            # line = line.lower()
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())

    assert len(ref) == len(tgt)

    # BLEU and ROUGE
    rouge_sum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, counter = 0, 0, 0, 0, 0, 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        rouge_sum += cal_ROUGE(rr, cc)
        # bleu1_sum += cal_BLEU([rr], cc, ngram=1)
        # bleu2_sum += cal_BLEU([rr], cc, ngram=2)
        # bleu3_sum += cal_BLEU([rr], cc, ngram=3)
        # bleu4_sum += cal_BLEU([rr], cc, ngram=4)
        counter += 1
        
    refs, tgts = [' '.join(i) for i in ref], [' '.join(i) for i in tgt]
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_BLEU(refs, tgts)

    # Distinct-1, Distinct-2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    rdistinct_1, rdistinct_2 = cal_Distinct(references)

    # BERTScore < 512 for bert
    # Fuck BERTScore, slow as the snail, fuck it
    # ref = [' '.join(i) for i in ref]
    # tgt = [' '.join(i) for i in tgt]
    # bert_scores = cal_BERTScore(ref, tgt)
    
    # Embedding-based metric: Embedding Average (EA), Vector Extrema (VX), Greedy Matching (GM)
    # load the dict
    dic = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    print('[!] load the GoogleNews 300 word2vector by gensim over')
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    no_save = 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        ea_sum_ = cal_embedding_average(rr, cc, dic)
        vx_sum_ = cal_vector_extrema(rr, cc, dic)
        gm_sum += cal_greedy_matching_matrix(rr, cc, dic)
        # gm_sum += cal_greedy_matching(rr, cc, dic)
        if ea_sum_ != 1 and vx_sum_ != 1:
            ea_sum += ea_sum_
            vx_sum += vx_sum_
            counterp += 1
        else:
            no_save += 1

    print(f'[!] It should be noted that UNK ratio for embedding-based: {round(no_save / (no_save + counterp), 4)}')
    print(f'Model {args.model} Result')
    print(f'BLEU-1: {round(bleu1_sum, 4)}')
    print(f'BLEU-2: {round(bleu2_sum, 4)}')
    print(f'BLEU-3: {round(bleu3_sum, 4)}')
    print(f'BLEU-4: {round(bleu4_sum, 4)}')
    print(f'ROUGE: {round(rouge_sum / counter, 4)}')
    print(f'Distinct-1: {round(distinct_1, 4)}; Distinct-2: {round(distinct_2, 4)}')
    print(f'Ref distinct-1: {round(rdistinct_1, 4)}; Ref distinct-2: {round(rdistinct_2, 4)}')
    print(f'EA: {round(ea_sum / counterp, 4)}')
    print(f'VX: {round(vx_sum / counterp, 4)}')
    print(f'GM: {round(gm_sum / counterp, 4)}')
