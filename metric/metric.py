from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse
import codecs
import numpy as np
import math
from bert_score import score
from rouge import Rouge


def cal_BLEU(refer, candidate, ngram=1):
    smoothie = SmoothingFunction().method2
    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33, 0.33, 0.33, 0)
    elif ngram == 4:
        weight = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(refer, candidate, 
                         weights=weight, 
                         smoothing_function=smoothie)


def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity


def cal_ROUGE(refer, candidate):
    if len(candidate) == 0:
        candidate = ['<unk>']
    elif len(candidate) == 1:
        candidate.append('<unk>')
    if len(refer) == 0:
        refer = ['<unk>']
    elif len(refer) == 1:
        refer.append('<unk>')
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    return scores[0]['rouge-2']['f']


def cal_BERTScore(refer, candidate):
    _, _, bert_scores = score(candidate, refer, 
                              bert="bert-base-uncased", no_idf=True)
    bert_scores = bert_scores.tolist()
    bert_scores = [0.5 if math.isnan(score) else score for score in bert_scores]
    return np.mean(bert_scores)


if __name__ == "__main__":
    pass
