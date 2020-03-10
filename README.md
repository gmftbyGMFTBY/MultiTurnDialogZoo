# Multi-turn Dialog Zoo
A batch of ready-to-use multi-turn or single-turn dialogue baselines.

Welcome PRs and issues.

## TODO
* Memory Network
* HVMN
* Pure Transformer (in development, poor performance)
* GAN-based multi-turn dialogue generation
* RL-based fine-tuning dialogue models
* Fix the architecture of the decoder (add the context vector $c$ and last token embedding $y_{t-1}$ for predicting $y_t$)

## Dataset 
The preprocess script for these datasets can be found under `data/data_process` folder.
1. DailyDialog dataset
2. Ubuntu corpus
3. EmpChat
4. DSTC7-AVSD
5. PersonaChat

## Metric
1. PPL: test perplexity
2. BLEU(1-4): nlg-eval version or multi-bleu.perl or nltk
3. ROUGE-2
4. Embedding-based metrics: Average, Extrema, Greedy (slow and optional)
5. Distinct-1/2
6. BERTScore
7. [BERT-RUBER](https://github.com/gmftbyGMFTBY/RUBER-and-Bert-RUBER) 

## Requirements
1. Pytorch 1.2+ (Transformer support & pack_padded update)
2. Python 3.6.1+
3. tqdm
4. numpy
5. nltk 3.4+
6. scipy
7. sklearn (optional)
8. [rouge](https://github.com/pltrdy/rouge)
8. **GoogleNews word2vec** or **glove 300 word2vec** (optional)
9. pytorch_geometric (PyG 1.2) (optional)
10. cuda 9.2 (match with PyG) (optional)
11. tensorboard (for PyTorch 1.2+)
12. perl (for running the multi-bleu.perl script)

## Dataset format
Three multi-turn open-domain dialogue dataset (Dailydialog, DSTC7_AVSD, PersonaChat) can be obtained by this [link](https://github.com/PaddlePaddle/models/tree/75e463a22ef6cbd43f47917a62ee43d13a05831e/PaddleNLP/Research/Dialogue-PLATO)

Each dataset contains 6 files
* src-train.txt
* tgt-train.txt
* src-dev.txt
* tgt-dev.txt
* src-test.txt
* tgt-test.txt

In all the files, one line contain only one dialogue context (src) or the dialogue response (tgt).
More details can be found in the example files.
In order to create the graph, each sentence must begin with the 
special tokens `<user0>` and `<user1>` which denote the speaker.
The `__eou__` is used to separate the multiple sentences in the conversation context.
More details can be found in the small data case.

## How to use

* Model names: `Seq2Seq, SeqSeq_MHA, HRED, HRED_RA, VHRED, WSeq, WSeq_RA, DSHRED, DSHRED_RA, HRAN, MReCoSa, MReCoSa_RA`
* Dataset names: `daildydialog, ubuntu, dstc7, personachat, empchat`

### 0. Ready
Before running the following commands, make sure the essential folders are created:
```bash
mkdir -p processed/$DATASET
mkdir -p data/$DATASET
mkdir -p tblogs/$DATASET
mkdir -p ckpt/$DATASET
```

Variable `DATASET` contains the name of the dataset that you want to process


### 1. Generate the vocab of the dataset

```bash
# default 25000 words
./run.sh vocab <dataset>
```

### 2. Generate the graph of the dataset (optional)

```bash
# only MTGCN and GatedGCN need to create the graph
# zh or en
./run.sh graph <dataset> <zh/en> <cuda>
```

### 3. Check the information about the preprocessed dataset

Show the length of the utterances, turns of the multi-turn setting and so on.
```bash
./run.sh stat <dataset>
```

### 4. Train N-gram LM (Discard)
Train the N-gram Language Model by NLTK (Lidstone with 0.5 gamma, default n-gram is 3):

```bash
# train the N-gram Language model by NLTK
./run.sh lm <dataset>
```

### 5. Train the model on corresponding dataset

```bash
./run.sh train <dataset> <model> <cuda>
```

### 6. Translate the test dataset:

```bash
# translate mode, dataset dialydialog, model HRED on 4th GPU
./run.sh translate <dataset> <model> <cuda>
```

Translate a batch of models
```bash
# rewrite the models and datasets you want to translate
./run_batch_translate.sh <cuda>
```

### 7. Evaluate the result of the translated utterances

```bash
# get the BLEU and Distinct result of the generated sentences on 4th GPU (BERTScore need it)
./run.sh eval <dataset> <model> <cuda>
```

Evaluate a batch of models
```bash
# the performance are redirected into the file `./processed/<dataset>/<model>/final_result.txt`
./run_batch_eval.sh <cuda>
```

### 8. Get the curve of all the training checkpoints (discard, tensorboard is all you need)

```bash
# draw the performance curve, but actually, you can get all the information from the tensorboard
./run.sh curve <dataset> <model> <cuda>
```

### 9. Perturbate the source test dataset

Refer to the paper: `Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study`

```bash
# 10 mode for perturbation
./run.sh perturbation <dataset> <zh/en>
```

## Ready-to-use Models
* Seq2Seq-attn: `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation`
* Seq2Seq-MHA: `Attention is All you Need`. It should be noted that vanilla Transformer is very hard to obtain the good performance on these datasets. In order to make sure the stable performance, i leverage the multi-head self-attention (1 layer, you can change it) on the RNN-based Seq2Seq-attn, which shows the better performance.
* HRED: `Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models`. Enhanced HRED with the utterance-level attention.
* HRED-WA: Building the word-level attention on HRED model.
* WSeq: `How to Make Context More Useful? An Empirical Study on Context-Aware Neural Conversational Models`
* WSeq-WA: Building the word-level attention on WSeq model.
* VHRED: `A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues`, without BOW loss (still in development, welcome PR)
* DSHRED: `Context-Sensitive Generation of Open-Domain Conversational Responses`, dynamic and static attention mechanism on HRED
* DSHRED-WA: Building the word-level attention on DSHRED
* ReCoSa: `ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation`. It should be noted that this implementation here is a little bit different from the original codes, but more powerful and practical (3 layer multi-head self-attention but only 1 layer in the orginal paper).
* ReCoSa-WA: Building word-level attention on ReCoSa
* HRAN: `Hierarchical Recurrent Attention Network for Response Generation`, actually it's the same as the HRED with word-level attention mechanism.


## FAQ
