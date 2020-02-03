# Multi-turn modeling
Tradtional RNN-based or HRED-based method model the context relationship implictly.
Our motivation is to prove that explicit multi-round context modeling or explicit edge among the context utterances can effectively provide more meaningful information for dialogue generation.

## Dataset 
1. DailyDialog dataset
2. Ubuntu corpus

## Metric
1. PPL
2. BLEU-1~4
3. ROUGE
4. Embedding Average, Vector Extrema, Greedy Maching   
    __Greedy Matching is also very slow__ (optional)
5. Distinct-1/2
6. human annotation

## Requirements
1. Pytorch 1.2+ (Transformer support & pack_padded update)
2. Python 3.6+
3. tqdm
4. numpy
5. nltk 3.4+
6. scipy
7. sklearn
8. [rouge](https://github.com/pltrdy/rouge)
8. glove 300 dimension word embedding (Create the graph and embedding-based metric)
9. pytorch_geometric (PyG 1.2)
10. cuda 9.2 (match with PyG)
11. tensorboard (for PyTorch 1.2+)
12. perl (for running the multi-bleu.perl script)

## Dataset format
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
Before running the following commands, make sure the essential folders are created:
```bash
# $DATASET contains the name of the dataset that you want to process
mkdir -p processed/$DATASET
mkdir -p data/$DATASET
mkdir -p tblogs/$DATASET
mkdir -p ckpt/$DATASET
```

Generate the vocab of the dataset

```bash
# default 25000 words
./run.sh vocab dailydialog 
```

Generate the graph of the dataset

```bash
# only MTGCN and GatedGCN need to create the graph
# zh or en
./run.sh graph dailydialog zh 0 
```

Train the model (HRED / WSeq / Seq2Seq / Transformer / MReCoSa) on the dataset (dailydialog / Ubuntu):

```bash
# train mode, dataset dailydialog, model HRED on 4th GPU
# max epochs is 100. You can simply stop the training when the performance shown in tensorboard is flatten (ctrl-c). 
./run.sh train dailydialog HRED 4
tensorboard --logdir tblogs
```

Train the N-gram Language Model by NLTK (Lidstone with 0.5 gamma, default n-gram is 3):

```bash
# train the N-gram Language model by NLTK
./run.sh lm dailydialog
```

Translate the test dataset and caulculate the test perplexity by using n-gram model:

```bash
# translate mode, dataset dialydialog, model HRED on 4th GPU
./run.sh translate dailydialog HRED 4
```

Evaluate the result of the translated utterances

```bash
# get the BLEU and Distinct result of the generated sentences on 4th GPU (BERTScore need it)
./run.sh eval dailydialog HRED 4
```

Get the curve of all the training checkpoints

```bash
# draw the performance curve, but actually, you can get all the information from the tensorboard
./run.sh curve dailydialog MTGCN 4
```

Perturbate the source test dataset

```bash
# 10 mode for perturbation
./run.sh perturbation dailydialog
```

## Experiment

### 1. Models
* __HRED-attn__: 2016 hierarchical seq2seq model with attention on context encoder
* __WSeq__: 2017 ACL modified HRED model with the Cosine attention weight on conversation context
* __DSHRED__: 2018 COLING Dynamic and Static attention for HRED
* __ReCoSa__: 2019 ACL state-of-the-art generatice dialogue method, PPL is larger than the ReCoSa paper(ACL 2019) because of the more open dialogue topic (more open, harder to match with the ground-truth)
* __MTGCN__: GCN for context modeling
* __GatedGCN__: Gated GCN for context modeling, in paper we called it **GatedGNN**

### 2. Automatic evaluation

<table border="1" align="center">
  <tr>
    <th rowspan="2">Models</th>
    <th colspan="8">DailyDialog</th>
  </tr>
  <tr>
    <td>PPL</td>
    <td>BLEU1</td> 
    <td>BLEU2</td>
    <td>BLEU3</td>
    <td>BLEU4</td>
    <td>BERTScore</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td>Seq2Seq-attn</td>
    <td><strong></strong></td>
    <td><strong></strong></td>
    <td><strong></strong></td>
    <td><strong><strong></td>
    <td><strong><strong></td>
    <td><strong><strong></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GatedGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

<table border="1" align="center">
  <tr>
    <th rowspan="2">Models</th>
    <th colspan="8">Ubuntu</th>
  </tr>
  <tr>
    <td>PPL</td>
    <td>BLEU1</td> 
    <td>BLEU2</td>
    <td>BLEU3</td>
    <td>BLEU4</td>
    <td>BERTScore</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td>Seq2Seq-attn</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GatedGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

Performance Curve:

        
### 3. Human judgments
        
<table>
  <tr>
    <th rowspan="2">Baselines</th>
    <th colspan="3">GatedGCN</th>
  </tr>
  <tr>
    <td>win%</td>
    <td>tie%</td>
    <td>loss%</td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

### 4. Graph ablation analyse
1. complete graph
2. w/o user dependency edge
3. w/o sequence dependency edge
4. difference way to construct the graph and the influence for the performance
        
Note: More edges better performance
        
### 5. PPL Perturbation analyse
More details of this experiment can be found in [ACL 2019 Short paper for context analyse in multi-turn dialogue systems](https://arxiv.org/pdf/1906.01603.pdf).
        
        
#### 5.1 Dailydialog
<table>
  <tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">Test PPL</th>
    <th colspan="5">Utterance-level</th>
    <th colspan="5">Word-level</th>
  </tr>
  <tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GatedGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>
            
#### 5.2 Ubuntu
        
<table>
  <tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">Test PPL</th>
    <th colspan="5">Utterance-level</th>
    <th colspan="5">Word-level</th>
  </tr>
  <tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GatedGCN</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>