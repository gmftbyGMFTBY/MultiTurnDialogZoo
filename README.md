# Multi-turn modeling
Tradtional RNN-based or HRED-based method model the context relationship implictly.
Our motivation is to prove that explicit multi-round context modeling or explicit edge among the context utterances can effectively provide more meaningful information for dialogue generation.

## Dataset 
1. DailyDialog dataset
2. Cornell movie

## Metric
1. PPL
2. BLEU-4
3. ROUGE
4. Embedding Average, Vector Extrema, Greedy Maching
5. Distinct-1/2
6. human annotation

## Requirements
1. Pytorch >= 1.2 (Transformer support & pack_padded update)
2. Python >= 3.6
3. tqdm
4. numpy
5. nltk
6. scipy
7. sklearn
8. [rouge](https://github.com/pltrdy/rouge)
8. GloVe 300 dimension word embedding (Create the graph and embedding-based metric)
9. Pytorch_geometric (PyG 1.2)
10. CUDA 9.2 (match with PyG)

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

## How to use
Generate the vocab of the dataset

```python
./run.sh vocab dailydialog 
```

Generate the graph of the dataset

```python
# only MTGCN and GCNRNN need to create the graph
# The average context coverage in the graph: 0.7935/0.7949/0.7794 (train/test/dev) 
./run.sh graph dailydialog MTGCN none 0 
```

Train the model (HRED / WSeq / Seq2Seq / Transformer / MReCoSa) on the dataset (dailydialog / cornell):

```python
# train mode, dataset dailydialog, model HRED, pretrained [bert/none] on 4th GPU
./run.sh train dailydialog HRED bert 4
```

Translate the test dataset to generate the sentences for evaluating:

```python
# translate mode, dataset dialydialog, model HRED, pretrained [bert/none] on 4th GPU
./run.sh translate dailydialog HRED bert 4
```

Evaluate the result of the translated utterances

```python
# get the BLEU and Distinct result of the generated sentences on 4th GPU (BERTScore need it)
./run.sh eval dailydialog HRED none 4
```

Get the curve of all the training checkpoints

```python
./run.sh curve dailydialog MTGCN none 4
```

Pertubate the source test dataset

```python
./run.sh perturbation dailydialog
```

## Experiment

### 1. Models
* __Seq2Seq__: seq2seq with attention
* __HRED-attn__: hierarchical seq2seq model with attention on context encoder
* __WSeq__: modified HRED model with the Cosine attention weight on conversation context
* __ReCoSa__: 2019 ACL state-of-the-art generatice dialogue method, PPL is larger than the ReCoSa paper(ACL 2019) because of the more open dialogue topic (more open, harder to match with the ground-truth)
* __MTGCN__: GCN for context modeling
* __GatedGCN__: Gated GCN for context modeling

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
    <th colspan="8">Cornell</th>
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
        
### 4. Layers of the GCNConv (GatedGCN)
<table border="1" align="center">
  <tr>
    <th rowspan="2">Models</th>
    <th colspan="4">DailyDialog</th>
    <th colspan="4">Cornell</th>
  </tr>
  <tr>
    <td>PPL</td>
    <td>BLEU4</td> 
    <td>Dist-1</td>
    <td>Dist-2</td>
    <td>PPL</td>
    <td>BLEU4</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td>GatedGCN(1)</td>
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
    <td>GatedGCN(2)</td>
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
    <td>GatedGCN(3)</td>
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
    <td>GatedGCN(4)</td>
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
    <td>GatedGCN(5)</td>
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


### 5. Graph ablation analyse
1. complete graph
2. w/o user dependency edge
3. w/o sequence dependency edge
4. difference way to construct the graph and the influence for the performance
        
Note: More edges better performance
        
### 6. PPL Perturbation analyse
More details of this experiment can be found in [ACL 2019 Short paper for context analyse in multi-turn dialogue systems](https://arxiv.org/pdf/1906.01603.pdf).
        
#### 6.1 Dailydialog
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
            
#### 6.2 Cornell
        
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