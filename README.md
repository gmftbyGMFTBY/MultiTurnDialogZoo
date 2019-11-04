# Multi-turn modeling
Tradtional RNN-based or HRED-based method model the context relationship implictly.
Our motivation is to prove that explicit multi-round context modeling or explicit edge among the context utterances can effectively provide more meaningful information for dialogue generation.
* Seq2Seq
* HRED-based (HRED, WSeq)
* Attention-based (ReCoSa)
* Our proposed model

## Dataset 
1. DailyDialog
2. Cornell

## Metric
1. PPL
2. BLEU-4
3. Distinct-1
4. Distinct-2
5. human annotation

## Requirements
1. Pytorch >= 1.2 (Transformer support & pack_padded update)
2. Python >= 3.6
3. tqdm
4. numpy
5. nltk
6. scipy
7. sklearn
8. [rouge](https://github.com/pltrdy/rouge)
8. GloVe 300 dimension word embedding

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

Train the model (HRED / WSeq / Seq2Seq / Transformer / MReCoSa / ReCoSa) on the dataset (dailydialog / cornell):

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

## Experiment

### 1. Models
* seq2seq: seq2seq with attention
* HRED-attn: hierarchical seq2seq model with attention on context encoder
* WSeq: modified HRED model
* ReCoSa: 2019 ACL state-of-the-art generatice dialogue method
* MTGCN: GatedGCN architecture without the Gated RNN mechanism
* GCNRNN: combine the GCN and RNN hidden state
* GatedGCN: Gated MTGCN

### 2. Automatic evualtion

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
    <td><strong>29.1222</strong></td>
    <td><strong>0.1801</strong></td>
    <td><strong>0.1225</strong></td>
    <td><strong>0.1073<strong></td>
    <td><strong>0.1003<strong></td>
    <td><strong>0.4528<strong></td>
    <td>0.0237</td>
    <td>0.1101</td>
  </tr>
  <tr>
    <td>HRED-attn</td>
    <td>32.4078</td>
    <td>0.1773</td>
    <td>0.1199</td>
    <td>0.1050</td>
    <td>0.0979</td>
    <td>0.4476</td>
    <td>0.0222</td>
    <td>0.1132</td>
  </tr>
  <tr>
    <td>WSeq</td>
    <td>32.8483</td>
    <td>0.1641</td>
    <td>0.1116</td>
    <td>0.0994</td>
    <td>0.0941</td>
    <td>0.0286</td>
    <td>0.0168</td>
    <td>0.0717</td>
  </tr>
  <tr>
    <td>ReCoSa</td>
    <td>31.5414</td>
    <td>0.1618</td>
    <td>0.1113</td>
    <td>0.0986</td>
    <td>0.0930</td>
    <td>0.0323</td>
    <td>0.0176</td>
    <td>0.0764</td>
  </tr>
  <tr>
    <td>MTGCN</td>
    <td>40.0649</td>
    <td>0.1623</td>
    <td>0.1105</td>
    <td>0.0981</td>
    <td>0.0928</td>
    <td>0.4349</td>
    <td><strong>0.0279<strong></td>
    <td>0.1443</td>
  </tr>
  <tr>
    <td>GatedGCN</td>
    <td>40.6785</td>
    <td>0.1633</td>
    <td>0.1098</td>
    <td>0.0964</td>
    <td>0.0904</td>
    <td>0.4347</td>
    <td>0.0267</td>
    <td><strong>0.1676<\strong></td>
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
    <td>32.4078</td>
    <td>0.1350</td>
    <td>0.1001</td>
    <td>0.0934</td>
    <td>0.0924</td>
    <td>0.0145</td>
    <td>0.0162</td>
    <td>0.0698</td>
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
    <td>40.6785</td>
    <td>0.1258</td>
    <td>0.0930</td>
    <td>0.0869</td>
    <td>0.0859</td>
    <td>0.0126</td>
    <td><strong>0.0358<\strong></td>
    <td><strong>0.1462<\strong></td>
  </tr>
</table>

### 2. Layers of the GCNConv (GatedGCN)

#### 2.1 Test the influence of the layers number for the performance. (1, 2, 3, 4, 5 layers)

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
    <td>GCNRNN(1)</td>
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
    <td>GCNRNN(2)</td>
    <td>42.1443</td>
    <td>0.0879</td>
    <td>0.0154</td>
    <td>0.0646</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GCNRNN(3)</td>
    <td>42.8841</td>
    <td>0.0881</td>
    <td>0.0166</td>
    <td>0.0680</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GCNRNN(4)</td>
    <td>44.3583</td>
    <td>0.0887</td>
    <td>0.0125</td>
    <td>0.0502</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GCNRNN(5)</td>
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


#### 2.2 Test the influence of the GatedGCN layers number for the performance. (1, 2, 3, 4, 5 layers)

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


### 3. Graph ablation analyse
1. complete graph
2. w/o user dependency edge
3. w/o sequence dependency edge
4. difference way to construct the graph and the influence for the performance