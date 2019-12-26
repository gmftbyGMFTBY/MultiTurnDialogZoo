#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.24

mode=$1     # graph/stat/train/translate/eval/curve
dataset=$2
model=$3
pretrained=$4
CUDA=$5

# try catch
if [ ! $model ]; then
    model='none'
    pretrained='none'
    CUDA=0
fi

# hierarchical
if [ $model = 'HRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'WSeq' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'ReCoSa' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'Transformer' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'MReCoSa' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'Seq2Seq' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'MTGCN' ]; then
    hierarchical=1
    graph=1
elif [ $model = 'GCNRNN' ]; then
    hierarchical=1
    graph=1
elif [ $model = 'GatedGCN' ]; then
    hierarchical=1
    graph=1
else
    hierarchical=0
    graph=0
fi

# transformer decode mode
if [ $model = 'ReCoSa' ]; then
    transformer_decode=1
elif [ $model = 'Transformer' ]; then
    transformer_decode=1
elif [ $model = 'MReCoSa' ]; then
    transformer_decode=0
else
    transformer_decode=0
fi

# maxlen and batch_size
if [ $hierarchical == 1 ]; then
    maxlen=50
    batch_size=64
else
    maxlen=200
    batch_size=32
fi

# ========== Ready Perfectly ========== #
echo "========== $mode begin =========="

if [ $mode = 'lm' ]; then
    echo "[!] Begin to train the N-gram Language Model"
    python utils.py \
        --dataset $dataset \
        --mode lm 

elif [ $mode = 'perturbation' ]; then
    echo "[!] Begin to perturbation the source test dataset"
    for i in {1..10}
    do
        python utils.py \
            --mode perturbation \
            --perturbation_in ./data/$dataset/src-test.txt \
            --perturbation_out ./data/$dataset/src-test-perturbation-${i}.txt \
            --perturbation_mode $i
            
        python utils.py \
            --mode graph \
            --src ./data/$dataset/src-test-perturbation-${i}.txt \
            --tgt ./data/$dataset/tgt-test.txt \
            --src_vocab ./processed/$dataset/iptvocab.pkl \
            --tgt_vocab ./processed/$dataset/optvocab.pkl \
            --graph ./processed/$dataset/test-graph-perturbation-${i}.pkl \
            --threshold 0.4 \
            --maxlen $maxlen \
            --no-bidir
    done

elif [ $mode = 'vocab' ]; then
    # Generate the src and tgt vocabulary
    echo "[!] Begin to generate the vocab"
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --file ./data/$dataset/src-train.txt ./data/$dataset/src-dev.txt

    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/optvocab.pkl \
        --file ./data/$dataset/tgt-train.txt ./data/$dataset/tgt-dev.txt
        
elif [ $mode = 'stat' ]; then
    # analyse the graph information in the dataset
    echo "[!] analyze the graph coverage information"
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/$model/train-graph.pkl \
         --hops 3 
         
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/$model/test-graph.pkl \
         --hops 3
         
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/$model/dev-graph.pkl \
         --hops 3
        
elif [ $mode = 'graph' ]; then
    # generate the graph file for the MTGCN model
    
    python utils.py \
        --mode graph \
        --src ./data/$dataset/src-train.txt \
        --tgt ./data/$dataset/tgt-train.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --graph ./processed/$dataset/train-graph.pkl \
        --threshold 0.4 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3
    
    python utils.py \
        --mode graph \
        --src ./data/$dataset/src-test.txt \
        --tgt ./data/$dataset/tgt-test.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --graph ./processed/$dataset/test-graph.pkl \
        --threshold 0.4 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3

    python utils.py \
        --mode graph \
        --src ./data/$dataset/src-dev.txt \
        --tgt ./data/$dataset/tgt-dev.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --graph ./processed/$dataset/dev-graph.pkl \
        --threshold 0.4 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3
        
elif [ $mode = 'train' ]; then
    # cp -r ./ckpt/$dataset/$model ./bak/ckpt    # too big, stop back up it
    rm -rf ./ckpt/$dataset/$model
    mkdir -p ./ckpt/$dataset/$model
    rm ./processed/$dataset/$model/ppl.txt
    cp -r tblogs/$dataset/ ./bak/tblogs
    rm tblogs/$dataset/$model/*
    
    echo "[!] back up finished"

    # pretrained embedding
    if [ $pretrained = 'bert' ]; then
        echo "[!] Begin to generate the bert embedding"
        python utils.py \
            --mode pretrained \
            --vocab ./processed/$dataset/$model/iptvocab.pkl \
            --pretrained ./processed/$dataset/$model/ipt_bert_embedding.pkl
        echo "[!] End to generate the src bert embedding"

        python utils.py \
            --mode pretrained \
            --vocab ./processed/$dataset/$model/optvocab.pkl \
            --pretrained ./processed/$dataset/$model/opt_bert_embedding.pkl
        echo "[!] End to generate the tgt bert embedding"
        embed_size=768
    else
        echo "[!] Donot use the pretrained embedding"
        embed_size=300
    fi
    
    # Train
    echo "[!] Begin to train the model"

    CUDA_VISIBLE_DEVICES="$CUDA" python train.py \
        --src_train ./data/$dataset/src-train.txt \
        --tgt_train ./data/$dataset/tgt-train.txt \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --src_dev ./data/$dataset/src-dev.txt \
        --tgt_dev ./data/$dataset/tgt-dev.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --train_graph ./processed/$dataset/train-graph.pkl \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --dev_graph ./processed/$dataset/dev-graph.pkl \
        --pred ./processed/${dataset}/${model}/pred.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --seed 100 \
        --epochs 100 \
        --lr 5e-5 \
        --batch_size $batch_size \
        --weight_decay 1e-6 \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 500 \
        --teach_force 1 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --embed_size $embed_size \
        --patience 5 \
        --dataset $dataset \
        --grad_clip 3 \
        --dropout 0.3 \
        --d_model $embed_size \
        --hierarchical $hierarchical \
        --transformer_decode $transformer_decode \
        --pretrained $pretrained \
        --graph $graph \
        --maxlen $maxlen \
        --position_embed_size 30 \
        --context_threshold 2 \
        --dynamic_tfr 100 \
        --dynamic_tfr_weight 0.05 \
        --dynamic_tfr_counter 5 \
        --dynamic_tfr_threshold 0.3 \
        --bleu perl \
        --contextrnn \
        --no-debug

elif [ $mode = 'translate' ]; then
    
    if [ $pretrained = 'bert' ]; then
        embed_size=768
    else
        embed_size=300
    fi
    
    rm ./processed/$dataset/$model/ppl.txt

    CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --min_threshold 0 \
        --max_threshold 30 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 500 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --seed 20 \
        --embed_size $embed_size \
        --d_model $embed_size \
        --dataset $dataset \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --pred ./processed/${dataset}/${model}/pred.txt \
        --hierarchical $hierarchical \
        --tgt_maxlen 50 \
        --pretrained $pretrained \
        --graph $graph \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --position_embed_size 30 \
        --contextrnn \
        --plus 0 \
        --context_threshold 2 \
        --ppl origin
        
    # 10 perturbation
    for i in {1..10}
    do
        echo "========== running the perturbation $i =========="
        CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
            --src_test ./data/$dataset/src-test-perturbation-${i}.txt \
            --tgt_test ./data/$dataset/tgt-test.txt \
            --min_threshold 0 \
            --max_threshold 30 \
            --batch_size $batch_size \
            --model $model \
            --utter_n_layer 2 \
            --utter_hidden 500 \
            --context_hidden 500 \
            --decoder_hidden 500 \
            --seed 20 \
            --embed_size $embed_size \
            --d_model $embed_size \
            --dataset $dataset \
            --src_vocab ./processed/$dataset/iptvocab.pkl \
            --tgt_vocab ./processed/$dataset/optvocab.pkl \
            --maxlen $maxlen \
            --pred ./processed/${dataset}/${model}/pred.txt \
            --hierarchical $hierarchical \
            --tgt_maxlen 50 \
            --pretrained $pretrained \
            --graph $graph \
            --test_graph ./processed/$dataset/test-graph-perturbation-${i}.pkl \
            --position_embed_size 30 \
            --contextrnn \
            --plus 0 \
            --context_threshold 2 \
            --ppl origin
    done

elif [ $mode = 'eval' ]; then
    CUDA_VISIBLE_DEVICES="$CUDA" python eval.py \
        --model $model \
        --file ./processed/${dataset}/${model}/pred.txt
        
elif [ $mode = 'curve' ]; then
    # do not add the BERTScore evaluate when begin to curve mode
    # evaluation will be too slow

    if [ $pretrained = 'bert' ]; then
        embed_size=768
    else
        embed_size=300
    fi
    
    rm ./processed/${dataset}/${model}/conclusion.txt
    
    # for i in {1..30}
    for i in $(seq 20 5 100)
    do
        # translate
        CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
            --src_test ./data/$dataset/src-test.txt \
            --tgt_test ./data/$dataset/tgt-test.txt \
            --min_threshold $i \
            --max_threshold $i \
            --batch_size $batch_size \
            --model $model \
            --utter_n_layer 2 \
            --utter_hidden 500 \
            --context_hidden 500 \
            --decoder_hidden 500 \
            --seed 20 \
            --embed_size $embed_size \
            --d_model $embed_size \
            --dataset $dataset \
            --src_vocab ./processed/$dataset/iptvocab.pkl \
            --tgt_vocab ./processed/$dataset/optvocab.pkl \
            --maxlen $maxlen \
            --pred ./processed/${dataset}/${model}/pred.txt \
            --hierarchical $hierarchical \
            --tgt_maxlen 50 \
            --pretrained $pretrained \
            --graph $graph \
            --test_graph ./processed/$dataset/test-graph.pkl \
            --position_embed_size 30 \
            --contextrnn \
            --plus 0 \
            --context_threshold 2

        # eval
        echo "========== eval ==========" >> ./processed/${dataset}/${model}/conclusion.txt
        CUDA_VISIBLE_DEVICES="$CUDA" python eval.py \
            --model $model \
            --file ./processed/${dataset}/${model}/pred.txt >> ./processed/${dataset}/${model}/conclusion.txt
            
    done

else
    echo "Wrong mode for running"
fi

echo "========== $mode done =========="
