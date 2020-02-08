#!/bin/bash
# Author: GMFTBY
# Time: 2020.2.8

mode=$1     # graph/stat/train/translate/eval/curve
dataset=$2
model=$3
CUDA=$4

# try catch
if [ ! $model ]; then
    model='none'
    CUDA=0
fi

# hierarchical
# no-role, no-corrleation, no-sequential for graph ablation study
if [ $model = 'HRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'WSeq' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'MReCoSa' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'DSHRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'Seq2Seq' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'Transformer' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'MTGCN' ]; then
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
if [ $model = 'Transformer' ]; then
    transformer_decode=1
else
    transformer_decode=0
fi

# maxlen and batch_size
# for dailydialog dataset, 50 and 200 is the most appropriate settings
if [ $hierarchical = 1 ]; then
    maxlen=50
    batch_size=128
elif [ $transformer_decode = 1 ]; then
    maxlen=150
    batch_size=48
else
    maxlen=150
    batch_size=64
fi

# ========== Ready Perfectly ========== #
echo "========== $mode begin =========="

if [ $mode = 'lm' ]; then
    echo "[!] Begin to train the N-gram Language Model"
    python utils.py \
        --dataset $dataset \
        --mode lm 
        
elif [ $mode = 'transformer_preprocess' ]; then
    echo "[!] Preprocess the dataset for trainsformer(GPT2) model"
    python utils.py \
        --dataset $dataset \
        --mode preprocess_transformer \
        --ctx 200

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
        --cutoff 20000 \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --file ./data/$dataset/src-train.txt

    python utils.py \
        --mode vocab \
        --cutoff 20000 \
        --vocab ./processed/$dataset/optvocab.pkl \
        --file ./data/$dataset/tgt-train.txt
        
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
        --lang $3 \
        --fully \
        --no-self-loop \
    
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
        --lang $3 \
        --fully \
        --no-self-loop \

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
        --lang $3 \
        --fully \
        --no-self-loop \
        
elif [ $mode = 'train' ]; then
    # cp -r ./ckpt/$dataset/$model ./bak/ckpt    # too big, stop back up it
    rm -rf ./ckpt/$dataset/$model
    mkdir -p ./ckpt/$dataset/$model
    
    if [ ! -d "./processed/$dataset/$model" ]; then
        mkdir -p ./processed/$dataset/$model
    else
        echo "[!] ./processed/$dataset/$model: already exists"
    fi
    if [ ! -f "./processed/$dataset/$model/ppl.txt" ];then
        echo "[!] ./processed/$dataset/$model/ppl.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/ppl.txt
    fi
    
    cp -r tblogs/$dataset/ ./bak/tblogs
    rm tblogs/$dataset/$model/*
    
    echo "[!] back up finished"
    
    # Train
    echo "[!] Begin to train the model"
    
    # set the lr_gamma as 1, means that don't use the learning rate schedule
    # Transformer: lr(threshold) 1e-4, 1e-6 / others: lr(threshold) 1e-4, 1e-6
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
        --seed 30 \
        --epochs 100 \
        --lr 1e-4 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 512 \
        --teach_force 1 \
        --context_hidden 512 \
        --decoder_hidden 512 \
        --embed_size 256 \
        --patience 5 \
        --dataset $dataset \
        --grad_clip 3.0 \
        --dropout 0.3 \
        --d_model 512 \
        --nhead 8 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --dim_feedforward 2048 \
        --hierarchical $hierarchical \
        --transformer_decode $transformer_decode \
        --graph $graph \
        --maxlen $maxlen \
        --position_embed_size 30 \
        --context_threshold 2 \
        --dynamic_tfr 15 \
        --dynamic_tfr_weight 0.0 \
        --dynamic_tfr_counter 10 \
        --dynamic_tfr_threshold 1.0 \
        --bleu nltk \
        --contextrnn \
        --no-debug \
        --lr_mini 1e-6 \
        --lr_gamma 0.5 \
        --warmup_step 4000 \

elif [ $mode = 'translate' ]; then
    rm ./processed/$dataset/$model/ppl.txt

    CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --min_threshold 0 \
        --max_threshold 63 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 500 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --seed 20 \
        --embed_size 500 \
        --d_model 500 \
        --dataset $dataset \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --pred ./processed/${dataset}/${model}/pred.txt \
        --hierarchical $hierarchical \
        --tgt_maxlen 50 \
        --graph $graph \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --position_embed_size 30 \
        --contextrnn \
        --plus 0 \
        --context_threshold 2 \
        --ppl origin
        
    exit    # comment this line for ppl perturbation test, or only translate the test dataset 
    # 10 perturbation
    for i in {1..10}
    do
        echo "========== running the perturbation $i =========="
        CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
            --src_test ./data/$dataset/src-test-perturbation-${i}.txt \
            --tgt_test ./data/$dataset/tgt-test.txt \
            --min_threshold 0 \
            --max_threshold 63 \
            --batch_size $batch_size \
            --model $model \
            --utter_n_layer 2 \
            --utter_hidden 500 \
            --context_hidden 500 \
            --decoder_hidden 500 \
            --seed 20 \
            --embed_size 500 \
            --d_model 300 \
            --dataset $dataset \
            --src_vocab ./processed/$dataset/iptvocab.pkl \
            --tgt_vocab ./processed/$dataset/optvocab.pkl \
            --maxlen $maxlen \
            --pred ./processed/${dataset}/${model}/pred.txt \
            --hierarchical $hierarchical \
            --tgt_maxlen 50 \
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
            --embed_size 300 \
            --d_model 500 \
            --dataset $dataset \
            --src_vocab ./processed/$dataset/iptvocab.pkl \
            --tgt_vocab ./processed/$dataset/optvocab.pkl \
            --maxlen $maxlen \
            --pred ./processed/${dataset}/${model}/pred.txt \
            --hierarchical $hierarchical \
            --tgt_maxlen 50 \
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