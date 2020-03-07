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

# running mode (hierarchical/graph):
# no-hierarchical: Seq2Seq; hierarchical: HRED, VHRED, WSeq, ...
# graph: MTGAT, MTGCN; no-graph: Seq2Seq, HRED, VHRED, WSeq, ...
if [ $model = 'HRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'HRAN' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'HRAN-ablation' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'VHRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'KgCVAE' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'WSeq' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'WSeq_RA' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'MReCoSa' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'MReCoSa_RA' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'DSHRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'DSHRED_RA' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'Seq2Seq' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'Seq2Seq_MHA' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'Transformer' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'MTGCN' ]; then
    hierarchical=1
    graph=1
elif [ $model = 'MTGAT' ]; then
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
# for dailydialog dataset, 20 and 150 is the most appropriate settings
if [ $hierarchical = 1 ]; then
    maxlen=50
    tgtmaxlen=30
    batch_size=64
elif [ $transformer_decode = 1 ]; then
    maxlen=200
    tgtmaxlen=25
    batch_size=64
else
    maxlen=150
    tgtmaxlen=25
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
    # run.sh perturbation dailydialog en 
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
            --maxlen $maxlen \
            --no-bidir \
            --threshold 0.8 \
            --lang $3 \
            --no-fully \
            --no-self-loop
    done

elif [ $mode = 'vocab' ]; then
    # Generate the src and tgt vocabulary
    echo "[!] Begin to generate the vocab"
    
    if [ ! -d "./processed/$dataset" ]; then
        mkdir -p ./processed/$dataset
        echo "[!] cannot find the folder, create ./processed/$dataset"
    else
        echo "[!] ./processed/$dataset: already exists"
    fi
    
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --file ./data/$dataset/src-train.txt

    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/optvocab.pkl \
        --file ./data/$dataset/tgt-train.txt
        
    # generate the whole vocab for VHRED and KgCVAE (Variational model)
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/vocab.pkl \
        --file ./data/$dataset/tgt-train.txt ./data/$dataset/src-train.txt
        
elif [ $mode = 'stat' ]; then
    # analyse the graph information in the dataset
    echo "[!] analyze the graph coverage information"
    echo "[!] train information:"
    python utils.py \
         --mode stat \
         --dataset $dataset \
         --hops 3 \
         --split train
         
    echo "[!] test information"
    python utils.py \
         --mode stat \
         --dataset $dataset \
         --split test \
         --hops 3
         
    echo "[!] dev information"
    python utils.py \
         --mode stat \
         --dataset $dataset \
         --split dev \
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
        --threshold 0.8 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3 \
        --no-fully \
        --no-self-loop \
    
    python utils.py \
        --mode graph \
        --src ./data/$dataset/src-test.txt \
        --tgt ./data/$dataset/tgt-test.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --graph ./processed/$dataset/test-graph.pkl \
        --threshold 0.8 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3 \
        --no-fully \
        --no-self-loop \

    python utils.py \
        --mode graph \
        --src ./data/$dataset/src-dev.txt \
        --tgt ./data/$dataset/tgt-dev.txt \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --graph ./processed/$dataset/dev-graph.pkl \
        --threshold 0.8 \
        --maxlen $maxlen \
        --no-bidir \
        --lang $3 \
        --no-fully \
        --no-self-loop \
        
elif [ $mode = 'train' ]; then
    # cp -r ./ckpt/$dataset/$model ./bak/ckpt    # too big, stop back up it
    rm -rf ./ckpt/$dataset/$model
    mkdir -p ./ckpt/$dataset/$model
    
    # create the training folder
    if [ ! -d "./processed/$dataset/$model" ]; then
        mkdir -p ./processed/$dataset/$model
    else
        echo "[!] ./processed/$dataset/$model: already exists"
    fi
    
    # delete traninglog.txt
    if [ ! -f "./processed/$dataset/$model/trainlog.txt" ]; then
        echo "[!] ./processed/$dataset/$model/trainlog.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/trainlog.txt
    fi
    
    # delete metadata.txt
    if [ ! -f "./processed/$dataset/$model/metadata.txt" ]; then
        echo "[!] ./processed/$dataset/$model/metadata.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/metadata.txt
    fi
    
    cp -r tblogs/$dataset/ ./bak/tblogs
    rm tblogs/$dataset/$model/*
    
    # Because of the posterior, the Variational models need to bind the src and tgt vocabulary
    if [[ $model = 'VHRED' || $model = 'KgCVAE' ]]; then
        echo "[!] VHRED or KgCVAE, src vocab == tgt vocab"
        src_vocab="./processed/$dataset/vocab.pkl"
        tgt_vocab="./processed/$dataset/vocab.pkl"
    else
        src_vocab="./processed/$dataset/iptvocab.pkl"
        tgt_vocab="./processed/$dataset/optvocab.pkl"
    fi
    
    # dropout for transformer
    if [ $model = 'Transformer' ]; then
        # other repo set the 0.1 as the dropout ratio, remain it
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    else
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    fi
    
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
        --src_vocab $src_vocab \
        --tgt_vocab $tgt_vocab \
        --train_graph ./processed/$dataset/train-graph.pkl \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --dev_graph ./processed/$dataset/dev-graph.pkl \
        --pred ./processed/${dataset}/${model}/pure-pred.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --seed 30 \
        --epochs 100 \
        --lr $lr \
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
        --dropout $dropout \
        --d_model 512 \
        --nhead 4 \
        --num_encoder_layers 8 \
        --num_decoder_layers 8 \
        --dim_feedforward 2048 \
        --hierarchical $hierarchical \
        --transformer_decode $transformer_decode \
        --graph $graph \
        --maxlen $maxlen \
        --tgt_maxlen $tgtmaxlen \
        --position_embed_size 30 \
        --context_threshold 2 \
        --dynamic_tfr 15 \
        --dynamic_tfr_weight 0.0 \
        --dynamic_tfr_counter 10 \
        --dynamic_tfr_threshold 1.0 \
        --bleu nltk \
        --contextrnn \
        --no-debug \
        --lr_mini $lr_mini \
        --lr_gamma 0.5 \
        --warmup_step 4000 \
        --gat_heads 8 \

elif [ $mode = 'translate' ]; then
    rm ./processed/$dataset/$model/pertub-ppl.txt
    rm ./processed/$dataset/$model/pred.txt
    
    if [ $model = 'Transformer' ]; then
        # other repo set the 0.1 as the dropout ratio, remain it
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    else
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    fi
    
    if [[ $model = 'VHRED' || $model = 'KgCVAE' ]]; then
        echo "[!] VHRED or KgCVAE, src vocab == tgt vocab"
        src_vocab="./processed/$dataset/vocab.pkl"
        tgt_vocab="./processed/$dataset/vocab.pkl"
    else
        src_vocab="./processed/$dataset/iptvocab.pkl"
        tgt_vocab="./processed/$dataset/optvocab.pkl"
    fi

    CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 512 \
        --context_hidden 512 \
        --decoder_hidden 512 \
        --seed 30 \
        --dropout $dropout \
        --embed_size 256 \
        --d_model 512 \
        --nhead 4 \
        --num_encoder_layers 8 \
        --num_decoder_layers 8 \
        --dim_feedforward 2048 \
        --dataset $dataset \
        --src_vocab $src_vocab \
        --tgt_vocab $tgt_vocab \
        --maxlen $maxlen \
        --pred ./processed/${dataset}/${model}/pure-pred.txt \
        --hierarchical $hierarchical \
        --tgt_maxlen $tgtmaxlen \
        --graph $graph \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --position_embed_size 30 \
        --contextrnn \
        --plus 0 \
        --context_threshold 2 \
        --ppl origin \
        --gat_heads 8 \
        --teach_force 1
        
    # exit    # comment this line for ppl perturbation test, or only translate the test dataset 
    # 10 perturbation
    for i in {1..10}
    do
        echo "========== running the perturbation $i =========="
        CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
            --src_test ./data/$dataset/src-test-perturbation-${i}.txt \
            --tgt_test ./data/$dataset/tgt-test.txt \
            --min_threshold 0 \
            --max_threshold 100 \
            --batch_size $batch_size \
            --model $model \
            --utter_n_layer 2 \
            --utter_hidden 512 \
            --context_hidden 512 \
            --decoder_hidden 512 \
            --seed 30 \
            --dropout $dropout \
            --embed_size 256 \
            --d_model 512 \
            --nhead 4 \
            --num_encoder_layers 8 \
            --num_decoder_layers 8 \
            --dim_feedforward 2048 \
            --dataset $dataset \
            --src_vocab $src_vocab \
            --tgt_vocab $tgt_vocab \
            --maxlen $maxlen \
            --pred ./processed/${dataset}/${model}/perturbation-${i}-pred.txt \
            --hierarchical $hierarchical \
            --tgt_maxlen $tgtmaxlen \
            --graph $graph \
            --test_graph ./processed/$dataset/test-graph-perturbation-${i}.pkl \
            --position_embed_size 30 \
            --contextrnn \
            --plus 0 \
            --context_threshold 2 \
            --ppl origin \
            --gat_heads 8 \
            --teach_force 1
    done

elif [ $mode = 'eval' ]; then
    # before this mode, make sure you run the translate mode to generate the pred.txt file for evaluating.
    CUDA_VISIBLE_DEVICES="$CUDA" python eval.py \
        --model $model \
        --file ./processed/${dataset}/${model}/pure-pred.txt
        
elif [ $mode = 'curve' ]; then
    # this part of codes is useless (tensorboard is all you need)
    # already discard
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
            --pred ./processed/${dataset}/${model}/pure-pred.txt \
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
            --file ./processed/${dataset}/${model}/pure-pred.txt >> ./processed/${dataset}/${model}/conclusion.txt
            
    done

else
    echo "Wrong mode for running"
fi

echo "========== $mode done =========="
