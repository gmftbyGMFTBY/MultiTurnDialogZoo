#!/bin/bash
# RUN TRANSLATE AT A TIME
# ./run_batch_translate.sh <cuda>
# The results file are written into the corresponding `pred.txt`
cuda=$1
success=()
datasets=(dailydialog)
models=(WSeq VHRED DSHRED)
for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        echo "===== TRANSLATE $dataset - $model TRANSLATE =====" 
        ./run.sh translate $dataset $model $cuda
        if [ $? -ne 0 ]; then
            echo "===== FAILED ====="
        else
            echo "===== SUCCESS ====="
            success+=( "$dataset-$model" )
        fi
    done
done

# SHOW THE SUCCESS LIST
for item in ${success[@]}
do
    echo "===== $item translate success ====="
done