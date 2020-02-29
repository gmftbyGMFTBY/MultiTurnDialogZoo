#!/bin/bash
# RUN EVAL AT A TIME
success=()
datasets=(dailydialog)
models=(VHRED DSHRED)
for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        echo "===== EVAL $dataset - $model EVAL =====" 
        ./run.sh eval $dataset $model > ./processed/$dataset/$model/final_result.txt
        if [ $? -ne 0 ]; then
            echo "===== FAILED ====="
        else
            echo "===== SUCCESS ====="
            success+=( "$dataset-$model" )
        fi
    done
done

# SHOW THE SUCCESS LIST
echo "===== SUCCESS LIST ====="
for item in ${success[@]}
do
    echo "===== $item eval success ====="
done
echo "[!] all files are written into the corresponding file final_result.txt"