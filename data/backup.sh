dataset=$1

echo "[!] backup dataset $dataset"

mkdir ${dataset}_backup
cp $dataset/*.txt ${dataset}_backup
tar -czvf $dataset.tgz ${dataset}_backup
