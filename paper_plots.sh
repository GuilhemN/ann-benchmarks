DATASETS=(
    "movielens10m-jaccard"
    "gowalla"
    "dblp"
)

for ((i = 0; i < ${#DATASETS[@]}; i++))
do
    python3 plot.py --dataset ${DATASETS[$i]}-jaccard -x qps -y k-nn --output results/paper/${DATASETS[$i]}-recall.png
    python3 plot.py --dataset ${DATASETS[$i]}-jaccard -x qps -y quality --output results/paper/${DATASETS[$i]}-quality.png
done
