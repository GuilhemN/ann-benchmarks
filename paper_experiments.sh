#!/bin/bash

DATASETS=(
    "movielens10m"
    "gowalla"
    "dblp"
)

for count in 15 30 60
do
    for ((i = 0; i < ${#DATASETS[@]}; i++))
    do
        python3 run.py --algorithm "hnsw(nmslib)*" --dataset ${DATASETS[$i]}-jaccard --parallelism 5 -k $count
        python3 plot.py --dataset ${DATASETS[$i]}-jaccard --count $count -x qps -y k-nn --output results/paper/${DATASETS[$i]}-k${count}-recall.png
        python3 plot.py --dataset ${DATASETS[$i]}-jaccard --count $count -x qps -y quality --output results/paper/${DATASETS[$i]}-k${count}-quality.png
    done
done