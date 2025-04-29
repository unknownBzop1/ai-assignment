#!/bin/bash

cd /home/work/unknownbzop1/ai-assignment

for batch in 64
do
    for lr in 0.0005 0.001 0.002 0.003 0.005
    do
        python3 training.py --batch_size $batch --epochs 20 --lr $lr
    done
    git add .
    git commit -m date +"%y%m%d-%H%M updates"
    git push origin main
done
