#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ./main.py \
  --tag single_gpu \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --output outputs \
  # --log-wandb