#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2

torchrun --standalone --nnodes=1 --nproc_per_node=2 ./main.py \
  --tag single_node_gpu2 \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --output outputs \
# python -m torch.distributed.launch \
# --nproc_per_node 2 ./main.py \
# --tag single_node_gpu2 \
# --epochs 10 \
# --batch-size 128 \
# --learning-rate 0.001 \
# --output outputs \