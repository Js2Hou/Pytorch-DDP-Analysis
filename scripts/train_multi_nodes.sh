#!/bin/bash
# Node 1: *(IP: 10.106.26.17, and has a free port: 1234)*

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2

torchrun --rdzv_id=1234 --rdzv_backend=c10d --rdzv_endpoint="10.106.26.17:6666" \
    --nnodes=1:4 --nproc_per_node=2 ./main.py \
    --tag multi_node2_gpu4 \
    --epochs 10 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --output outputs \ 

# python -m torch.distributed.launch --nproc_per_node=2 \
#     --nnodes=2 --node_rank=0 --master_addr="10.106.26.17" \
#     --master_port=1234 ./main.py \
#     --tag multi_node2_gpu4 \
#     --epochs 2 \
#     --batch-size 128 \
#     --learning-rate 0.001 \
#     --output outputs \ 