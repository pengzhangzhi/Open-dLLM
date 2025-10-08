#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"

MODEL_PATH="fredzzp/open-dcoder-0.5B"
TEMPERATURE=0.6
STEPS=64
ALG="p2"
BATCH_SIZE=32


 torchrun --nproc_per_node 4 eval_infill.py \
    --model_path "$MODEL_PATH" \
    --task humaneval_infill \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --alg "$ALG" \
    --batch_size "$BATCH_SIZE" \
    --use_ddp


torchrun --nproc_per_node 4 eval_infill.py \
    --model_path "$MODEL_PATH" \
    --task santacoder-fim \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --alg "$ALG" \
    --batch_size "$BATCH_SIZE" \
    --use_ddp
