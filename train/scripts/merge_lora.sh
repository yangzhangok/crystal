#!/bin/bash

MODEL_NAME="${MODEL_NAME:-/home/u1120230285/zhangyang/checkpoints/qwen2.5vl-7b}"
MODEL_PATH="${MODEL_PATH:-/home/u1120230285/zhangyang/crystal/train/output/lora_vision_test/lora_random_drop_5_10}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/home/u1120230285/zhangyang/crystal/train/output/lora_merged/lora_random_drop_5_10}"
VISIBLE_CUDA_DEVICES="${VISIBLE_CUDA_DEVICES:-0}"   

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES"

python src/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_NAME"  \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization \
