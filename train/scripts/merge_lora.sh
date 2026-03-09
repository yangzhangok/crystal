#!/bin/bash

MODEL_NAME="${MODEL_NAME:-/public/home/h202105865/ldy_workdir/zy/qwen2.5vl-7b}"
MODEL_PATH="${MODEL_PATH:-output/lora_vision_test/lora_direct_FT_4lr_answer_label}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-output/lora_merged/lora_direct_FT_4lr_answer_label}"
VISUAL_MODEL_ID="${VISUAL_MODEL_ID:-[]}" #'sam', 'depth', 'dino'
VISIBLE_CUDA_DEVICES="${VISIBLE_CUDA_DEVICES:-0}"   

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES"

python src/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_NAME"  \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization \
    --anchor-model-id "$VISUAL_MODEL_ID"
