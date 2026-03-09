#!/bin/bash

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

export MASTER_PORT=22810
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=src:$PYTHONPATH
export WANDB_PROJECT="${WANDB_PROJECT:-Qwen_BASE}"
export WANDB_MODE=offline

RUN_NAME="${RUN_NAME:-test_setup}"

BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
NUM_DEVICES="${NUM_DEVICES:-8}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

STAGE_0_STEP="${STAGE_0_STEP:-0}"
STAGE_1_STEP="${STAGE_1_STEP:-0}"
STAGE_2_STEP="${STAGE_2_STEP:-0}"

VQA_ONLY_STAGE="${VQA_ONLY_STAGE:-0}"
MAX_STEPS="${MAX_STEPS:-4000}"

OUTPUT_DIR="${OUTPUT_DIR:-output/lora_vision_test/default_output_dir}"

DATA_PATH="${DATA_PATH:-dataset/covt_dataset.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-dataset/image_dir}"

VISUAL_MODEL_ID="${VISUAL_MODEL_ID:-['sam', 'depth', 'dino']}"

deepspeed \
    --master_port $MASTER_PORT \
    src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['embed_tokens', 'lm_head', 'dino', 'sam', 'depth', 'SD', 'internvit', 'pidinet', 'siglip', 'metaclip']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1024 * 28 * 28)) \
    --image_resized_width 448 \
    --image_resized_height 448 \
    --learning_rate 2e-4 \
    --projection_layer_lr 4e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 4 \
    --dataloader_num_workers 0 \
    --deepspeed scripts/zero2.json \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --anchor_model_id "$VISUAL_MODEL_ID" \
    --training_stage "full" \
    --stage_0_step $((STAGE_0_STEP * BATCH_PER_DEVICE)) \
    --stage_1_step $((STAGE_1_STEP * BATCH_PER_DEVICE)) \
    --stage_2_step $((STAGE_2_STEP * BATCH_PER_DEVICE)) \
    --vqa_only_stage $VQA_ONLY_STAGE \
    --vloc $VLOC \
    --random_drop $RANDOM_DROP \
    --need_KL $NEED_KL \
    --ce_loss $CE_LOSS \
    --random_cut $RANDOM_CUT \
    --attention_drop $ATTENTION_DROP