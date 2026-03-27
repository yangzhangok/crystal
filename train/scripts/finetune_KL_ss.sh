set -e

export BNB_CUDA_VERSION=121
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/u1120230285/miniconda3/envs/covt/lib
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_MODEL="/home/u1120230285/zhangyang/checkpoints/qwen2.5vl-7b" 

OUT_DIR_STAGE234="output/lora_vision_test/lora_random_drop_5_10"
FINAL_MERGED_MODEL="output/lora_merged/lora_random_drop_5_10"

DATA_PATH="/home/u1120230285/zhangyang/covt-dataset/data.json"
IMAGE_FOLDER="/home/u1120230285/zhangyang/covt-dataset/images"

# VISUAL_MODEL_ID="[]" #,'dino'

# VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
MODEL_NAME="$BASE_MODEL" \
MODEL_PATH="$BASE_MODEL" \
OUTPUT_DIR="$OUT_DIR_STAGE234" \
RUN_NAME="lora_random_drop_5_10" \
STAGE_0_STEP=0 \
STAGE_1_STEP=0 \
STAGE_2_STEP=3000 \
VQA_ONLY_STAGE=4000 \
VLOC=True \
RANDOM_DROP=True \
NEED_KL=True \
CE_LOSS="double" \
RANDOM_CUT=False \
ATTENTION_DROP=False \
MAX_STEPS=4000 \
DATA_PATH="$DATA_PATH" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
bash scripts/run_self_supervise.sh


echo "==== [4/4] Second merge LoRA (final) ===="

MODEL_NAME="$BASE_MODEL" \
MODEL_PATH="$OUT_DIR_STAGE234" \
SAVE_MODEL_PATH="$FINAL_MERGED_MODEL" \
# VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/merge_lora.sh

echo "==== All processes completed, final model: $FINAL_MERGED_MODEL ===="

# cd ../VLMEvalKit/
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master-port=29501 run.py --data BLINK CV-Bench-2D CV-Bench-3D VStarBench HRBench4K HRBench8K RealWorldQA --model Crystal_drop_5_10 --verbose


