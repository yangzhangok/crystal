torchrun --nproc-per-node=4 --master-port=29505 run.py --data BLINK CV-Bench-2D CV-Bench-3D VStarBench MMVP MMStar HRBench4K HRBench8K HallusionBench TextVQA_VAL --model lora_LIVR --verbose


scancel 69280