torchrun --nproc-per-node=4 --master-port=29505 run.py --data BLINK CV-Bench-2D CV-Bench-3D VStarBench MMVP MMStar HRBench4K HRBench8K HallusionBench RealWorldQA --model CoVT-7B-seg_depth_dino_lora_stage234_KL_self_supervise_doublece_again_8k CoVT-7B-seg_depth_dino_lora_stage234_KL_self_supervise_doublece_randomdrop_again_8k --verbose


scancel 69109