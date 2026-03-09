<!-- Modified From VLMEvalKit -->
# Quickstart

## Step 1. Installation & Setup essential keys

**Installation.**

```bash
cd VLMEvalKit
bash install_eval.sh
```

**Setup Keys.**

To infer with API models (GPT-4v, Gemini-Pro-V, etc.) or use LLM APIs as the **judge or choice extractor**, you need to first setup API keys. VLMEvalKit will use an judge **LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**
- You can place the required keys in `$VLMEvalKit/.env` or directly set them as the environment variable. If you choose to create a `.env` file, its content will look like:

  ```bash
  # The .env file, place it under $VLMEvalKit
  # API Keys of Proprietary VLMs
  # QwenVL APIs
  DASHSCOPE_API_KEY=
  # Gemini w. Google Cloud Backends
  GOOGLE_API_KEY=
  # OpenAI API
  OPENAI_API_KEY=
  OPENAI_API_BASE=
  # StepAI API
  STEPAI_API_KEY=
  # REKA API
  REKA_API_KEY=
  # GLMV API
  GLMV_API_KEY=
  # CongRong API
  CW_API_BASE=
  CW_API_KEY=
  # SenseNova API
  SENSENOVA_API_KEY=
  # Hunyuan-Vision API
  HUNYUAN_SECRET_KEY=
  HUNYUAN_SECRET_ID=
  # LMDeploy API
  LMDEPLOY_API_BASE=
  # You can also set a proxy for calling api models during the evaluation stage
  EVAL_PROXY=
  ```

- Fill the blanks with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.

## Step 2. Evaluation

We use `run.py` for evaluation. To use the script, you can use `$VLMEvalKit/run.py` or create a soft-link of the script (to use the script anywhere):

**Arguments**

- `--data (list[str])`: Set the dataset names that are supported in VLMEvalKit (names can be found in the codebase README).
- `--model (list[str])`: Set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).
- `--mode (str, default to 'all', choices are ['all', 'infer'])`: When `mode` set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference.
- `--api-nproc (int, default to 4)`: The number of threads for OpenAI API calling.
- `--work-dir (str, default to '.')`: The directory to save evaluation results.

**Command for Evaluating Image Benchmarks **

You can run the script with `python` or `torchrun`:

```bash
# To evaluate CoVT with segmenation tokens, depth tokens and DINO tokens on some vision-centric benchmarks, run:
python run.py --data BLINK --model CoVT-7B-seg_depth_dino --verbose
# Or you can change the model (CoVT-7B-seg, CoVT-7B-depth, CoVT-7B-seg_depth_dino_edge)
python run.py --data BLINK --model CoVT-7B-seg CoVT-7B-depth CoVT-7B-seg_depth_dino_edge --verbose
# And evaluate them on different benchmarks
python run.py --data VStarBench CV-Bench-2D CV-Bench-3D --model CoVT-7B-seg_depth_dino --verbose

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# However, that is only suitable for VLMs that consume small amounts of GPU memory.
torchrun --nproc-per-node=8 run.py --data BLINK --model CoVT-7B-seg_depth_dino --verbose
```

After running, you can find the output results under `VLMEvalKit/outputs`.

You can find more detailed instruction [here](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md).