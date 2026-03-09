#!/usr/bin/env bash
set -e
set -o pipefail

echo ">>> Using current Python environment: $(which python)"
echo ">>> Upgrading pip"
python -m pip install --upgrade pip

# ===== Installing Qwen2.5-VL dependencies =====
echo ">>> Installing Qwen2.5-VL dependencies"
pip install -r requirements.txt

# ===== Installing other visual models dependencies =====
echo ">>> Installing Segment Anything"
pip install git+https://github.com/facebookresearch/segment-anything.git

echo ">>> Installing Depth Anything v2"
pip install -r src/anchors/DepthAnything/requirements.txt

echo ">>> Installation finished."
