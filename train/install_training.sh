#!/usr/bin/env bash
set -e
set -o pipefail

echo ">>> Using current Python environment: $(which python)"
echo ">>> Upgrading pip"
python -m pip install --upgrade pip

# ===== Installing Qwen2.5-VL dependencies =====
echo ">>> Installing Qwen2.5-VL dependencies"
pip install -r requirements.txt

echo ">>> Installation finished."
