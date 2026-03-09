#!/usr/bin/env bash
set -e
set -o pipefail

echo ">>> Using current Python environment: $(which python)"
echo ">>> Upgrading pip"
python -m pip install --upgrade pip

echo ">>> Installing VLMEvalKit dependencies"
pip install -e .
pip install datasets scikit-learn
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
echo ">>> Installation finished."
