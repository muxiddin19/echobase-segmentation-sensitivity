#!/usr/bin/env bash
# Environment for the artifact (tested: Ubuntu 18.04, Python 3.10, TITAN Xp / CUDA 11.8 wheels). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d .venv ] || uv venv -q --python 3.10 .venv
. .venv/bin/activate
# the exact package set of our runs (gpulab2 and gp231); torch 2.1.2+cu118 comes from the PyTorch index
uv pip sync configs/requirements_lock.txt --extra-index-url https://download.pytorch.org/whl/cu118     --index-strategy unsafe-best-match
python -m pytest -q tests echobase/tests
echo SETUP_DONE
