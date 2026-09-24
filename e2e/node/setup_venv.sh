#!/usr/bin/env bash
# Prepare the environment for the native-node experiments in e2e/node/, entirely inside this repo:
#   <repo>/.venv                        python venv with torch (CUDA 12.8 wheels, needed for Blackwell)
#   <repo>/.venv/bin/cuda-checkpoint    NVIDIA cuda-checkpoint (scripts/install_cuda_checkpoint.sh)
# .venv is git-ignored. Nothing outside the repo is created; no root needed.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
venv="$repo/.venv"
torch_index=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv" || {
    echo "python3 -m venv failed (python3-venv missing?). Installing it needs root: ask the admin." >&2
    exit 1
  }
fi
"$venv/bin/python" -m pip install -q --upgrade pip
"$venv/bin/python" -c "import torch" 2>/dev/null || "$venv/bin/pip" install torch --index-url "$torch_index"
"$venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

[[ -x "$venv/bin/cuda-checkpoint" ]] || "$repo/scripts/install_cuda_checkpoint.sh" --prefix "$venv/bin"
echo "ready: $venv"
