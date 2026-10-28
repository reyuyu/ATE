#!/usr/bin/env bash
# Fixed handoff dependencies and CPU validation only. Never start a GPU job.
set -euo pipefail

PYTHON=${1:?Usage: bash install_server_env.sh VENV_PYTHON RESOURCE_ROOT MODEL_DIR NEW_AUDIT_DIR}
RESOURCES=${2:?Missing resource root}
MODEL=${3:?Missing fixed model directory}
AUDIT=${4:?Missing NEW audit directory}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir "$AUDIT"
exec > >(tee -a "$AUDIT/environment_install.txt") 2>&1
trap 'printf "Environment/CPU stage FAILED at %s; no GPU job started.\n" "$(date --iso-8601=seconds)"' ERR
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export PIP_CACHE_DIR="$RESOURCES/pip-cache"
export TMPDIR="$RESOURCES/source"
export HF_HOME="$RESOURCES/hf-home"
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="$RESOURCES/matplotlib"
WHEELS="$RESOURCES/wheels"
mkdir -p "$WHEELS"

# Hashes are from the official PyTorch cu124 simple index, not mirror filenames.
aria2c --no-conf=true --all-proxy= --continue=true --auto-file-renaming=false \
  --allow-overwrite=false --file-allocation=none --split=4 --max-connection-per-server=4 \
  --max-tries=5 --retry-wait=15 --connect-timeout=20 --timeout=90 \
  --show-console-readout=false --summary-interval=60 --console-log-level=warn \
  --checksum=sha-256=d4c3e9a8d31a7c0fcbb9da17c31a1917e1fac26c566a4cfbd8c9568ad7cade79 \
  --dir="$WHEELS" --out='torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl' \
  'https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-linux_x86_64.whl'
aria2c --no-conf=true --all-proxy= --continue=true --auto-file-renaming=false \
  --allow-overwrite=false --file-allocation=none --split=2 --max-connection-per-server=2 \
  --max-tries=5 --retry-wait=15 --connect-timeout=20 --timeout=90 \
  --show-console-readout=false --summary-interval=60 --console-log-level=warn \
  --checksum=sha-256=137376805aca5ba57bd2c7a3ecb8569df961dbe82b128aac9b3b0a7125ef9385 \
  --dir="$WHEELS" --out='torchvision-0.21.0+cu124-cp311-cp311-linux_x86_64.whl' \
  'https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-linux_x86_64.whl'

# PyPI 3.2.0 metadata supplies this exact wheel hash; preserve partial ranges.
aria2c --no-conf=true --all-proxy= --continue=true --auto-file-renaming=false \
  --allow-overwrite=false --file-allocation=none --split=4 --max-connection-per-server=4 \
  --max-tries=5 --retry-wait=15 --connect-timeout=20 --timeout=90 \
  --show-console-readout=false --summary-interval=60 --console-log-level=warn \
  --checksum=sha-256=8009a1fb093ee8546495e96731336a33fb8856a38e45bb4ab6affd6dbc3ba220 \
  --dir="$WHEELS" --out='triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' \
  'https://files.pythonhosted.org/packages/a7/2e/757d2280d4fefe7d33af7615124e7e298ae7b8e3bc4446cdb8e88b0f9bab/triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'

"$PYTHON" -m pip install --find-links "$WHEELS" --constraint "$ROOT/requirements.txt" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "$WHEELS/torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl" \
  "$WHEELS/torchvision-0.21.0+cu124-cp311-cp311-linux_x86_64.whl"
"$PYTHON" -m pip install --find-links "$WHEELS" --requirement "$ROOT/requirements.txt" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PYTHON" -m pip check > "$AUDIT/pip_check.txt"
"$PYTHON" -m pip freeze > "$AUDIT/pip_freeze.txt"
cd "$ROOT"
"$PYTHON" -c 'import importlib.metadata as m, platform, sys; from attention_inheritance_remote import PINS; actual={n:m.version(n) for n in PINS}; assert actual==PINS; print(platform.python_version(), sys.executable, actual)' \
  > "$AUDIT/pinned_versions.txt"
"$PYTHON" cpu_tests_remote.py --model-path "$MODEL" --suite inheritance --output-dir "$AUDIT/cpu_inheritance"
"$PYTHON" cpu_tests_remote.py --model-path "$MODEL" --suite all --output-dir "$AUDIT/cpu_all"
"$PYTHON" -m unittest cpu_remote_tools_tests -v > "$AUDIT/remote_controls_tests.txt" 2>&1
# Image verification is a separate data stage; historical inventories contain
# old absolute paths and must use handoff/verify_history_remote.py after recovery.
printf 'Fixed environment and CPU checks completed at %s. GPU experiment NOT started.\n' "$(date --iso-8601=seconds)"
