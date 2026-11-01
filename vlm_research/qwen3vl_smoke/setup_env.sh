#!/usr/bin/env bash
set -euo pipefail

old_python='/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
new_env='/data/lk/深度学习/.envs/ate-qwen3vl'
results='vlm_research/results/qwen3vl_8b_single_gpu_smoke'
export CUDA_VISIBLE_DEVICES=''
export PIP_CACHE_DIR='/data/lk/ATE-resources/cache/pip-qwen3vl'
export TMPDIR='/data/lk/ATE-resources/tmp/qwen3vl'
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
"$old_python" -c 'import json; assert json.load(open("vlm_research/results/qwen3vl_8b_single_gpu_smoke/download_validation.json"))["passed"]'
if [[ ! -e "$new_env/bin/python" ]]; then
    "$old_python" -m venv "$new_env"
fi
"$new_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11); assert sys.prefix.endswith("ate-qwen3vl")'
"$new_env/bin/python" -m pip install --index-url "${QWEN_PIP_INDEX_URL:-https://pypi.org/simple}" \
    --find-links /data/lk/ATE-resources/wheels \
    -r vlm_research/qwen3vl_smoke/requirements.txt
"$new_env/bin/python" -m pip check
"$new_env/bin/python" -m pip freeze > "$results/requirements-actual.txt"
