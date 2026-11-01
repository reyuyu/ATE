#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export TMPDIR='/data/lk/ATE-resources/tmp/internvl3'
export PIP_CACHE_DIR='/data/lk/ATE-resources/cache/pip-qwen3vl'
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
envdir='/data/lk/深度学习/.envs/ate-internvl3'
if [[ ! -e "$envdir/bin/python" ]]; then
    '/data/lk/深度学习/.envs/ate-vlm-research/bin/python' -m venv "$envdir"
fi
"$envdir/bin/python" -c 'import sys; assert sys.prefix.endswith("ate-internvl3")'
"$envdir/bin/python" -m pip download --no-deps --dest /data/lk/ATE-resources/wheels \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple timm==0.9.16 einops==0.8.1
"$envdir/bin/python" -m pip install --no-index \
    --find-links /data/lk/ATE-resources/wheels -r vlm_research/internvl3_smoke/requirements.txt
"$envdir/bin/python" -m pip check
"$envdir/bin/python" -m pip freeze > vlm_research/results/internvl3_8b_single_gpu_smoke/requirements-actual.txt
