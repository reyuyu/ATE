set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=60
E=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449
W=/home/featurize/ate-local-envs/restore-wheelhouse
O=/home/featurize/work/ATE/vlm_research/results/featurize_29535_environment_restore
if [ ! -e "$E/bin/python" ]; then python3 -m venv "$E"; fi
mkdir -p "$W"
if ! "$E/bin/python" -c 'import torch,torchvision; assert torch.__version__=="2.6.0+cu124" and torchvision.__version__=="0.21.0+cu124"'; then
 "$E/bin/python" -m pip download --no-deps --dest "$W" --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' 'triton==3.2.0'
fi
"$E/bin/python" -m pip install --find-links "$W" --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$O/previous_requirements.lock.txt"
"$E/bin/python" -m pip check
"$E/bin/python" -c 'import torch,torchvision,transformers,json; assert torch.__version__=="2.6.0+cu124" and torchvision.__version__=="0.21.0+cu124" and transformers.__version__=="4.49.0"; assert not torch.cuda.is_initialized(); print(json.dumps({"status":"PASS","torch":torch.__version__,"torchvision":torchvision.__version__,"transformers":transformers.__version__,"CUDA_build":torch.version.cuda,"CUDA_initialized":False}))'
