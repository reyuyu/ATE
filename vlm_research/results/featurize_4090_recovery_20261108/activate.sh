# Source this file. It configures the environment without starting an experiment.
source /home/featurize/.venvs/ate-frozen/bin/activate
export ATE_CODE_ROOT=/home/featurize/work/ATE-4090-recovery-20261108
export ATE_DATA_ROOT=/home/featurize/data/ate-recovery
export ATE_MODEL_PATH=/home/featurize/ate-local-models/llava-1.5-7b-hf
export PYTHONPATH="$ATE_CODE_ROOT:$ATE_CODE_ROOT/vlm_research/results/featurize_4090_recovery_20261108${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/featurize/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ATE_CODE_ROOT"
