#!/usr/bin/env bash
# Durable user-service entrypoint; only these four child workers are owned here.
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")/.."
OUT="$PWD/results/cross_dataset_b4_c2"
PY='/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>"$OUT/supervisor.lock"
flock -n 9
"$PY" -c 'import json; from benchmark_interventions.run import OUTPUT,load_config; from benchmarks.dataset_registry import digest; c=load_config(OUTPUT); cpu=json.loads((OUTPUT/"cpu_validation.json").read_text()); gate=json.loads((OUTPUT/"smoke/validation.json").read_text()); assert cpu["passed"] and cpu["source_sha256"]==c["source_sha256"]; assert gate["passed"] and gate["config_sha256"]==digest(c); print("CPU and real-GPU gates passed; launching frozen paired jobs",flush=True)'
GPUS=(GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 GPU-b08489af-0a8a-eb48-55ba-8346733af417 GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 GPU-96bc853f-ceea-2404-c704-c48a3d1ca185)
PIDS=()
for shard in 0 1 2 3; do
    mkdir -p -- "$OUT/shard$shard"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" -u -m benchmark_interventions.run --shard "$shard" >>"$OUT/shard$shard/console.txt" 2>&1 &
    PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if ((failed)); then
    printf '%s\n' 'PARTIAL: at least one worker failed. All failure records preserved; no full-suite summary or result push.' >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES='' "$PY" -u -m benchmark_interventions.analyze >>"$OUT/analysis_console.txt" 2>&1
printf '%s\n' 'All 2620 paired samples and independent raw-logit audits completed.'
# Raw full-vocabulary NPZ arrays stay on the data disk, not in GitHub.
git add -- 'results/cross_dataset_b4_c2/*.json' 'results/cross_dataset_b4_c2/*.csv' 'results/cross_dataset_b4_c2/*.md' 'results/cross_dataset_b4_c2/*.txt' 'results/cross_dataset_b4_c2/figures/*.png'
while IFS= read -r path; do
    case "$path" in
        vlm_research/results/cross_dataset_b4_c2/*) ;;
        *) printf 'Unexpected staged path; refusing automatic result commit: %s\n' "$path" >&2; exit 1 ;;
    esac
done < <(git diff --cached --name-only)
git -c user.name=Codex -c user.email=codex@localhost commit --quiet -m 'Publish complete frozen cross-dataset Full B4 C2 paired scores and audits'
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git push origin HEAD:refs/heads/codex/cross-dataset-b4-c2
git rev-parse HEAD
printf '%s\n' 'Completed, GPU workers released, experiment branch pushed; main unchanged.'
