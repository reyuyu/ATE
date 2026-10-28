# Server recovery, 2026-10-28

## Scope and actual state

Checkout: `/data/lk/深度学习/ATE`. Latest upstream main retrieved for this handoff:
`7970be981e3462560cb66d83e1014eaa1b1ebdae`, containing baseline
`bcc3c3da2ff7d416cd988c83264fd3ffc18afaed`.
Review branch: `codex/ate-server-handoff-20261028`.

**Recovery is not finished; no new 7B experiment has run on this server.**

- All 1,487 original tracked files were restored against their Git blob IDs.
  All 1,143 hashes in the handoff artifact manifest pass. Original tracked
  algorithms, manifests, configurations and results are unchanged.
- Historical Phase3.2/3.3 have 1,920/3,120 rows. Keep the 240-image balanced
  cohort and 12 validation images. All 53 task Attention caches exist and have
  shape `[32,32,2,576]`; these are NOT caches for all 240 images.
- A path-aware Phase3.5 audit passes all 115 repository checks. Its seven image
  checks (five distinct images) are pending because raw images are still absent.
  Original `--verify-existing` failures from old absolute image paths are retained.
- Seven new path/download-control tests pass in AFAC. These are NOT the seven
  original inheritance tests and NOT the 101-test fixed-environment acceptance.
  The original 7/101 tests are queued after the fixed environment installation.
- Four RTX3090 GPUs were idle at the recorded preflight. No training, new
  conditions, 240-image intervention, GPU watcher or automatic GPU launch exists.

See `results/server_recovery_20261028/` for the timestamped snapshot, repository
verification, failed legacy checks, corrected path audit and actual utility logs.
Snapshots describe their collection time, not a live completion claim.

## Fixed resources and paths

Resources, archives, weights, environment wheels and download logs are outside Git:
`/data/lk/ATE-resources/`.

- Dataset: `TIGER-Lab/MMEB-V2`, recovery revision
  `e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1`, only
  `image-tasks/mmeb_v1.tar.gz` (7,031,608,592 bytes). Expected archive SHA256:
  `feeaf4a8319cba587b982ec976fba225d13e134a3217e1cad5c251e64299346a`.
  This is a NEW recovery pin; the old download revision remains unknown.
  Extract the complete `MMEB/GQA/` subdirectory, not other MMEB tasks.
- Expected image root:
  `/data/lk/ATE-resources/data/image-tasks/MMEB/GQA`.
  Downloading the correct archive does not certify historical inputs: the
  frozen 252 SHA256s plus image decode/size checks must pass, and all 785 source
  images must be audited. No replacement, resampling or JPEG recompression.
- Model: `llava-hf/llava-1.5-7b-hf`, revision
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`. The 12 small files are present;
  the three weight shards are still downloading. Model directory:
  `/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.
  Shard sizes and LFS SHA256 must match the unchanged `configs/model.json`.
- AFAC is unchanged. Its Python3.11.15 created the independent environment
  `/data/lk/深度学习/.envs/ate-vlm-research`. Twelve exact CUDA12.4 dependencies
  were copied only after checking their version and every wheel RECORD hash;
  no shared writable links were made. Other fixed dependencies are installing.
  Target versions remain `requirements.txt`; actual versions and CPU results
  must be read from the installation audit, not inferred from this target.

The downloader uses `huggingface_hub==0.29.3` in a separate download-library
directory; AFAC itself was not downgraded. Aria2 performs four range connections
per large file, with fixed revision, size, LFS metadata and final SHA256 checks.
Sparse `.transfer` file length is NOT download progress or a verified weight.

## ModelScope transfer switch, 22:58 CST

The HF range jobs were stopped before starting the ModelScope jobs. Existing
`.transfer` files and their `.aria2` range maps are reused, not deleted or
treated as completed files. This changes transport only, not the canonical HF
resource pins, `configs/model.json`, input data or any experiment condition.

- Dataset copy: ModelScope `TIGER-Lab/MMEB-V2`, revision
  `de0e137ddc5b7f12d2b84081d41b1a2ff53902a0`. The archive has the exact pinned
  7,031,608,592-byte size and SHA256 above.
- Weight copy: ModelScope `llava-hf/llava-1.5-7b-hf`, revision
  `456d5355816bd34e90e7f882985f3b5499f2d484`. Each of the three shard metadata
  entries must match the original HF shard size and SHA256 before transfer.
  The 12 small HF files are reused offline from their existing local cache;
  ModelScope configuration/tokenizer files are NOT substituted.
- `--transport modelscope` still uses aria2 with four connections per large
  file, the same range maps and final size/SHA256 checks. A file is renamed
  from `.transfer` only after successful final verification. Metadata and
  pre-transfer range-map SHA256 are recorded in each resource directory's
  `.download-sources/` subdirectory. These are metadata/continuation audits,
  NOT evidence that the full content has already passed its final hash check.
- The separate utility suite now has 14 passing tests in AFAC, including
  wrong-copy metadata rejection, retained partials after failure, resume-map
  preservation, bad-content rejection and offline reuse of original HF small
  files. This does NOT replace the original 7/101 acceptance suites.
- The fixed environment job failed at 22:50:56 CST: the Triton 3.2.0 wheel
  transfer was truncated and rejected by pip's expected SHA256. The check
  must not be bypassed. Environment installation and original 7/101 CPU tests
  remain incomplete; no GPU experiment has started.

Current transfer units and logs:

```bash
systemctl --user status ate-restore-data-modelscope-20261028 --no-pager
systemctl --user status ate-restore-model-modelscope-20261028 --no-pager
# /data/lk/ATE-resources/audit/download_data_modelscope_20261028.txt
# /data/lk/ATE-resources/audit/download_model_modelscope_20261028.txt
```

## Background resource jobs

These are preparation jobs, not model inference or a GPU poller:

```bash
systemctl --user status ate-restore-data-range-20261028 --no-pager
systemctl --user status ate-restore-model-range-20261028 --no-pager
systemctl --user status ate-restore-data-modelscope-20261028 --no-pager
systemctl --user status ate-restore-model-modelscope-20261028 --no-pager
systemctl --user status ate-environment-cpu-paths-20261028 --no-pager
```

Logs are in `/data/lk/ATE-resources/audit/`, especially
`download_data_range_20261028.txt`, `download_model_range_20261028.txt` and
`pinned_env_20261028_retry_paths/environment_install.txt`. The environment job
was intended to run the original 7 and 101 CPU suites after installing the fixed
versions, but is currently failed as described above. A fresh successful install
attempt and the actual CPU acceptance reports are still required.
It will NOT start the GPU experiment. Download failures stay failures and
partials are retained for a manual retry of the same pin.

Earlier direct-Git/tar and single-stream attempts are retained outside Git.
An interrupted clone left one empty CSV placeholder: it was quarantined as
`source/empty_head_statistics_csv_clone_placeholder`. Three complete members
were recovered from a partial tar only after BOTH Git-blob and frozen SHA256
verification; the partial archive itself is never labelled complete. Final
all-file audit: `audit/checkout_hydration_final_20261028.json`.
Earlier resource/environment units were explicitly stopped and superseded;
their interrupted JSON `running` fields do not mean those units remain active.

## Continue after resource completion

Use the fixed environment below, not AFAC, for experiment acceptance. First
inspect the three preparation units and CPU reports. Do not run concurrently
with an unfinished environment installation.

```bash
cd /data/lk/深度学习/ATE/vlm_research
PY=/data/lk/深度学习/.envs/ate-vlm-research/bin/python
GQA=/data/lk/ATE-resources/data/image-tasks/MMEB/GQA
MODEL=/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9

# Select NEW audit filenames; existing audits must not be overwritten.
CUDA_VISIBLE_DEVICES='' "$PY" handoff/verify_recovery.py \
  --image-root "$GQA" --verify-source-images --model-dir "$MODEL" \
  --output /data/lk/ATE-resources/audit/recovery_complete_001.json
CUDA_VISIBLE_DEVICES='' "$PY" handoff/verify_history_remote.py \
  --image-root "$GQA" \
  --output /data/lk/ATE-resources/audit/history_complete_001.json

# Manual one-time GPU check: choose ONE free device with >=16000MiB free.
nvidia-smi
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" attention_inheritance_remote.py --model-path "$MODEL" --image-root "$GQA"
```

`attention_inheritance_remote.py` creates a fresh `remote_<host>_<timestamp>`
directory, refuses overwrite, records runtime-only path mapping and hashes,
enforces fixed dependency versions, reruns recovery and CPU checks, then applies
the SAME `VisualAttentionInheritance`, input mapping and scoring functions.
Only `15133706`, Full L23/L24 sources, lambda `[0,.5,1]`, L24..L31, P-1=602,
visual positions5..580. No EOS, no label/order changes, no training.

Historical Full and cache checks are NOT disabled. If the RTX3090 differs from
the historical RTX4090, the runner saves actual native logits, hidden states,
attention and numerical differences BEFORE stopping. Investigate the separate
cross-machine audit; do not silently relax tolerances or call the six conditions
completed. Both lambda-zero controls must be exact to the same-machine Full.

Outputs include per-condition CSV/JSON, raw logits, even failing layer audits,
environment/time/VRAM, final cleanup checks, figure and report. Historical
Physical K24 is labelled historical only. No result is manufactured for a
condition that has not completed. Future 240-image work remains unstarted.
