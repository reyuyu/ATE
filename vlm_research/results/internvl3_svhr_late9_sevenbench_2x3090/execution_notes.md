# Execution Notes

- Isolated branch: `codex/internvl3-svhr-late9-sevenbench-2x3090`.
- Worktree: `/data/lk/ATE-worktrees/internvl3-svhr-late9-sevenbench-2x3090`.
- Frozen GPU execution commit: `215339e8f172dbd78ac9c09388abb9fd0a786f16`.
- Physical GPU0: `GPU-a63c6362-dcfd-4a26-c44b-014b40127c89`.
- Physical GPU2: `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`.
- GPU1/GPU3 belong to unrelated concurrent experiments. No external process was terminated.
- No new storage on `/data1`; no historical result, model or dataset deletion.

## Frozen Execution

The initial smoke launch stopped before loading the model because Torch reports
the CUDA UUID without the `GPU-` display prefix. The fix normalizes only this
display prefix. Failure evidence is retained under `attempts/01_uuid_representation/`.
CPU tests and both seven-benchmark GPU smokes passed before formal workers started.
No runtime package, physical engine, sample, protocol or processor binding changed
while the formal workers were executing.

Each worker owns one SQLite database in `server_only/`, commits each prediction,
and records the frozen protocol/source hashes. Retrying the exact frozen command
skips existing verified records. Do not change bindings to resume under a new
protocol without a separate, audited migration.

## CPU Analysis

`vlm_research/internvl3_svhr_late9_report.py` is an independent CPU-only reporter,
outside the frozen GPU package. It may be developed while GPU workers execute.
The InternVL environment does not contain matplotlib; plotting uses the existing
`ate-vlm-research` CPU environment. No inference environment was upgraded.

After both formal workers exit, run from this worktree:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/lk/深度学习/.envs/ate-vlm-research/bin/python -u \
  -m vlm_research.internvl3_svhr_late9_report --final
```

`--final` requires every locally available frozen request to have all three methods,
requires every unavailable request to have explicit BLOCKED evidence, re-scores
every saved prediction, verifies actual KV shapes and frozen bindings, checks that
the two worker processes exited and released CUDA memory, and seals result hashes.
Intermediate reports generated during development are not final coverage claims.

The original source and data authorities are preserved in `source_manifest.json`,
`protocol.json`, `model_integrity.json`, `dataset_manifest.json` and the historical
reuse audit. Reused generation GPU UUIDs are not replaced with this run's input
verification GPU UUIDs.

## Dataset Blockers And Efficiency

The missing MME image must match its historical SHA256; the local same-name image
does not. OCRBench HF images mostly differ from the historical official ZIP bytes.
`data_recovery_attempts.json` records failed official downloads and a rejected
author-repository candidate. The latter temporary download alone was removed
after hashing and checking all 1000 image payloads. No replacement bytes were used.

Efficiency is a separate fixed-32-token, 48-image experiment, permitted only after
all seven full quality benchmarks complete. Missing exact input data therefore
leaves efficiency NOT_RUN_QUALITY_INCOMPLETE; natural EOS wall time is not a speed
benchmark, and a smaller persistent visual KV payload is not proof of acceleration.

## Delivery Boundary

Commit locally on this experimental branch only. Do not push, create a PR, merge,
or modify main. Release only this experiment's GPU0/GPU2 workers. Do not start
additional configurations or another model.
