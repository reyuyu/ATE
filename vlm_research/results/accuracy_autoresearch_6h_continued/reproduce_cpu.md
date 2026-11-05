# CPU Reproduction and Execution Boundary

This is a continuation of commit `60d262f83a981e49a614ef8213e5311c3c7c6469`.
The original six-hour clock is not reset. The previous FINAL was inspected;
`05_recheck` is exploratory historical-cohort re-evaluation, not a fresh FINAL.

## Shared Inputs

`execution_freeze.json` identifies the unchanged shared manifest, split, model
weight files, physical GPU UUIDs and execution source hashes. Historical records
are read directly from the previous worktree only after payload, sample,
configuration, prompt and source bindings pass. No model/image/old-array copy is
required. A complete clone retains the parent commit's historical records; the
server paths in `runtime.py` describe this actual execution, not a portable data
download command.

## Recompute Completed Results Without GPU

Working directory:
`/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h-continued`

Python:
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`

Use the frozen environment and hide all CUDA devices for these CPU-only steps:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python -m unittest vlm_research.accuracy_autoresearch_continued.tests -v
python -m vlm_research.accuracy_autoresearch_continued.manage analyze 05_recheck --offline
python -m vlm_research.accuracy_autoresearch_continued.manage performance 06_recheck_performance --offline
python -m vlm_research.accuracy_autoresearch_continued.report
python -m vlm_research.accuracy_autoresearch_continued.finish full-cohort
python -m vlm_research.accuracy_autoresearch_continued.finish catalog
```

The offline option only recomputes CPU statistics from complete checkpoints. It
cannot dispatch a GPU job or bypass a GPU deadline. It is rejected for init,
promotion and freezing. `finish audit` additionally requires the actual worker
completion/release records and original weight files. Artifact hashes describe
the delivered files; regeneration changes timestamps and is not expected to
reproduce timestamped file bytes.

The report command regenerates numerical tables, figures and the initial report
body. The delivered report also contains a manually reviewed conclusion based
on those tables; preserve that conclusion when checking regenerated numbers.

## GPU Run Was Bounded and Is Not Auto-Restarted

The only GPU entry point was `runtime --rank 0/1`, with each process exposed to
its single frozen authorized GPU2/GPU3 UUID. Both workers used an immutable
source snapshot, one resident model each, and the original image-cluster
assignment. `06_recheck_performance` has `finish_after=true`. No additional
condition may be dispatched after the original freeze. Any future GPU study
requires separate user authorization, a new protocol and an independent output
directory; do not delete completion/deadline files to restart this study.
