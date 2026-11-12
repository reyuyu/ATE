# CPU-only Score Recalculation

Date: 2026-11-12 (Asia/Shanghai).
Base: `65c4cac2f925ebefb073c8db03caab119fcc765a`.
Branch: `codex/internvl3-svhr-late9-sevenbench-2x3090`.

## Boundaries

No model inference, GPU allocation, new downloads, data substitution, scoring-rule
changes, environment upgrades, main modification, force push or PR creation.
The original 106,041 predictions are read-only. SQLite files are opened using
`mode=ro` and `query_only=ON`, with file hashes checked before and after.

## Development Checks

The existing `ate-vlm-research`, `ate-internvl3` and `ate-qwen3vl` environments
do not contain scikit-learn, needed to execute the pinned official MME scorer's
`compute_metric`. The first CPU attempt stopped at `ModuleNotFoundError` before
writing new score products. No packages were installed or changed.

The existing `/home/likai/anaconda3/bin/python` has scikit-learn 1.7.2 and is used
only for offline scoring. Actual Python and NumPy versions are recorded in
`score_recalculation_validation.json`. `CUDA_VISIBLE_DEVICES=''` is required.

A second development check reached the official scorer and stopped because
the new stdout parser initially included its component `total score:` lines
alongside the 14 task lines. Only the new parser was fixed to accept the frozen
14 task names, then a complete 2800-point synthetic fixture was added to the
CPU tests. Neither official source nor scoring semantics were changed.

An exact float-value comparison also exposed 108 TextVQA final-bit differences
between the existing Python 3.11 execution records and this CPU environment.
Maximum difference: 2.220446049250313e-16. All parsed answers are identical.
The full old/new values are preserved in `score_recalculation_discrepancies.json`.
Acceptance uses the original reporter's existing 1e-10 score threshold, not a
new or relaxed model-logits threshold; no model-logits validation runs here.
The five complete benchmark score fields and all historical records stay intact.

## Reproduction

From this worktree:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/likai/anaconda3/bin/python -m unittest \
  vlm_research.test_internvl3_svhr_partial_scores -v

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/likai/anaconda3/bin/python -m vlm_research.internvl3_svhr_partial_scores

env CUDA_VISIBLE_DEVICES='' /home/likai/anaconda3/bin/python \
  -m vlm_research.internvl3_svhr_partial_scores --seal
```

The official MME scorer is invoked on temporary CPU text files preserving each
complete same-image pair. Its 14 task scores are checked against independent
Fraction-based arithmetic. Missing tasks produce missing values, never zero.
The official OCRBench scoring loop is run on all nine available predictions.
Incorrect, empty, verbose or truncated outputs are not silently filtered.

## Integrity And Publication

The original artifact ledger is preserved in `pre_recalculation_artifact_hashes.json`.
Only `report_zh.md` among the old ledger's covered files is intentionally updated.
The renewed `artifact_hashes.json` covers the current report and all new outputs.
The original `final_cpu_review.json` and `validation.json` remain historical
evidence, not claims that this supplement has completed missing datasets.

The user explicitly authorized publishing this experiment branch. The initial
remote check found no same-name branch; origin is `git@github.com:reyuyu/ATE.git`.
The configured fetch refspec originally tracked only main, so the two original
SVH authority branches were explicitly fetched as read-only provenance refs.
No commits were reset, rebased, merged into main or force-pushed.

Final status remains `PARTIAL_5_OF_7_FULL_BENCHMARKS`, with the historical
`EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER` label unchanged.
