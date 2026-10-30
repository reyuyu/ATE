# Observation 2 Result Reproduction

Complete: 200 COCO images, 591 frozen Decode positions, 11,229 intervention
forwards, 11,820 CSV rows including 591 Full baselines. No final-answer accuracy
or GPU acceleration claim. See `research_summary.md` for eight research answers.

The frozen GPU collector commit is
`13b2f575e0485dba89284d977855e34ab58b9701`.
All twelve stage/worker environment records use this commit and the same pinned
model. Added CPU-only analysis does not alter collector sources or old results.

## Exact Commands

From `/data/lk/ATE-worktrees/token-role-decode/vlm_research`:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/data/lk/深度学习/.envs/ate-vlm-research/bin/python
$PY -m unittest -v test_token_role_decode test_token_role_analysis
$PY -u -m paper_rq1_token_role_decode.finalize review
```

No GPU is used by these read-only collection checks. Review also verifies all
404 NPZ hashes and recomputes twelve deterministic raw-logit metric rows.
The original automatically generated report is preserved under
`audit/pre_delivery/`; the original `analysis_completion.json` still describes
that pre-delivery version. Use `final_completion.json` for delivery integrity.

For interrupted **GPU collection**, only on the same inputs and idle GPUs:

```bash
$PY -u -m paper_rq1_token_role_decode.launch
```

The launcher sets one fixed GPU UUID per worker, checks idle status, resumes
valid atomic checkpoints, waits for all four smoke workers, then all Full
workers, freezes positions centrally, and only then starts ablation. Completed
shards are reused. Do not delete manifests or rerun data selection to change
images, target words, failures or numerical bounds.

For a new isolated reproduction, follow the collector package README to prepare
the exact data, run CPU acceptance, commit sources and freeze before launch.
CPU statistical regeneration is:

```bash
$PY -u -m paper_rq1_token_role_decode.analyze
$PY -u -m paper_rq1_token_role_decode.case_figure
```

These recreate the numerical tables and five PNG/PDF figures, refine the
post-hoc case's actual token-position strip, and write the automatic English
report. They are **not** commands for overwriting the finalized
Chinese interpretation or delivery inventory in the completed worktree. Run it
in a separate reproduction worktree with matching local resource paths and a
fresh result inventory; do not replace this result's historical audit files.

## Artifact Map

- `sample_manifest.json`: original official image/caption/source/input binding.
- `execution_manifest.json`, `cpu_validation.json`: code/model/environment freeze.
- `full_caption_generations.csv`, `full_token_annotations.csv`: all native tokens,
  per-step confidence/entropy, exact word/subword alignment including exclusions.
- `selected_token_manifest.json`, `target_freeze.json`: 591 targets frozen before
  any formal ablation. Missing roles retained explicitly, no fabricated examples.
- `per_token_layer_ablation.csv`: every Full, Whole, Image and All32 observation.
- `token_type_summary.csv`: per-layer and nine-measured-layer mean statistics,
  medians/tails/absolute changes. Nine-layer mean is not a simultaneous bypass.
- `paired_token_type_comparison.csv`, `controlled_effect_analysis.csv`: matched
  image contrasts and fixed effects. Ordinary95% CI plus planned22-comparison
  Bonferroni bootstrap intervals. Relation comparisons are descriptive.
- `all32_ablation_summary.csv`, `all32_*_examples.csv`: severe and near-unchanged
  diagnostics. The near-unchanged CSV is honestly header-only (zero examples).
- `classification_audit*`: frozen lexicon/context audit, not gold POS/grounding.
- `smoke/`, `full/`, `ablation/`: separate four-shard logs, atomic per-image
  checkpoints, environments, completed status and peak memory.
- `delivery_review.json`, `raw_logits_manifest.csv`, `worker_performance.csv`:
  independent CPU integrity checks and observed cost; no speedup comparison.
- `figures/`: all five real-data figures, PNG and vector PDF.
- `case_study_token_positions.csv`: actual native positions and measured case
  losses; no values imputed at other words or unmeasured layers. Original case
  PNG/PDF are preserved under `audit/pre_delivery/figures/`.
- `report.md`, `research_summary.md`: quantitative report and Chinese conclusions.

Raw native/altered FP32 full-vocabulary logits are kept at
`/data/lk/ATE-resources/token_role_decode/raw/` (404 NPZ files, 1,084,958,731 bytes).
They are not uploaded as Git blobs. Their exact paths, sizes and hashes are
committed, as are per-condition metrics/checkpoints. Images and pinned model
weights remain in the existing `/data/lk/ATE-resources/` resource tree.
