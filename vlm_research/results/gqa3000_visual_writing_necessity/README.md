# GQA3000 Visual Writing Layer Necessity

Independent branch: `codex/gqa3000-visual-writing-necessity-curve`.

Execution source commit: `5c0fa4c5ab` (full hash in `protocol.json`).
Frozen matrix: 3000 questions, 2409 image clusters, 34 conditions, 102000
complete greedy generations and corresponding fixed-gold teacher-forced NLLs.

## Completion Gate

Use `completion.json`, not this README or a partial progress file, as the
completion certificate. Until its status is `COMPLETE`, the experiment is
partial and no full-cohort conclusion or paper curve is available.

Two workers run on physical GPU2/GPU3 with fixed UUIDs and disjoint image-cluster
shards. The one-shot CPU finalizer waits for both workers to exit successfully,
checks resource release and hashes, audits all conditions, performs the frozen
20000-draw paired cluster bootstrap, generates figures, archives raw records,
and commits/pushes this experiment branch. It never starts another experiment,
changes conditions, or silently retries failed model runs.

## Protocol

All conditions prohibit original P-1 and every real generated/gold-prefix query
from directly reading original image keys at all 32 layers. All-Writer permits
all historical R queries to read image keys in all layers; No-Writer permits
none. Each Drop-L condition masks R-to-image connections at only that layer.
All-Writer is not native Full. Image queries, MLP, residual and KV updates stay
native; this is a pre-softmax source mask, not physical computation removal.

Gold continuation tokenization preserves the exact prompt prefix. NLL includes
all fixed official answer tokens, excludes EOS, and has no extra BOS. Per-answer
token means are averaged equally across questions. Teacher forcing and free
generation have disjoint cache storage after the identical prefill.

## Artifacts

- `protocol.json`, `sample_manifest.json`, `gold_sequences.json`,
  `input_bindings.json`, `shards.json`, `overlap_audit.json`: frozen bindings.
- `cpu_validation.json`, `cpu_analysis_validation.json`, `shard*/smoke.json`:
  real correctness checks. No tolerances were relaxed.
- `shard*/progress.json`: mutable monitoring snapshots, not result summaries.
- `per_question.csv`, `negative_cases.csv`: complete outputs and signed effects.
- `condition_summary.csv`, `layer_necessity_summary.csv`, `task_summary.csv`,
  `image_disjoint_summary.csv`: paired cluster estimates and intervals.
- `figure_A_visual_writing_necessity.*`, `figure_B_accuracy_sensitivity.*`,
  `figure_C_concentration.*`, `task_gold_nll.*`, `task_accuracy.*`: PNG/PDF.
- `report_zh.md`, `validation.json`, `resource_release.json`,
  `final_hash_audit.json`, `analysis_complete.json`, `completion.json`.
- `raw_checkpoints/*.tar.gz`: exact raw checkpoint bytes, with relative paths
  and SHA256s verified against `checkpoint_hashes.json` and
  `raw_archive_manifest.json`. Loose runtime checkpoints are intentionally
  excluded from Git; the archives preserve all 102000 records.

Initial pre-smoke engineering failures are preserved separately and are never
counted as formal observations. Native Full and old 240-question writing results
are not substituted for this run's All-Writer baseline.

This is an expanded analysis of an existing research cohort, not independent
external confirmation. Single-layer necessity is not multi-layer sufficiency,
an information fraction, or evidence of actual inference acceleration.
