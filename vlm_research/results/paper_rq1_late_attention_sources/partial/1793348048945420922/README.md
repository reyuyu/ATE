# Late-Layer Attention Source Decomposition

See report.md for the actual current completion status; never infer completion
from the presence of runnable code. All source, sample and protocol hashes bind
the unchanged collector commit `9352a999c0cb4b5c0ffbdc7fd2941165874630ab`. CPU-only analysis is
separate; it cannot alter frozen worker sources or Observation2 artifacts.

## Resume

From `/data/lk/ATE-worktrees/late-attention-sources/vlm_research` on the same local resources, only when allfour GPUs are idle:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/data/lk/深度学习/.envs/ate-vlm-research/bin/python
$PY -m unittest -v test_late_attention_sources test_token_role_decode test_late_source_statistics
$PY -u -m paper_rq1_late_attention_sources.launch
$PY -u -m paper_rq1_late_attention_sources.analyze
$PY -u -m paper_rq1_late_attention_sources.finalize complete
```

Do not rerun data selection/CPU freeze to replace the already frozen samples.
Valid completed smoke shards/checkpoints are reused. Allfour Smoke completions
are required before Full; all Full captions are required before central target
selection; the exact target SHA is required before source interventions.
An occupied GPU stops that stage without terminating any other process. See
resource_event.json for the initial refused GPU0 process, and original logs.

## Artifacts

- sample_manifest.json, data_manifest.json: independent200+1 smoke image binding.
- cpu_validation.json, execution_manifest.json: tested collector/protocol/model.
- smoke/, full/, ablation/: actual per-GPU environments/logs/checkpoints only.
- selected_token_manifest.json: created only after native Full200 completes.
- source_ablation_per_token.csv: all same-prefix Full and source interventions.
- native_source_contributions.csv: actual native mass/projected norm/cosines.
- source_layer_summary.csv, native_source_summary.csv: signed/absolute/tail/flip
  summaries, with image-cluster confidence intervals.
- planned_primary_contrasts.csv: frozen41-comparison family; role, source, deep
  continuity and recent/older comparisons. Full controls/rank audit retained.
- visual_function_paired_comparison.csv, deep_layer_peak_analysis.csv,
  recent_vs_older_history.csv: user-requested primary breakdowns.
- numerical_validation.json: complete only after all formal shards pass.
- figures/: five PNG/PDF real-data figures, created only after complete gates.
- partial/: earlier resource-blocked snapshots, never formal result evidence.

Absent artifacts mean not yet executed, not zero effects. Raw full-vocabulary
FP32 NPZ files stay at `/data/lk/ATE-resources/late_attention_sources/raw/`, with
SHA256 in checkpoints and completed raw_logits_manifest.csv. Model weights and
raw images are not Git blobs. Single-step LogP loss is not answer accuracy,
source losses are not additive, and no actual speedup is measured.
