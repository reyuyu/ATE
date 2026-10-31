# RQ2 Experiment 2.1: Single-layer source knockout

Read `report_zh.md` for results once `integrity.json` reports complete. A frozen
protocol or a partial checkpoint directory is not a completed experiment.

## Frozen Scope

- Original Figure 1 commit: `221cdde56dde53908aaa953fe26672fd33334f84`.
- Base worktree commit: `2b10df01904a7fc8090756a90de72c75f91e0f47`.
- Branch: `codex/rq2-single-layer-source-knockout`; main stays unchanged.
- 240 questions: six tasks, 40 per task; seed `20261031`, stable hash selection.
- 235 image SHA256 clusters, assigned whole to two disjoint 120-question shards.
- Physical GPU2 and GPU3 only, bound by UUID in `protocol.json`.
- 32 layers x Image/Prompt/BOS OFF = 96 conditions per question, 23,040 forwards.
- Original last prompt query P-1 only, pre-softmax hard masking. No decode,
  generation, full-answer accuracy, training, or extra intervention conditions.
- Prompt OFF excludes P-1 itself. Historical Prompt Mass includes self; the CSV
  retains both definitions and the actual self mass to prevent conflation.

## Reproduction and Resume

Use `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`, with
`PYTHONPATH=vlm_research`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.

For a fresh output location, prepare samples and run CPU validation:

```bash
python -m rq2_source_knockout.run prepare
python -m rq2_source_knockout.run cpu
```

Commit the exact source before the GPU smokes. Run `smoke --rank 0` with
`CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6` and
`smoke --rank 1` with
`CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`.
Only after both pass, run `python -m rq2_source_knockout.run freeze` and commit
the frozen protocol. Launch `worker --rank 0/1` with the same respective UUIDs.
Each worker refuses occupied GPUs and validates existing checkpoints before
resuming. Completed workers return without allocating GPU memory.

After both completion files exist, run:

```bash
python -m rq2_source_knockout.finalize
```

The independent finalizer preserves all original frozen files. It replays loss
subtraction in FP32 exactly: the frozen draft collector used an incorrect exact
comparison against Python FP64 subtraction. There is no tolerance relaxation,
rerun, result substitution, or change to sampling, metrics, statistics, or plots.
The finalizer also checks baseline vectors, identities, disjoint image clusters,
all 96 conditions, blocked key counts, source masses, and checkpoint hashes.

## Figures and Statistics

- `figure1_source_necessity.pdf/png`: six tasks, same loss-axis scale.
- `overall_source_necessity_and_flip.pdf/png`: exploratory 240-question aggregate.
- `source_sensitivity_heatmap.pdf/png`: 32 x 3, shared signed color scale.
- `top1_flip_tasks.pdf/png`: first-answer prediction changes, not accuracy.
- Shaded regions: gray L0-L1, green L2-L5, salmon L13-L16.
- Loss is fixed Full-top1 logP loss in nats, including negative values.
- 20,000 whole-image SHA256 cluster bootstrap draws; question-weighted means.
- Pointwise 95% intervals are exploratory, not multiplicity-corrected evidence.
- `mass_loss_correlations.csv` distinguishes within-layer question correlations,
  across-layer means, and depth-confounded pooled observations.

Native Full is paired on the same GPU for all 240 questions to obtain exact
current distributions and self mass. Historical vectors and masses are checked,
not silently mixed into an inconsistent paired baseline. Current Full vectors
are retained in `shard*/full_logits/`; intervention metrics and hashes are in
`shard*/checkpoints/`. Full intervention vocabulary vectors are not stored.
