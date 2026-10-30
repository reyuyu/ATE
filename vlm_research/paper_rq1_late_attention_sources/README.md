# Late-layer Attention Source Decomposition

Independent new200-image confirmation/extension of Observation2, not a reanalysis
of its selected examples. Previous collector and labeling sources are unchanged.
Frozen10 layers; seven source/bypass modes only as specified in protocol.py.
No training, free continuation, mask renormalization, kernel bypass or speed claim.

From this worktree's `vlm_research` directory:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/data/lk/深度学习/.envs/ate-vlm-research/bin/python
$PY -m paper_rq1_late_attention_sources.data
$PY -m paper_rq1_late_attention_sources.run cpu
# Commit exact tested collector, tests and samples before freeze.
$PY -m paper_rq1_late_attention_sources.run freeze
$PY -u -m paper_rq1_late_attention_sources.launch
$PY -u -m paper_rq1_late_attention_sources.analyze
```

Resume with the same launch command; never resample or replace frozen targets.
User authorized GPUs1/2/3 after initial GPU0 resource blockage. Frozen images
and four50-image logical shards are unchanged: GPU1 executes shards0/3 in
sequence, GPU2 shard1, GPU3 shard2. Original Full and intervention always use
the same physical GPU. Old Smoke was archived; new binding reruns Smoke.
Each stage gates all four complete same-code logical workers. Each worker independently
loads one model on its designated UUID, checks idle status and writes per-image
atomic checkpoints. The coordinator freezes target positions after all Full
captions and before any formal ablation. Changed code, data, cache alignment,
nonfinite values or failed numerical reconstruction stop execution. Never edit
running worker code or loosen the frozen rounding bounds.

Every intervened current Query consumes the previous native-generated token and
uses a fresh deep clone of the same Full history. Native observation is a separate
unchanged forward that collects all10 layers at once. Original image/nonimage
decomposition bounds are retained; four-source and history-split sums add only
explicit FP32 summation/projection error bounds. Bias is not counted per group.

Results: `results/paper_rq1_late_attention_sources/`.
Raw FP32 full-vocabulary NPZ: `/data/lk/ATE-resources/late_attention_sources/raw/`.
Raw paths and SHA256 are bound in checkpoints; model weights/images are not Git
artifacts. All three lexical roles are inherited candidates, not gold grounding.
Statistics and primary41-contrast family are frozen in protocol.py before GPU
output. Image-cluster Bootstrap, same-image controls, signed and absolute effects
are required; LogP loss is not full-answer accuracy or additive source attribution.
