# Token-Dependent Decode Attention Necessity

Frozen LLaVA-1.5-7B FP16/eager/Transformers4.49.0; COCO200 images.
No training or free continuation after an intervention. Original model/prompt and
historical sources/results remain unchanged. Source freeze covers GPU modules and
their shared helpers; CPU-only analysis is separate and cannot alter collection.

Run from this worktree's `vlm_research` directory with the pinned Python:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/data/lk/深度学习/.envs/ate-vlm-research/bin/python
$PY -m paper_rq1_token_role_decode.data
$PY -m paper_rq1_token_role_decode.run cpu
# Commit all final collector sources, test and data before this freeze.
$PY -m paper_rq1_token_role_decode.run freeze
$PY -u -m paper_rq1_token_role_decode.launch
$PY -u -m paper_rq1_token_role_decode.analyze
```

Resume using the same launch command; completed shards and valid per-image
checkpoints are reused, no sample silently skipped. Never rerun data preparation
to replace the frozen sample. If source/input bindings differ, stop and inspect;
do not edit executing worker code. Failed shards retain explicit failure records.
The launcher waits for every shard before advancing; it never terminates other
processes. Smoke is a separate image. Full200 generation finishes, alignment and
targets freeze centrally, then formal ablation begins. Main types use strict
single-token word alignment and fixed lexicons; all exclusions are auditable.

Results live in `results/paper_rq1_token_role_decode/`; large native/altered logits
NPZ files live in `/data/lk/ATE-resources/token_role_decode/raw/`, bound by SHA256.
Every target uses the original generated history. Current target y_t is predicted
by consuming y_(t-1), never a teacher-forced y_t. Image intervention subtracts
the original image PV contribution without changing the softmax denominator.
Projection bias stays in the unmodified remainder. Image PV and Wo accumulation
are FP32, followed by one cast of the changed output to FP16. Smoke checks a
predeclared propagated rounding-error bound, not an adjustable empirical atol.

Primary role comparisons average the nine measured layers only, with image-level
pairing and bootstrap; controlled analysis adjusts original confidence/entropy
and generation position. Constant main-analysis subword length=1 is explicitly
not estimable. No interpolation to unmeasured layers, no final accuracy or speedup.
