# QWEN3VL_SHORT_ANSWER_PROMPT_ABLATION

This is an independent output-instruction experiment, not a continuation or
replacement of the original ATE unified-HF scores.

Frozen code commit before GPU execution: `48b8616965`.
Frozen algorithm tree (read-only):
`/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090`,
commit `78cbd30d0b322d2dc5e1352d1d9b7d2dcd5ab62a`.

The Qwen SVH algorithm remains exit-before-L22, answer Attention bypass L1-L6,
no direct original-image access for P-minus-1 or Decode, unchanged DeepStack,
mRoPE, BF16, eager attention and original pixel budget. Historical strict
Dense/Physical failures remain `FAIL_RETAINED`. Wrapper smoke checks cannot
promote that status to a strict algorithmic numerical PASS.

## Reproduction

Run from this worktree with the existing environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
PY=/data/lk/深度学习/.envs/ate-qwen3vl/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest test_prompts -v
CUDA_VISIBLE_DEVICES='' "$PY" freeze_prompts.py
CUDA_VISIBLE_DEVICES='' "$PY" orchestrate_prompts.py
```

Freezing refuses an existing protocol. The controller exposes only physical
GPU1 and GPU3 to its two children after the children independently check that
their authorized GPU is empty. Existing tasks are never stopped. The workers
retain one model each through Discovery, frozen selection, Validation and the
small separate performance measurements. Only this controller's own children
can be terminated on an error. The controller has a three-hour execution limit.

All outputs are under
`vlm_research/results/qwen3vl_short_answer_prompt_ablation/`.

## Split and Scoring

The stable image-SHA partition is global across all three tasks. Each stage
contains 150 GQA questions, 150 ScienceQA questions and 150 complete MMBench
Circular groups. Discovery has 566 MMBench rotated rows; Validation has 552.
No image cluster crosses Discovery/Validation, and no Circular group is split.
These remain historical research-cohort images, not independent external data.

P0 quality records are reused only after exact question/gold/image/prompt/
expanded-token/tensor/model/generation/protocol binding checks. P0 performance
is measured afresh in the dedicated microbenchmark. P1/P2 add only the frozen
output instructions. No constrained decoding or one-token output cap is used.

The Discovery-only selection rule is in protocol.json. Validation never changes
the selected prompt. MMBench is all-rotations-correct Circular accuracy;
question-level ScienceQA strict scoring and GQA exact match are unchanged.
The prior conservative ScienceQA parser is a separately labelled diagnostic,
with truncated responses unresolved. Every confidence interval resamples image
clusters and uses 20,000 replicates. All inference is exploratory.

## Runtime and Storage Notes

Performance probes are selected by input hash before any new outputs, one per
task per GPU. Each condition has one warmup and three measured repeats, both
natural EOS and a separate exact-32-token timing mode. Only the latter ignores
EOS for a fixed-budget comparison; it is never used for quality scoring.

The data partition had about 1.1 GiB free at start. This experiment stores no
models, images or large tensors, uses compressed private SQLite checkpoints,
and stops below 512 MiB free or above a 180-MiB output budget. The original
experiment's 2-GiB stop policy was not changed or bypassed to restart that task.

An initial CPU freeze attempt failed because the new output directory did not
yet exist. The private Store now creates its own parent directory; this was
fixed before protocol freeze or GPU use. No model, algorithm or numerical
threshold changed. CPU regression tests subsequently passed.

No automatic remote push or main-branch modification is part of this task.
