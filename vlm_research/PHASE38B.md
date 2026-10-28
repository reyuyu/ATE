# Phase 3.8b C2 Confirmation

Branch: `exp/phase3_8b-c2-confirm240`, based on Phase 3.8 results commit
`2415b69264d20c5da1c47c18bdb0b5adcafa69a7`. Never modify main or older results.

All original Phase3.2 manifest images, unique question/image IDs; Discovery16
comes only from the actual Phase3.8 manifest, Confirmation224 is its complement.
Primary inference uses Confirmation224 (112 left,112 right), not pooled240.
This is a frozen preselected upstream cohort, not an unbiased GQA benchmark.

Only two real conditions: C0 unmodified Full, C2 zero-based layers2/3/4,
actual prompt P-1 MLP output zeroed before residual addition. The exact original
`phase38_sink_hooks.zero_query` is reused. No Sink/ROI labels are required or
fabricated. All Attention modules, all tokens/positions and32layers retained.
Later P-1 Attention changes naturally; future answer-token hidden states need
not be unchanged. Native observer exactness and post-C2 restoration are checked
for BOTH candidates. All earlier positions (including image tokens) are hashed
at every layer. No model or installed-package source edits.

Offline B1 only adds1.0 to Full right-minus-left score. It is never a GPU
condition or a genuine full-vocabulary absolute LogP estimate. No bias fitting.

Frozen paired bootstrap: NumPy default_rng seed382402026,10000 image-level
paired replicates, percentile95%CI, separate left/right resampling. Exact
two-sided McNemar on paired discordants. No multiple-testing adjustment;
secondary/group results and discovery/pooled intervals labeled accordingly.
All tails/ties retained; margin>0 only. No condition or sample expansion.

## Execution

Commit all implementation/test/summary code before `prepare`. Fixed interpreter:
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Commands run from `vlm_research/`:

1. CPU-visible `CUDA_VISIBLE_DEVICES=''`: `phase38b_c2_confirm.py prepare`.
2. CPU-visible `CUDA_VISIBLE_DEVICES=''`: `phase38b_c2_confirm.py cpu`.
3. GPU1-visible `CUDA_VISIBLE_DEVICES=1`: `phase38b_c2_confirm.py smoke`.
4. GPU1-visible `CUDA_VISIBLE_DEVICES=1`: `phase38b_c2_confirm.py run`.
5. CPU-visible `CUDA_VISIBLE_DEVICES=''`: `summarize_phase38b_c2.py`.

Smoke fixed IDs02474907left and00784767right must reproduce old C0/C2 logits
BITWISE. All16 discovery images repeat this strict regression during formal run.
GPU UUID is checked before loading. Never kill another task. Every checkpoint
binds code/config/data, all model-file hashes, actual loaded-state hash,
environment and physical GPU. Native Full is always computed on this machine.
Historical RTX4090 discrepancies are retained as diagnostics, never substituted.
Complete checkpoints are immutable. Incomplete attempts are preserved and halt
for explicit investigation. Resume on the frozen code commit only; archive
result-commit hashes separately. CPU and smoke gates precede formal execution.

stdout must be logged OUTSIDE the recursively hashed result directory until
the process exits, avoiding any live-console self-hash. Summary has a single
successful finalization gate and never overwrites it. Publish real results on
the experiment branch only; do not automatically merge. Release model process
after execution. No training, new layers, conditions, datasets, decode or exit.
