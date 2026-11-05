# CPU evidence reproduction

The accepted scientific outcome is `ROOT_CAUSE_IDENTIFIED_REPAIR_FAILED`. These commands do not start new GPU inference. All thresholds remain max≤0.5, mean≤0.02, Top1 and complete tokens100% at every effective step.

## Portable checks from a Git clone

Install/use an existing Python environment containing NumPy and PyTorch (recorded environment: Torch2.6.0). No model weights, image files or GPU are required for:

```sh
CUDA_VISIBLE_DEVICES= python vlm_research/results/qwen3vl_c3v2_numerical_debug/reproduction/offline_check.py
```

This independently recomputes all12 paired effective-logit rows, strict Native/No-op equality and the first differing operators in the saved aligned BF16 tensor extracts. BF16 arrays are stored as exact int16 bit patterns; descriptors identify native dtypes and positions. The baseline has two steps (first answer and EOS). RMSE and mean are computed in float32, matching collection.

Original GPU0/1 were never used. Shard0 means GPU3 and shard1 means GPU2; these two runs independently repeat the SAME original input, not different statistical samples. Repair was only tested on GPU3. Detailed probe v1 had selected heads0/16/31; v2 covered all32 heads at L1/L22. Each source manifest binds the exact version used, rather than retroactively assigning the newest code to all runs.

## Server-only full archive checks

Existing worktree: `/data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-debug`.

Existing environment: `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`.

```sh
cd /data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-debug
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest vlm_research.qwen3vl_c3v2_numerical_debug.tests vlm_research.qwen3vl_c3v2_numerical_debug.repair_tests
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python vlm_research/results/qwen3vl_c3v2_numerical_debug/reproduction/verify_delivery.py
```

The second command verifies full model hashes, all external raw tensor hashes, source Git blobs, old artifacts, images, all12 rows, cross-GPU vector equality, compact first-divergence tensors, JSON/CSV parsing, PDF text/pages, disk reserve and current GPU release. It writes fresh CPU validation/resource audit records; running it after publication changes those audit timestamps locally. It never launches a GPU model. GPU release checks are observations at the recorded time, not reservations for future work.

`finalize_cpu.py` consolidates pre-existing private `.pt` files and writes aligned extracts/cache evidence. `cleanup_cpu.py` checks normal and exceptional context cleanup with a tiny random CPU Qwen model, not frozen data inference. `plot_cpu.py` reads delivered CSV/JSON only; run under the existing `ate-vlm-research` plotting environment with Matplotlib. These scripts contain the original absolute server paths for traceability.

## Frozen GPU sources — historical execution only

Diagnostic entry: `vlm_research.qwen3vl_c3v2_numerical_debug.run`; repair entry: `repair_run`. Source commits and actual GPU UUIDs are in the source manifests, environment records and protocol. The original input, exact IDs, hashes, layouts, positions and other unexecuted frozen smoke inputs are in sample_manifest.json. All GPU commands were subject to the frozen absolute cutoff and original-case stop gate.

**Do not resume or expand GPU inference as part of CPU reproduction.** The repaired original input failed; no other length smoke, Accuracy dataset, E2E benchmark or layer search was run. Existing result wrappers inherited incidental per-call timing fields from the original worker; they are not controlled benchmark measurements and were not analyzed as performance claims.

Large raw `.pt` tensors are external to Git under `/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_debug/`, with SHA256, byte size and path in raw_tensor_index.json. Small NPZ extracts and full effective logits are included in Git. Full model files were rehashed at CPU delivery verification; this is explicitly recorded as a post-diagnosis full-byte audit.

## Dummy key alignment

Repair adds one temporarily masked zero key per early attention layer. Length-based probe metadata initially labeled it original P−1. `repair_dummy_key_alignment.json` preserves the24 analysis-side corrections to a distinct dummy ID−1; raw arrays and model positions were not modified. Compare common real keys, not intentionally unused pre-mask scores.
