# Reproduction and evidence scope

Outcome: `PARTIAL_REPAIR`. Only the original fixed GQA input06905471 was run. No additional smoke, Accuracy or E2E benchmark was authorized by a passed full-C gate; all three full-C candidates failed. The two GPUs repeat the same input, not independent dataset samples.

## Portable CPU checks (no model or GPU required)

From a checkout containing the delivered result directory, use Python with PyTorch and NumPy:

```sh
CUDA_VISIBLE_DEVICES= python vlm_research/results/qwen3vl_c3v2_numerical_repair2/reproduction/offline_check.py
```

This verifies26 complete paired-logit rows and saved first-divergence evidence, plus Native/No-op and cross-GPU strict vector equality. Effective logits retain the entire vocabulary and both real steps, including EOS. BF16 first-difference arrays are losslessly encoded through int16 bit reinterpretation; metadata specifies original dtype, shape and logical positions.

## Exact historical sources

- Phase1: `788cae76233a80af744ea044f408a6d8cc52084c`.
- Phase2: `77465e4e3a3c83c13fb274318afc431f7e6b01d2`.
- Parent evidence: `fe142bf2a5e0187e352591c699b4e977c40cb41c`.

Per-file SHA bindings are in source_manifest.json and source_manifest_phase2.json. The latter probe assigns transient zero-image columns distinct negative logical IDs, rather than mislabeling them real image KV. Phase1 used the earlier unchanged probe source; manifests must be checked against their own Git commits, not all against the latest source.

## Server CPU reconstruction

Worktree `/data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-repair2`; inference Python `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`; plotting Python `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.

```sh
cd /data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-repair2
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest vlm_research.qwen3vl_c3v2_numerical_repair2.tests vlm_research.qwen3vl_c3v2_numerical_debug.tests vlm_research.qwen3vl_c3v2_numerical_debug.repair_tests
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python vlm_research/results/qwen3vl_c3v2_numerical_repair2/reproduction/cleanup_cpu.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python vlm_research/results/qwen3vl_c3v2_numerical_repair2/reproduction/finalize_cpu.py
CUDA_VISIBLE_DEVICES= /data/lk/深度学习/.envs/ate-vlm-research/bin/python vlm_research/results/qwen3vl_c3v2_numerical_repair2/reproduction/plot_cpu.py
```

The scripts retain original absolute server paths for auditability. finalize_cpu verifies external raw SHA, cache contents/prefixes, real execution shapes, source Git blobs and previous artifacts; it writes fresh validation/resource timestamps locally. It never launches GPU inference. CPU cleanup uses a tiny random36-layer CPU Qwen to exercise real normal/exception patch restoration, not a dataset evaluation.

## Historical GPU invocation (do not automatically rerun)

```sh
# Only after explicit current-task permission, idle checks and source freeze:
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_c3v2_numerical_repair2.phase1 0
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_c3v2_numerical_repair2.phase2 0
CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 PYTHONPATH=. /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_c3v2_numerical_repair2.phase2 1 verify
```

These historical commands have frozen absolute cutoffs and must not silently resume or expand work. GPU3 was used first and released before GPU2 independent replication. GPU0/1 were untouched. Native references remained BF16 throughout. FP32 candidates changed only temporary K/V/gate/up compute after physical exit, never model parameter storage or logits postprocessing.

Large raw PT archives remain outside Git at `/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_repair2/`; raw_tensor_index.json gives exact paths, sizes and SHA. Archives contain record-level tensor shapes/dtypes and source binding. Arrays in Git support portable numerical checking, but do not substitute for claiming every full raw archive is hosted by GitHub.

`actual_matrix_shapes.csv` is the canonical shape record: it accounts for temporary zero columns. Original engine observers see pre-padding inputs and are retained separately in physical_execution_audit.json. Decode key restoration adds QK/AV work; no persistent dummy KV or skipped-layer attention is reintroduced. Single-run local FP32 GPU-event sums and inherited worker timing fields are diagnostic overhead observations, not a controlled latency benchmark or speedup result.
