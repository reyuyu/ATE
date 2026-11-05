# Reproduction and evidence audit

This delivery is **PARTIAL_REPAIR**, not a released inference optimization. Do not launch Accuracy, E2E, or more smoke cases from these instructions. The fixed original input failed the unchanged complete-model gate.

## CPU-only portable verification

From this directory run `python verify_evidence.py` with NumPy installed. It opens no GPU, validates every delivery SHA, checks the exact BF16 operator excerpts, recomputes complete valid-logit Max/Mean errors, and checks bitwise equality between the two GPUs. NumPy's aggregate mean uses float64 to avoid Torch/NumPy float32 reduction order differences; this is only a CPU check of already archived vectors. The actual model gate and all tensor equality checks retain their original criteria.

`arrays/operator_excerpt_bf16_bits.npz` contains **all 26 text rows, first 32 output channels** per stored operator/variant, encoded as int16 BF16 bit patterns. It is a small excerpt, not the complete projection. Complete operator matrices and Chrome traces are on the server; see `raw_tensor_index.json` and each shard's tensor/trace indexes. `arrays/full_valid_logits.npz` contains **every vocabulary position at every actual generated step**, for both GPUs' F, B, W22-functional, P, C paths. These vectors are sufficient to independently recompute the complete-model gate.

## Frozen execution stages

- `source_manifest.json`: local operator replay/profiling source.
- `source_manifest_full_model_v1.json`: initial L22-only integration.
- `source_manifest_full_model_v2.json`: strict-equal layer-boundary guided continuation to L26.
- `source_manifest_full_model_rank0.json`: remaining-layer same-input gate followed by one fixed L22-L35 C test.
- `source_manifest_full_model.json`: identical numerical policy for second-GPU replication, adding bounded component evidence archival.

Each manifest identifies an immutable Git commit and source-file SHA. Source snapshots can be inspected with `git show COMMIT:PATH`; earlier full-model scripts are intentionally not identical to the latest archival script. No frozen previous source was overwritten in its own worktree. Later commits affect only this independent branch.

## Server resources and GPU prerequisites

Model: `/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

Python: `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python` (Torch2.6.0+cu124, Transformers4.57.6, BF16/Eager). Model files, processor, input SHA, source, dtype and operator stride are bound in the delivery.

New task resources: `/data/lk/ATE-resources/experiments/qwen3vl_c3v2_l22_kernel_diagnosis/` (protected task folder; not included in Git). Parent frozen input tensors: `/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_repair2/phase1_rank0/F_probes.pt` and `operator_shadows.pt`. No download is needed or performed. Do not overwrite any parent files or historical delivery.

Any future GPU reproduction requires its own applicable user authorization and a fresh `nvidia-smi` UUID/memory/process check. This completed task has released GPU3 `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185` and GPU2 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`. Never assume they remain idle.

Build the task-local cuBLASLt bridge with the exact argument vector recorded in `bridge_build.json` (use `subprocess.run(command, check=True)`, not a string shell). The source, resulting `.so`, header/library paths and hashes are recorded. No shared CUDA, PyTorch, cuBLAS, driver or system installation was changed.

Only the operator replay uses Torch Profiler. At the operator source commit, the commands were the module `vlm_research.qwen3vl_c3v2_l22_kernel_diagnosis.worker 0` on GPU3, then `...worker 1 verify` on GPU2 after freezing the policy. Each process exposes exactly its one UUID through `CUDA_VISIBLE_DEVICES`. Second-card `verify` runs the selected configuration only, not the rejected heuristic candidate.

The complete-model module is `vlm_research.qwen3vl_c3v2_l22_kernel_diagnosis.full_model`. Earlier versions were run at their recorded source commits before the final version. The final version deliberately requires stored earlier stage evidence and exact archived input/projection tensors; it does not silently manufacture missing prerequisites. The second-card command was `...full_model 1 verify`. No new input was run because the original failed.

## CPU analysis and figures

`python -m vlm_research.qwen3vl_c3v2_l22_kernel_diagnosis.analyze` recomputes analysis from the **server raw tensors** and updates indexes; it does not infer or initialize CUDA. It requires the external resources above, and should be run in a fresh review copy to preserve this completed delivery.

`plot.py` in that source module consumes the delivered CSV/JSON only. It uses NumPy/Matplotlib and was run with `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`, `CUDA_VISIBLE_DEVICES=''`. PNG and vector PDF pairs are provided. `cpu_tests.txt` records 27 inherited numerical/position/cache/dead-key regression tests newly rerun in this task.

## Interpretation limits

The fixed candidate uses BF16 inputs/output and explicit Lt configuration with FP32 compute descriptor, preserving the opaque library's own partial/reduction behavior. This does not establish the exact internal summation order. `splitKreduce_kernel` names and launch geometry are observed; complete instruction counts and exact issued tensor-core FLOPs are unmeasured without Nsight counters.

26 actual input/output rows do not imply 26 rows of hardware arithmetic. Gate/Up's fixed kernel/grid are identical to Dense and its local event latency increases roughly62%. `physical_flops_audit.json` distinguishes logical `2MNK`, tile capacity and added dummy-key arithmetic. No E2E speedup is claimed.
