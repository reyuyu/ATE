# Reproduction and evidence audit

Final status: **PARTIAL_REPAIR**. The original fixed question passed exactly; the median smoke failed first at L22 `q_proj`. The longest smoke, formal Accuracy and E2E were **not run**. Do not continue strict-BF16 repair from this completed task.

## Portable CPU check

Run `python verify_evidence.py` here with NumPy. It validates the delivery SHA index, every archived logit array, recomputes full-vocabulary Max/Mean differences and checks the original exact F/C match. No GPU is opened.

`arrays/full_valid_logits_bf16_bits.npz` stores every vocabulary entry at every actual generated step for rank0 Native, Noop, F, B, C-before, C-after, W22-functional and P-after, for both completed questions. Values use exact int16 BF16 bit patterns; uint16->uint32 left shift16 then reinterpret float32 reconstructs the archived valid vectors. Conversion to BF16 was explicitly verified to round-trip the original float32 archive with `torch.equal`. It is not a lossy numerical compression. Rank1's complete vectors remain on the server, were compared bitwise against rank0, and are indexed by SHA.

The verifier's float64 CPU aggregate mean may differ from Torch's recorded float32 reduction in its last bits; the check allows less than1e-7 aggregate discrepancy only. Actual model equality and the0.02/0.5 gate have never been relaxed.

## Frozen run

Execution source: `19de4599c46f24c46caefbcebce643e72bfeafb1`, with every imported experiment source bound in `source_manifest.json`. Base: `0fe03aa1bb1923aee8b234ab3dccf59d7b21deb3`.

Model and processor: `/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

Inference Python: `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`, Torch2.6.0+cu124, Transformers4.57.6, BF16/Eager. No dependency install or shared environment modification.

Worktree: `/data/lk/ATE-worktrees/qwen3vl-c3v2-prefill-av-final`.

Private tensors: `/data/lk/ATE-resources/experiments/qwen3vl_c3v2_prefill_av_final/` (task-owned directory mode0700). The parent frozen F probe and fixed Lt bridge are read-only reused server resources with hashes in the audit. They are not included in Git. Old source/results/weights were never written.

The fixed order was `06905471`, `02497673`, `02287196`; the worker stops immediately at a failed smoke. Rank0 completed only the first two. Rank1 read the frozen completed-case list and reproduced those two, with no candidate selection.

GPU3 `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185` ran the source module `vlm_research.qwen3vl_c3v2_prefill_av_final.run 0`; GPU2 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6` then ran `...run 1`. Each process had only its UUID exposed in `CUDA_VISIBLE_DEVICES`. The GPUs are now released. Any future GPU work needs fresh applicable authorization and occupancy checks; these are historical commands, not authorization to run them again.

Each condition independently allocates its own cache. Native/Noop/HF and read-only Probe comparisons are strict. Late Prefill records are aligned by original absolute query/key positions, including P−1. QK statistics exclude the intentionally dead image columns; their actual Softmax mass is independently checked as exactlyzero. The original pre-repair layer trace is audited reuse after strict replay of its full valid-logit vectors; median traces are newly measured.

## Analysis and plotting

The CPU module `...prefill_av_final.analyze` requires the external raw tensors. It consolidates the layerwise CSV, full vectors, physical dimensions/costs, first divergences and release audit. Use a new review copy to preserve this completed result.

`plot.py` consumes the delivered CSV only. It was run using `/data/lk/深度学习/.envs/ate-vlm-research/bin/python` with `CUDA_VISIBLE_DEVICES=''`. PNG and vector PDF are both included.

`cpu_tests.txt` records36 tests:9 new padding checks and27 inherited numerical/position/cache regressions.

## Cost interpretation

`prefill_costs.csv` reports every layer's actual Q/K/V shape, matrix QK+AV FLOPs including zero columns, new allocations and single-call GPU-event duration. `physical_cost_summary.csv` also records repeated GQA K/V buffers and whole-request allocated peaks. Allocation components overlap in lifetime and must not be blindly summed into peak memory. Timings exclude the full padding-setup cost, have no repeated confidence interval and can contain host gaps. They are not E2E benchmarks or evidence of speedup. Exact tensor-core instruction FLOPs remain unmeasured; the inherited fixed-Lt tile-tail limitations still apply.
