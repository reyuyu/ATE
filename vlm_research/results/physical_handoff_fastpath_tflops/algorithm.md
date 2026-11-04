# Native-Precision Physical Visual Handoff Fast Path

## Frozen graph

LLaVA: FP16/eager, 32 language layers, R writing L0--L19,
trim original image Hidden rows before L20.
Qwen3-VL: BF16/eager, 36 language layers, R writing L0--L21,
trim before L22; native mRoPE and early DeepStack unchanged.
All original P-1 and incremental Decode queries in the handoff conditions
cannot directly access original image Keys. R is the original nonvisual span
after the last actual image Token and before P-1, including Qwen vision_end.
There is no question-only selector, attention bypass, or new cutoff search.

## References and contrasts

- Native Full: unchanged native blocks and native DynamicCache, all image access.
- D1 Dense: same writing dependency as the physical paths, but no Hidden trim
  or visual KV removal. Masks precomputed; in-place preallocated dense KV.
- Fast-P1: D1 dependency plus true Hidden row removal at the frozen cutoff.
  Before the cutoff KV remains dense; after the cutoff KV is text-only.
- Fast-P2: Fast-P1 plus direct nonvisual gather into final persistent KV storage
  during Prefill. Original visual K/V are transient inputs to the current native
  Prefill Attention, never full persistent cache entries. Decode is text-only.

The original debug engine is imported only during Smoke to establish D1 parity.
Original diagnostic FP32 linears and canonical tiles are never enabled.
Historical native-equivalence PARTIAL conclusions remain intact.

## Hot path

One CPU layout pass after the frozen Processor creates actual image, text,
question and R indices. Masks and all logical cache positions are prepared
once per request. The fast loop calls unchanged native Decoder blocks. No
torch.isin, key search, per-layer source-mask construction, hook auditing,
canonical tiles, or explicit per-layer synchronization occurs in production.

Each layer owns preallocated K and V buffers of physical Prefill length plus
max_new_tokens - 1. Append uses copy_ into a single new slot and returns a
prefix view, never torch.cat or a complete historical-cache copy. Fast-P2
uses index_select with an out view of the final cache, not full persistent
visual KV followed by deletion. Buffer capacity and live used KV are reported
separately. No cross-request Cache reuse is hidden from timing.

Cache get_seq_length reports the original logical sequence length, not the
physical length. Original position_ids/cache_position and rotary cos/sin are
indexed, never renumbered. Decode Keys are all causal past or current Self;
the Fast-P2 Decode attention mask is None (no visual keys exist). Dense paths
slice a precomputed source mask. DeepStack is applied at its native early
layers before the physical cutoff.

## Timing and quality

The exact historical pilot60 is reused, 10 questions/task, no outcome selection.
Each input/condition has 5 warmups and 20 timed repetitions. Conditions are
interleaved and rotated by repetition to limit ordering/clock bias. Cold
per-request mask/cache setup is included in synchronized E2E; image decoding
and CPU Processor preprocessing are excluded equally. The actual native
vision encoder is rerun in every request (no feature cache).

CUDA Events record model/vision/Prefill/Decode durations. E2E synchronized wall
time additionally includes Python/request setup and EOS checks. Natural answers
stop at EOS; fixed16/64 are explicitly engineering workloads. GPU predictions
remain on device in fixed mode; only natural EOS performs the necessary host
token check. Detailed audits, CPU logit copies, FLOPs census and profiler
instrumentation execute in separate passes, never measured latency repetitions.

Original logits_to_keep is retained (LLaVA=0, Qwen=1). Native HF generate parity
and matching-shape native/no-op torch.equal are required in GPU Smoke. All
numerical thresholds are inherited unchanged. Physical approximation is
reported independently of semantic correctness and accuracy; no 60-question
noninferiority claim is permitted.

## Arithmetic and hardware provenance

An independent instrumented shape census counts executed nn.Linear, Conv,
and torch.matmul QK/AV work, 2 FLOPs/MAC. It includes native vision, QKV, MLP,
attention projections and LM head, but excludes non-matrix operations such as
activation, normalization, softmax and rotary. Stage totals divided by the
corresponding unprofiled CUDA Event time are Estimated Algorithmic TFLOPS,
not hardware-achieved throughput.

Nsight Compute's actual metric query returned ERR_NVGPUCTRPERM. Hardware
floating-point counts, Tensor Core/SM throughput, DRAM bytes/bandwidth and L2
metrics are unavailable; no percentages or theoretical peaks are converted
to hardware TFLOPS. Separate PyTorch/CUPTI profiles provide kernel activity,
CPU launch overhead and profiled KV update durations only. Counter-based
roofline/bottleneck classification is explicitly unavailable.

## Reproduce

Use the exact per-model Python paths in protocol.json and set
CUDA_VISIBLE_DEVICES to the selected idle GPU UUID, CUBLAS_WORKSPACE_CONFIG
to :4096:8, OMP_NUM_THREADS/MKL_NUM_THREADS to 4, HF_HUB_OFFLINE to 1.

```bash
python -m vlm_research.physical_handoff_fastpath.tests -v
python -m vlm_research.physical_handoff_fastpath.run freeze
python -m vlm_research.physical_handoff_fastpath.run all --kind llava
python -m vlm_research.physical_handoff_fastpath.run all --kind qwen
python -m vlm_research.physical_handoff_fastpath.reporting.delivery --native-kv-profile llava
python -m vlm_research.physical_handoff_fastpath.reporting.delivery --native-kv-profile qwen
python -m vlm_research.physical_handoff_fastpath.analyze
python -m vlm_research.physical_handoff_fastpath.reporting.delivery
python -m vlm_research.physical_handoff_fastpath.reporting.delivery --seal-only
```

Each model runs on one independent GPU. Checkpoints are atomically replaced
per sample after all four conditions and 20 repetitions succeed. Resume accepts
only identical source/protocol/sample hashes and complete checkpoints. No
failed sample is silently skipped. Hardware semantic or OOM failure stops
the corresponding worker and saves failure.json. No main/history modifications.

The per-model KV microprofile runs only after the benchmark workers finish and
uses the same lightweight CUDA-event wrapper for all four cache implementations.
It is never mixed into latency repetitions. The reporting subpackage is separate
from the frozen GPU-worker source and has its own delivery hashes. Run the final
seal without redirecting its stdout to an artifact that is being hashed.

Large profiler traces are committed as lossless gzip with original/archive SHA256
in trace_archives.json. Raw traces stay local, and can be restored with gzip -dk.
