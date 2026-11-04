# Physical visual handoff numerical debug

This bounded follow-up does not overwrite the original PARTIAL prototype.
Only D1/P1/P2 are measured, with the original cutoffs and numerical thresholds.

## Reproduce

Run from this worktree, on `/data`. Use the historical LLaVA and Qwen environments
respectively. Verify GPU availability first; CUDA_VISIBLE_DEVICES must expose just
one independently reserved, idle GPU per worker.

```bash
python -m unittest vlm_research.physical_handoff_debug.tests -v
python -m vlm_research.physical_handoff_debug.run prepare
CUDA_VISIBLE_DEVICES=GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 python -m vlm_research.physical_handoff_debug.run worker --kind llava
CUDA_VISIBLE_DEVICES=OTHER_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 python -m vlm_research.physical_handoff_debug.run worker --kind qwen
```

Freeze and commit source before `prepare`. The protocol records source hashes and
the commit; do not edit these modules while workers are running. Each model has
an independent output directory. Native operator traces reconstruct actual eager
attention and require bitwise agreement with the installed implementation.

## Candidate kernels

`native` uses unchanged installed kernels. `fp32_linears` changes accumulation in
linear layers from the existing cutoff onward, and in the LM head, then casts to
the original activation dtype. `fp32_linears_attention` additionally changes the
two attention matrix products after the cutoff to FP32 accumulation, preserving
the native low-precision output boundary of each product. These are temporary
per-operator casts, not whole-model FP32 conversions. Model parameters stay frozen.
Apply each candidate to BOTH dense and physical paths; compare original native
diagnostics separately. No answer-dependent tie breaking or regenerated visual
rows is allowed. The first passing candidate must also pass fresh regression.

After the initial three candidates, the bounded fourth repair `canonical_tiles`
uses native-precision 32-row linear tiles (at most 31 zero padding rows, never
reconstructed visual rows), and evaluates the exactly disconnected text/visual
attention subgraphs separately after the cutoff. Dense and physical text kernels
then see the same Q/K lengths. The dense reference still computes its visual
rows, while the physical path never restores them. Decode uses the actual
per-layer original cache positions, not a guessed compact sequence length.
This amendment follows the recorded same-input Q/K/V shape diagnosis; the
original three candidate failures remain in the log. Neither thresholds nor
functional intervention conditions change.

The 10 fresh questions are selected by input hash, disjoint from all original
240 pilot images and smoke images. The historical 43-question tie audit performs
one prefill per path, not a new quality matrix.

Performance measurements use real CUDA events and synchronized wall time,
2 warmups and 5 repeats at fixed 16/64 generated positions (15/63 incremental
Decode steps). Fixed length deliberately ignores EOS only in this microbenchmark.
Normal correctness runs stop at EOS. A failed equivalence path is always labeled
`UNVALIDATED_PERFORMANCE`; throughput alone cannot establish correctness.

Failures leave tracebacks and earlier checkpoints. Never resume an old protocol
under modified source; record a new source revision and new attempt explicitly.

The CPU-only finalizer is outside the frozen worker package:

```bash
python -m vlm_research.physical_handoff_debug_finalize
```

Read the ROOT validation/report when deciding validity: model-local worker PASS
means matched-kernel regression only, NOT original-native parity. The finalizer
checks native-to-fixed drift separately and downgrades all performance labels
when that stricter parity gate fails. The 99 raw checkpoints are preserved in
`raw_checkpoints.tar.gz`; the artifact hash manifest covers archive and raw files.
The full per-layer position/storage audit JSON is preserved separately in
`raw_kv_audits.tar.gz`; readable summaries stay in `kv_memory_validation.csv`.
