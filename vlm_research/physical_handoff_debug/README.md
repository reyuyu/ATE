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
CUDA_VISIBLE_DEVICES=GPU_UUID python -m vlm_research.physical_handoff_debug.run worker --kind llava
CUDA_VISIBLE_DEVICES=OTHER_GPU_UUID python -m vlm_research.physical_handoff_debug.run worker --kind qwen
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
