# Physical Visual Handoff: frozen pilot design

Output location: `/data/lk/ATE-worktrees/dual-gpu-physical-visual-handoff-pilot/vlm_research/results/physical_visual_handoff_pilot/`.
All new output is on `/data`, not `/data1`. No historical files or model weights are removed.

## Scope and dependencies

Single image, one answer, batch one, eager attention, native model weights and native visual front end.
LLaVA uses FP16 / Transformers 4.49.0; Qwen uses BF16 / Transformers 4.57.6, native mRoPE and DeepStack.
The native model blocks, norms, MLPs, Q/K/V modules and attention kernels remain installed and unmodified.
Changes are scoped context managers, restored on every normal/exceptional exit.

## Information-flow references

All non-native conditions prevent original P-1 and subsequent answer queries from attending to original image keys.
D0 permits all R writers in all layers. D1 permits all R writers below the cutoff; D2 permits only the true
question-token subset below the cutoff. D3 additionally zeros the early-layer answer attention updates.
R begins immediately after the final image token and ends strictly before P-1, retaining Qwen vision_end.
Qwen question positions and dynamic image positions reuse frozen native processor bindings. LLaVA question
positions use offsets from the exact native processor rendering, checked against expanded input token IDs.

## Physical implementation

At LLaVA layer 20 / Qwen layer 22, remove image hidden rows and index-select the existing rotary cos/sin,
position IDs and original logical positions. All remaining blocks therefore receive fewer QKV/MLP rows.
DeepStack injections occur in the early Qwen layers and are executed using the native injection method;
the cutoff must be after every actual injection.

CompactKV retains separate original position arrays and physical KV tensors per layer. Generation always
advances the original logical length, never the physical cache length. Removing image KV uses index_select
to allocate compact owning tensors; weak references audit disposal of original tensors. No crop of a suffix,
renumbering, globally shared caches, hidden disk feature caches or retained full-cache copies are used.
P2 releases image KV after prefill. A separate exact-logit gate tests immediate per-layer release.

For P4, skipped early layers compute prefill attention on all rows except the final P-1. The final query and
its causal key are not needed by earlier queries. An exact zero update is appended for that row. All future
decode calls return a zero update without calling Q/K/V, QK, softmax, AV or o_proj. These layers never retain
persistent KV. Their norms, MLP and residual updates still execute. The physical path does not compute and
then zero the skipped answer output. Qwen six-layer bypass is a separately gated exploratory candidate.

## Acceptance and measurement

No-op uses exact all-returned-logit hashes in matched LM-head shapes. Shape-changing gates are frozen in
common.py before GPU execution; FP16 uses the existing physical-exit thresholds, BF16 uses explicit larger
quantization-aware thresholds, plus retained-text hidden relative RMSE/cosine. Mean/max relative errors are
reported, not used alone near zero. Greedy outputs must agree for the paired functional references; any
unexplained divergence stops expansion, with diagnostics retained. These gates do not establish mathematical
equivalence for arbitrary inputs.

60 preselected questions precede the remaining 180, six balanced tasks, 240 unique image SHA clusters.
Smoke images are disjoint from the quality cohort. No correctness-based selection. Whole-image paired
bootstrap uses 20,000 draws, fixed seed, exploratory ordinary intervals.

Benchmarks use fresh visual encoding, batch one, same eager backend, GPU event timings, five warmups and
20 measurements for fixed generated lengths 16/64/128, with median/p10/p90. Forced post-EOS tokens are
performance-only and never included in accuracy. The two model benchmark sections use an exclusive lock.
Profiler summaries retain operator shapes and counts rather than large trace dumps. Python cache management,
tensor indexing, synchronization overhead and CPU launch effects are part of the prototype, not a kernel
performance claim. Speedups must be compared to both Native and the matching dense functional reference.

## Stop boundary

No 3000-question experiments, training, new layer search, main modifications, other-process termination,
or historical-data deletion. Failed paths are partial, not successful optimizations. A small accuracy interval
including zero does not demonstrate noninferiority or safe multi-turn cache deletion.
