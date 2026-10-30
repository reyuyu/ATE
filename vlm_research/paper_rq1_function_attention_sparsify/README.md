# Function-Token Identification and Attention Source Sparsification

Independent branch `codex/function-token-attention-sparsify`, based on
`6964407e6b4c1291cb0ca7c41128947d3025520d`. No previous collector/results modified.

## Frozen Design

- Stage A: both previous COCO200 cohorts, all previously selected targets,
  single layers14/26/31. A1 jointly subtracts original-denominator projected
  Value contributions; A2 gathers retained cached K/V **before** QK/AV and
  renormalizes softmax. Whole-off is a shared zero-update control.
- Stage B: all native Decode positions, not just selected words. Input-hash
  image split300train/100calibration. Prefix logistic, pre-L31-hidden linear
  logistic, combined. The VLM is never trained. Explicit negative categories
  include EOS, special, ambiguous, unknown, multi-token and unaligned positions.
  Position divided by the fixed cap64 is online-valid; final length is forbidden.
- Stage C: new200 COCO images locked before GPU collection, identity/byte/RGB
  isolated from both cohorts and previous COCO Discovery. Greedy free generation,
  normal full-sized KV Cache; each policy uses its **own** actual prefix.
  Failed source/classifier gates stop corresponding dynamic strategies.
- Physics GPUs1/2/3 only, deterministic balanced image shards; GPU0 is occupied
  by unrelated jobs and is never acquired. Native and all conditions paired
  within the same GPU. CUDA-visible UUID is checked before model loading.
- Main quality uses official pycocoevalcap PTBTokenizer/CIDEr/SPICE, ordinary
  image-paired10000 bootstrap, frozen raw margins.02CIDEr/.01SPICE. Confidence
  intervals crossing0 do not establish equivalence or rare-harm safety.
- Real latency: fixed physical GPU1, first10 locked images,3warmups **per
  implementation**,5repeats, randomized condition order. No observer/profiler
  in primary timing. Separate event/profiler diagnostics. Model-generation
  wall time excludes image preprocessing and model loading; reported explicitly.

## Execution

From this worktree's `vlm_research`, using the pinned research Python:

```bash
python -m paper_rq1_function_attention_sparsify.data
# Commit collector, tests and sample manifest before freeze.
python -m paper_rq1_function_attention_sparsify.run freeze
python -m paper_rq1_function_attention_sparsify.launch
# Finite pipeline may start while StageA collects; it waits on its checkpoints.
python -m paper_rq1_function_attention_sparsify.pipeline
```

CPU tests: `test_function_attention_sparsify`, `test_function_predictor`,
`test_function_generation`. Every GPU process verifies its frozen sources,
model/data bindings and checkpoint identity. Failures are preserved, not skipped.

Raw logits/features live outside Git under
`/data/lk/ATE-resources/function_attention_sparsify/`, SHA-bound by checkpoints.
Small predictor weights, CSVs, manifests, logs, PNG/PDF and reports are committed.

## Interpretation

A1 already computes full Attention: it is not a speedup. A2 saves only L31
Decode QK/AV work, not K/V projection, full-cache storage or DynamicCache concat.
Gather/dispatch/classifier overhead can outweigh this limited computation.
Single-step native-token loss is not Caption accuracy. Lexical differences are
not verified grounding or hallucination annotations. Any final conclusion must
be based on the completed output report, not this prospective README.

The first rejected Smoke used a full-width FP16 QK reference against sparse-width
QK and failed the unchanged probability round bound. Its code/failure records
are preserved under `attempts/initial_reference_failure`. The independent
reference was corrected to use the same retained-column FP16 QK problem size
via a separate `bmm` implementation, scatter into a full masked softmax and
full-width AV. No rounding tolerance was enlarged; production sparse QK/AV did
not change. Formal collection starts only after the corrected Smoke passes.
