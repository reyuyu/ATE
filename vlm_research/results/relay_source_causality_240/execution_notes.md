# Relay source causality: frozen execution plan

Base: `8019c6009bb1d8c9f2dc3ffd8c18bf4dfb8c1fd1`, containing formal experiment
`8d12f8c05f2b246d029ec4d9ffe45d1f857fa947` and the completed I+A study.
Branch/worktree: `codex/relay-source-causality-240`,
`/data/lk/ATE-worktrees/relay-source-causality-240`.

Only the new module and new results directory are changed. Historical source,
results, checkpoints and model files are immutable. Existing Handoff controls
operate all 32 layers, while source-knockout controls target P-1. Neither is
equivalent to L2-L5 / R-only source interventions. Exact new condition names were
also searched across the tracked repository; no existing task/condition found.

Main: reuse audited Full and R-Bypass (R-All), 240 each; generate exactly 240
R-Image-OFF and 240 R-Text-OFF. Reuse manifest and input mapping bytes. The task's
59-character hash is a transcription error; actual frozen SHA256 is
`258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c`.

Mask rules: at only L2-L5 and original R queries, clone the native additive
causal mask and set selected currently legal edges to -inf. Image-OFF blocks
original image keys. Text-OFF blocks k<q and non-image, excluding native BOS.
BOS and Self remain legal. Other queries, P-1, decode, other layers and native
weights/QKV/MLP/residual/KV are untouched. Source hard masks renormalize; they are
not additive-strength counterparts of full output zeroing.

Actual eager post-softmax weights are inspected at the four target layers on
each generation. The attention prehook requests weights without changing the
model's output shape; the after-hook audits forbidden probability exactly zero,
source mass, and the historical row-sum bound (0.002), then releases matrices.
Full/no-op are compared strictly with torch.equal at logits_to_keep=1.
No installed library or historical module is modified.

Controls: copy the exact frozen Handoff image_swap_manifest.json (40 pairs),
verify equal text IDs, different image SHA256 and different official gold.
Generate Full/Image-OFF/Text-OFF at both endpoints (240 control generations).
Historical swap Full lacks explicit LM-head keep1, so recompute the matching
Full rather than mix paths. Full-vocabulary FP64 JS, answer change and both-end
correctness are separate metrics. No pair reconstruction or output selection.

No existing audited text QA task was located by tracked-file search. Freeze
first two primary questions per task (12 total) with text-only chat prompts.
Question-only Native and text-prefill output bypass generate 24 diagnostic
answers, with no gold/accuracy. Scope is all non-BOS original text queries before
P-1; P-1 and decode native. This is not a text QA capability benchmark.

Run CPU tests and baseline replay with CUDA hidden. Commit source/CPU acceptance,
then freeze full model SHA256, all source/dependency/input hashes, controls and
comparison definitions. Select one truly idle GPU and recheck before model load.
One model, all work serial. Three original fixed smokes must pass before formal
collection: native/full/noop strict equality, historical Full/R parity, source
probabilities, all32 MLP, real independent cache, input/parameter guards, and
normal/exception hook cleanup. Any numerical failure saves PARTIAL and stops.

After the process exits, CPU analysis replays all records and raw arrays,
recomputes all scores, cache/source events, swaps and diagnostics, performs shared
20,000-draw image-cluster bootstrap (seed20261102), verifies full model hashes,
and records GPU release. `analyze --check` repeats read-only saved-statistic and
artifact checks. Raw arrays remain outside Git at
`/data/lk/ATE-resources/experiments/relay_source_causality_240/`.

No physical skipping, information-contribution fractions, equivalence/NI,
cross-model work, training, larger datasets, kernel or KV pruning.
