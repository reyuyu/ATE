# Original-P-1 single-layer answer-accuracy profiles

The only intervention zeros one language-layer `o_proj` output at the original
prefill's final query. All subsequent decoding is native. Historical Native Full
must never be replaced with All-Writer or image-source ablations.

Preparation verifies committed inputs, native baselines and the original 480
LLaVA question x 32 layer generations. GPU smoke and each question's full-logit
exact replay gate historical reuse. LLaVA preserves the original full-prefill
LM-head shape (`logits_to_keep=0`); Qwen preserves its native `=1` path.

Execution order is CPU tests, source commit, freeze, two LLaVA workers, independent
CPU scoring/archive audit and GPU-release gate, two Qwen workers, independent
CPU scoring/archive audit, cross-model figures, result commit and branch push.
There is no background GPU availability polling or cross-model concurrency.

Each condition has an immutable atomic checkpoint. A worker verifies existing
checkpoint bindings before skipping it. Keep all failed-attempt logs; never
overwrite checkpoints or loosen exact numerical checks when resuming.

Statistics use question-weighted image-cluster pairs (20,000 bootstrap draws,
seed 20261104), ordinary 95% and model-specific Bonferroni-32/36 intervals.
Curves are unsmoothed; all negative results and harm/rescue cases are retained.
