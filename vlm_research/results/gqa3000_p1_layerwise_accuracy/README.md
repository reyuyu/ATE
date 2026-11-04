# GQA3000 original-P-1 single-layer answer-accuracy profiles

**Initial snapshot: PARTIAL / RUNNING.** This initial commit contains the frozen protocols, inputs,
historical reuse audit, CPU tests and dual-GPU LLaVA Smoke. It does not contain
completed 3000-question accuracy profiles and must not be cited as a full result.

For later commits, the authoritative final status is `completion.json` plus
`validation.json` in each model directory and `cross_model/completion.json`.
The pipeline commits completed results only after those completion gates pass.

LLaVA runs first (32 layers), followed only after complete scoring, validation
and GPU release by Qwen (36 layers). The only intervention zeros one selected
layer's original prefill P-1 `self_attn.o_proj` output; all decoding is native.

Exact historical Native Full baselines: LLaVA 2627/3000 (87.5667%), Qwen
2653/3000 (88.4333%). Neither is an All-Writer baseline. The audited original
480-question LLaVA generation subset supplies 15,360 Drop labels; the remaining
80,640 LLaVA and 108,000 Qwen Drop generations are new.

Native Full first-logit replay is required on the same GPU for every question.
Any exactness failure stops the model. All partial checkpoints and failures are
retained; no partial subset is interpreted as the full 3000-question result.

After completion, each model directory receives independent accuracy tables,
paired image-cluster confidence intervals, figures, archived checkpoints,
validation and `completion.json`. The cross-model directory is populated only
after both model stages pass. Final delivery creates a separate result commit.
