# Phase 3.7: frozen L0-L1 pathway pilot

Independent branch/worktree from main e89c8ac. Only new phase3_7 and test files;
do not edit any historical implementation, manifest or result. Exactly one GPU.

Freeze implementation, then CPU-only preparation of 16 stratified historical
images and all available real same-image question pairs within that pilot.
See protocol.RULES for the deterministic margin-bin selection, original-source
requirements, aggregation definitions and interpretation limitations. The
upstream author CSV is not misrepresented as independently checked original
Stanford annotation JSON. Do not invent missing questions, labels or ROIs.

Use the existing fixed Python3.11 environment and canonical local model/image
resources. Run from vlm_research, explicitly hiding GPUs for CPU steps:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest test_phase37 -v
# Commit implementation and tests before preparing the frozen manifest.
CUDA_VISIBLE_DEVICES='' python -m phase3_7.run prepare --model-path MODEL --image-root GQA
CUDA_VISIBLE_DEVICES='' python -m phase3_7.run acceptance --model-path MODEL
CUDA_VISIBLE_DEVICES=0 python -m phase3_7.run smoke
CUDA_VISIBLE_DEVICES=0 python -m phase3_7.run pilot
CUDA_VISIBLE_DEVICES='' python -m phase3_7.analyze
```

Only Native Full A and B0-B4 are research conditions. No-op, restored Full,
same-local-input Attention replay and QK extraction are correctness controls,
not additional causal experiments. Masks are the unchanged original directed
mask engine, restricted to L0/L1; B4 generalizes the target Key selection to
real Question tokens. All other connections keep their original additive
mask values, and softmax redistributes probability within masked queries.

Non-target Attention is checked against native attention with the same local
hidden input. L1 hidden inputs can differ after L0 intervention; demanding all
non-target L1 probabilities equal global Full would be incorrect. L2-L31
must retain native masks, not necessarily native probabilities. All visual
tokens remain present at every layer. No layer scan, physical exit, decode,
cache, training or extra dataset is authorized.

A uses Native prompt-only Prefill and per-real-query/head conditional maps.
B uses the original complete one-token candidate-prefill scoring protocol;
the already-input answer is not P-1. Do not mix raw Mass with spatial change,
or pair different Question tokens by absolute index. Matched L0 P-1 cases
must pass exact visual QK and visual-only softmax invariance checks, while
FP16 probability storage may produce tiny denominator-dependent differences.

Per-image checkpoints are hash bound to code, source, model, mappings and
sample identity. Resume only compatible complete checkpoints; failure artifacts
remain partial. Never overwrite completed samples or aggregate twice.
Use the frozen implementation commit for reproduction, not a later result
commit. Publish only the experiment branch, with honest PR authorization status;
do not push or merge main. Stop after this A/B pilot and await a new decision.
