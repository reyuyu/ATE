# Qwen3-VL Semantic Visual Writers

The byte-identical LLaVA `semantic_visual_writers/groups.py` is the only lexical
selector. Native Qwen processor IDs, offsets, variable image grids and mRoPE are
frozen separately. No label, historical correctness or model output selects words.
R includes `vision_end`: leaving that post-image query open would introduce an
uncontrolled visual relay in a source-mask experiment. No original token is removed.

The native historical Qwen generator, last-logit path, real caches, DeepStack and
residual Smoke observer are reused without modification. Only pre-attention input
masks at all 36 language layers change. All answer queries are blocked from original
image keys, including All-Writer. Historical native Full is not this baseline.

Run from the worktree root in the pinned Qwen environment, offline:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_semantic_visual_writers.run cpu
```

Commit all source before `freeze`, then commit all frozen artifacts before GPU use:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_semantic_visual_writers.run freeze
```

Use only the exact GPU UUID selected in `protocol.json`, after rechecking that it is
idle. There is no polling daemon, occupied-GPU takeover or numerical fallback.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u -m vlm_research.qwen3vl_semantic_visual_writers.run execute
```

Three strict real-GPU Smoke cases and all matched-layout No-Writer swap controls
must pass before the main matrix. Each completed generation is immutable and bound
to its exact sample, native inputs, selector, code and protocol. Preserve failed or
orphaned artifacts and stop; never silently overwrite or loosen equality.

After normal model exit, record a GPU release snapshot. CPU analysis uses the
historical plotting environment (no Qwen forward or new generation):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_semantic_visual_writers.analyze
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_semantic_visual_writers.analyze --check
```

The first aggregates once; the second only replays all 2800 checkpoints, raw array
hashes, masks, mRoPE, statistics and tables. Native positional layout differences
in image swaps are kept and reported separately, never misclassified as leakage.
Raw vocabulary arrays remain outside Git with hashes. No training, layer search,
router, activation patching, kernel or speedup implementation is included.
