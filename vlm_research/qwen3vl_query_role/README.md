# Qwen3-VL Query Roles, L1-L6, Fixed 240 Images

This module reuses the historical Qwen processor, scorer, cache-position checks,
and native BF16 residual/DeepStack Smoke observer. It does not change those modules.
It edits only selected language `self_attn.o_proj` output rows, not computation.

The original LLaVA sample manifest is copied byte-for-byte. The Qwen input mapping
is separate: I consists of actual image-pad tokens; R starts strictly after
`vision_end` and ends strictly before P-1; A includes P-1 and real decode queries.
The native post-image assistant/template text belongs to R, but `vision_end` does not.

Run from this worktree root. Freeze and CPU tests require no visible GPU:

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_query_role.run freeze
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_query_role.run cpu
```

Commit the source and frozen artifacts before `execute`. Recheck the selected
physical GPU UUID for external processes first. No automatic device selection,
waiting daemon, process termination, model precision or image-size fallback exists.

```bash
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u -m vlm_research.qwen3vl_query_role.run execute
```

Execution loads one model, runs three strict real-GPU Smoke cases, then the five
conditions serially. Every condition is checkpointed atomically and existing
checkpoints are verified rather than counted twice. Preserve failure records;
resume only on the exact frozen source, after diagnosing any failure. Never edit
source while execution is running. An orphan raw file intentionally blocks a
silent overwrite. Native generation is fresh-cache for every condition, including
EOS prediction. Raw first-vocabulary logits are stored outside Git with hashes.

Once the GPU process exits, record a GPU release snapshot and aggregate exactly
once using the separate historical CPU plotting environment:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_query_role.analyze
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_query_role.analyze --check
```

The second command validates completed artifacts without regenerating results.
The fixed 20,000 paired image-cluster draws serve all six declared contrasts and
the IA interaction; ordinary and Bonferroni-seven intervals are both retained.
The scope is exactly 240 questions and five conditions, with no extension.
