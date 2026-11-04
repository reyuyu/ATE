# Reproduction and Audit

## Execution Binding

- Base: `3b46d36dc80e46fde778c6fc93191c8e33f24122`.
- Initial execution source freeze: `65693faf28` and the exact per-file hashes in `execution_freeze.json`.
- CHECK selection code: `2f2dbfc40b`.
- Candidate/configuration freeze: `7259633d1c`; authoritative content is `final_candidate_freeze.json` and its hash in the FINAL jobs.
- The wall clock is immutable: start 2026-11-05 04:31:15 +08:00, candidate limit 08:31:15, GPU limit 10:01:15, delivery limit 10:31:15.
- No model or image files were copied into this worktree. The frozen `/data` model/data paths, file hashes and dependency versions are in the protocol and input manifests.

## Original Launch Commands

Working directory: `/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h`.
Each worker exposes exactly one authorized physical GPU and loads one resident model. Run the two commands in separate terminals, not sequentially in one blocking shell.

```bash
CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /data/lk/深度学习/.envs/ate-vlm-research/bin/python -u -m vlm_research.accuracy_autoresearch.worker --rank 0
```

```bash
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /data/lk/深度学习/.envs/ate-vlm-research/bin/python -u -m vlm_research.accuracy_autoresearch.worker --rank 1
```

The immutable `jobs/*.json` files define round membership and configurations. Workers resume only checkpoints with matching source/configuration/input hashes, never share KV and never count a retained record twice. `select_final` was called once after CHECK analysis and paired timing, before any FINAL answer existed. `supervise` requested clean shutdown after both shards completed FINAL quality and timing; it never sends signals to external processes.

These are the original commands, not permission to reset this completed task's clock or overwrite its results. A new GPU reproduction needs a separately authorized worktree, run directory and budget. Do not edit or remove the original deadline or shutdown marker to bypass the budget.

## CPU-Only Statistical Recalculation

Use a separate audit worktree/copy of this branch so regenerated timestamps and figures cannot overwrite the original delivery. The following commands do not load a model or dispatch GPU jobs. The offline flag bypasses only the CPU analysis time check, never the GPU clock.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.accuracy_autoresearch.analyze quality 09_final --offline-audit
```

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.accuracy_autoresearch.analyze performance 10_final_performance --offline-audit
```

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.accuracy_autoresearch.finish audit
```

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.accuracy_autoresearch.finish cases
```

`finish tables` rebuilds the numerical report tables and PNG/PDF from final CSV files; the manually reviewed Chinese research interpretation is retained in Git. Bootstrap seed is 20261105, 20,000 replicates, whole-image SHA256 clusters, question-weighted means. VQAv2 uses the pinned official ten-answer consensus evaluator. H0 and H1 are independent answer bindings, not interchangeable accuracy or timing records.
