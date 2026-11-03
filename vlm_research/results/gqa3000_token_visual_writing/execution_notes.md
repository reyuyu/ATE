# GQA3000 execution and audit notes

## Frozen identity

- Branch: `codex/gqa3000-token-visual-writing-curve`.
- Worktree: `/data/lk/ATE-worktrees/gqa3000-token-visual-writing-curve`.
- Base: `a125cc0f72f9fab3cabc24bf29ee458e23a38b69`.
- Execution source: `fd6da3078dba44429f72d9a26d3e934d7ea02eb7`.
- Postprocessing sources are linked by preserved `postprocessing_freeze.json`, `postprocessing_freeze_v2.json` and final `postprocessing_freeze_v3.json`. V2 adds required anomaly-count exports; V3 provides a7.2-inch journal-width four-panel main figure, direct paired-control CI annotations, and an eight-page vector supplement with six separate task figures. Neither revision changes metrics, comparisons, confidence bounds, seeds or token selections. All were frozen before outcome aggregation.
- Frozen Semantic-k reference: `9687b298efcdca651cc5446b5cb07da4a2b4e37c`.
- Frozen source-mask reference: `6b9467d6fb4e0f8535040809bee60397ca798ea8`.
- GQA3000 manifest SHA256: `b5f6dec34cc5c96c6cce2d080b141bbbc8379d3d4b72e67f322ac1e63aa8e9f2`.

The input-only design completed before loading a model. It fixes 44 labels on every original question, including natural-k and budget-scan controls. Labels with exactly identical writer positions are deduplicated only within that same original question. There are 132,000 logical generation/NLL labels and 109,518 canonical evaluations. This does not permit sharing predictions across different images or questions.

## Hardware and worker isolation

The initial live resource check found physical GPUs 0 and 1 free, while 2 and 3 were being used by unrelated tasks. The selected UUIDs are:

- Worker 0: `GPU-a63c6362-dcfd-4a26-c44b-014b40127c89`.
- Worker 1: `GPU-b08489af-0a8a-eb48-55ba-8346733af417`.

Each child exposes exactly its UUID through `CUDA_VISIBLE_DEVICES`, so both use their own logical `cuda:0`, with one model instance each. `shards.json` fixes image-cluster assignments; each worker has 1500 questions. No foreign process is killed, migrated or preempted. The foreground supervisor observes only this experiment's workers. No recurring automation or background resource-allocation poller is installed.

## Reproduction entry points

Run from the worktree with `PYTHONPATH=.:vlm_research`, `PYTHONDONTWRITEBYTECODE=1`, using `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.

1. `python -m vlm_research.gqa3000_token_visual_writing.prepare design` creates the immutable input design and intentionally refuses to overwrite an existing selection index. Do not rerun against this completed design.
2. `python -m vlm_research.gqa3000_token_visual_writing.prepare cpu` checks the inherited masks and new budget/Gold/shard logic without initializing CUDA. 57 tests passed.
3. Commit the execution sources, then `python -m vlm_research.gqa3000_token_visual_writing.prepare freeze` verifies all frozen inputs and full model-file SHA256 values, records dependencies, and chooses two genuinely idle GPUs.
4. `python -u -m vlm_research.gqa3000_token_visual_writing.driver --attempt initial` performs fresh live GPU checks, starts the isolated workers, and holds both at a barrier until both real-model smoke suites and cross-GPU strict equality pass.
5. Only after both workers finish: `python -m vlm_research.gqa3000_token_visual_writing.analyze replay`, followed by `statistics`, then `swaps`.
6. `python -m vlm_research.gqa3000_token_visual_writing.plot` exports the actual figures. Inspect actual PNG/PDF outputs and record visual QA before final completion.
7. `python -m vlm_research.gqa3000_token_visual_writing.finalize report`; write the evidence-based scientific interpretation from the fixed comparisons; then `finalize complete` verifies all artifacts and process release.

These commands describe the protocol, not authorization to rerun completed generation. Resume is only appropriate after a non-validation interruption, with no surviving workers and a new unique attempt identifier. A numerical validation failure or OOM writes `STOP.json`, prevents further forwards, and must remain preserved rather than bypassed.

## Strict numerical checks

Both GPUs passed the three frozen questions `00150746`, `11697551`, `11127723`, plus the input-only selected multi-Gold-token question `001035207`. Tests compare every valid native/No-op greedy logit, every teacher-forced Gold logit, and the original frozen receiver-mask implementation using `torch.equal`. No tolerance is introduced for these checks. Cross-GPU native greedy and Gold arrays are also strictly equal.

The receiver-mask controller keeps audit reductions on the GPU until an evaluation ends, but changes no model arithmetic, kernels, native weights, precision or mask semantics. It checks every actual forbidden attention probability at every layer and every real step, not a sample. Non-target mask entries must equal their native values exactly. The frozen normalization tolerance (`ROW_ATOL = 0.002`) applies only to FP16 attention row sums; forbidden probabilities and non-target mask comparisons remain exact. All 32 MLPs, independent caches, cache indices, input hashes, parameter versions and hook cleanup are checked as specified in the protocol.

Generation and teacher forcing start with separate real caches. Both use `logits_to_keep=1`, including the initial prompt. Their initial logit vectors must be strictly equal under the same condition. Teacher forcing consumes only the frozen official Gold prefix, never a freely generated prefix. Primary Gold NLL excludes EOS and any additional BOS. Each saved NLL is recomputed from saved raw FP16 vocabulary logits using CPU float64 log-softmax.

## Checkpoints and server-only arrays

Checkpoint JSON and NPZ writes use temporary files, fsync and atomic hard-link publication. Published records cannot be overwritten with different contents. A complete NPZ orphaned immediately before JSON publication can be recovered from its embedded metadata after all bindings and hashes validate.

The server resource root is `/data/lk/ATE-resources/experiments/gqa3000_token_visual_writing`. Each canonical NPZ contains its first logit vector, every Gold-token vocabulary vector, per-forward/per-layer audit matrices, and embedded immutable record metadata. No-Writer swap endpoints additionally retain every effective greedy logit vector, including EOS prediction, for strict image-independence checks. Raw arrays are server resources, not claimed to be present on GitHub. Their exact paths and SHA256 hashes are in checkpoints and `raw_logits_hashes.json`.

All 44 logical labels, complete answers, generated Token IDs, Gold Token IDs, per-token NLL, harm/rescue events and mask budgets are retained in six task-partitioned CSVs indexed by `per_question_index.json`. The partitioning prevents single GitHub files from exceeding its size limit.

## Interpretation boundaries

All-Writer blocks direct answer-stage access to raw image keys and is distinct from native Full. Question100 is distinct from R100. Some natural-k cases have k=0; no sample is removed. Some exact whole-word budget sets are not nested; successive points are descriptive policy comparisons rather than single-token marginal interventions.

The primary family and its Bonferroni one-sided bounds are frozen in `protocol.json`. Ordinary pointwise 95% intervals on curves and task panels are not simultaneous bands. Random seeds are averaged within question, and random-to-random harm events pair corresponding seeds. Image-swap endpoints are audited reuse of main-run records, with separate linked-pair statistics; they are not additional independent main questions.

No actual FLOPs, latency or memory reduction is inferred from this dense functional mask experiment. No layer search, router, kernel, KV pruning, training or cross-model experiment is included.
