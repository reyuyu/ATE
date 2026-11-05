# Dense C3 quality-restoration experiment

This branch implements only five frozen dense interventions, Smoke checks and complete-answer accuracy collection. There is no performance stage, CUDA-event timer, FLOPs calculation or physical image-token deletion.

From the worktree root:

1. `python -m vlm_research.qwen_recovery.prepare` runs the CPU-only prior-input/baseline audit and freezes the exact previous 1000 questions.
2. `python -m vlm_research.qwen_recovery.tests` checks all mask groups and the unchanged manifest/shards.
3. Freeze a source Git commit and `source_manifest.json`. A current-round GPU authorization record is required by each worker.
4. On each newly verified idle GPU UUID, `CUDA_VISIBLE_DEVICES=<uuid> python -m vlm_research.qwen_recovery.run <rank> smoke`. Each GPU loads one BF16 model; all conditions for an image share its worker.
5. Only after both 10-question Smoke shards pass, replace `smoke` with `accuracy`. Existing checkpoints are reused only if all source/input/protocol bindings match. A validation exception is recorded and raised without a precision, cache or mask fallback.
6. On the existing CPU plotting environment run `python -m vlm_research.qwen_recovery.analyze`, then `python -m vlm_research.qwen_recovery.verify_delivery`.

Native and F answers are audited inherited files at base commit `30d3e7a0c9318e71a5951aca9a05b6b78108db72`, indexed in `baseline_index.csv`. New A/B/C/D/E answers are independent full greedy requests. Full image Hidden rows and all dense K/V remain in all 36 layers, including the functional zero-output layers. A/B/C open direct image access only outside L1-L6. D restores their output under the original image mask. E changes only the late R image mask.

The source-manifest binding in each request hashes canonical JSON; protocol, model-manifest and sample-manifest bindings hash actual file bytes. All raw-output CSV indices hash file bytes. Source and model assets are verified before GPU collection. Model/image assets remain external to Git.

All statistics are exploratory on the same previous sample, clustered by Image SHA256. The best condition is selected by full-sample accuracy with A/B/C/D/E tie order; ordinary paired 95% intervals and five-way Bonferroni simultaneous 95% intervals are both reported. Passing the user's quality gate permits a proposed physical design, but **does not start physical implementation, correctness validation or timing in this round**.
