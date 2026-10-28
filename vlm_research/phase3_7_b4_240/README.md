# Phase 3.7B: Full/B4 Confirmation240

New worktree/branch from f1e661fd228ec600e5a63b3016d68bd2fde64940. Historical source and results remain unchanged. Only the existing Phase3.7 B4 definition is used; all extra hooks are read-only diagnostics.

The immutable manifest binds all original240 images, original input/candidate mappings, model/source hashes, original16 discovery membership and its224-image complement before formal GPU scoring. Image is the unit for the 20,000 paired percentile bootstrap resamples. Exact two-sided McNemar tests discordant predictions. The primary endpoint is confirmation224 accuracy difference; subgroup intervals and margins are descriptive, not substitutes for accuracy.

Run from vlm_research, using the existing pinned Python3.11 environment. Commit all new source/tests first. The CPU prepare step checks every image and canonical model file; it does not resample or rewrite history.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest test_phase37_b4_240 -v
# Freeze implementation commit before prepare/acceptance/smoke/formal.
CUDA_VISIBLE_DEVICES='' python -m phase3_7_b4_240.run prepare
CUDA_VISIBLE_DEVICES='' python -m phase3_7_b4_240.run acceptance
CUDA_VISIBLE_DEVICES=GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 python -m phase3_7_b4_240.run smoke
CUDA_VISIBLE_DEVICES=GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 python -m phase3_7_b4_240.run run
CUDA_VISIBLE_DEVICES='' python -m phase3_7_b4_240.analyze
CUDA_VISIBLE_DEVICES='' python -m phase3_7_b4_240.analyze --verify-existing
```

Only physical GPU0 is exposed by its verified UUID, not a potentially reordered logical index. Preflight refuses another compute process on it. GPU1-3 are never used or stopped. Per-image atomic, hash-bound checkpoints support safe resume at the same code/config commit; existing completed samples are verified and not overwritten. Failed attempts remain partial. Locks are execution coordination, not result data.

No-op and restored native forwards are correctness controls, not new research conditions. Every image has both candidate logPs for Full and B4, raw vocabulary prediction logits and zero-target/causal/input/weight/hook audits. Question states at all32 decoder outputs and final RMSNorm must be bitwise Full. Same-run new/old implementation smoke is exact; archived cross-run FP16 differences are only recorded, never fitted away.

Per-head P-1 mass is partitioned into Image, real Question, actual BOS and remaining valid text. Both the actual paired Full/B4 mass and a local native replay with identical actual layer input are recorded. L1 local replay differs from global Full after an L0 intervention; do not conflate them. Blocking Question may redistribute mass to BOS/other text, not just Image, and more Image mass does not establish better fusion.

Summary runs exactly once after all240 gates pass; verify-existing only reads. Close stage console logs before aggregation. Do not redirect the aggregator's live stdout into the output inventory (its completion line is printed after sealing hashes). Result-only descendant commits are allowed for verification, not for new GPU execution.

Statistical definitions follow image-paired resampling (paired samples use one index draw) and exact binomial discordance inference. Independent known-value/zero-difference tests are included. No significance claim is made from an uncorrected collection of subgroup tests; the cohort is an existing preselected GQA subset, not an unbiased population sample.

Only Full vs B4, no new layers, strengths, masks, datasets, training, exit or decode. Push the experiment branch only; report honest PR authorization status and do not auto-merge. Stop after this240-image research question.
