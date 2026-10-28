# Phase 3.6: frozen 240-image inheritance validation

Baseline: `ed666b68eef6380a8bd57852636c18bef1673b02`. Only branch
`exp/phase3_6-inheritance240` may be pushed; main is not an output target.
The later upstream `e89c8ac` adds offline Sink comparisons and is intentionally
not incorporated into this baseline-bound experiment.

The unchanged `VisualAttentionInheritance` implementation is reused. The six
research conditions are L23/L24 sources, L24-L31 at lambda .5/1, and L25-L31
at lambda1. Two lambda0 controls (L23/L24, L24-L31) and Native Full make nine
rows per image. No other research condition is permitted.

All 240 references are freshly captured from each image's actual Native Full,
head by head. The 53 existing task caches are independently hash bound and
compared at L23/L24, not transplanted across images. Cross-machine historical
Full/cache discrepancies are recorded without fitted thresholds. Same-machine
lambda0 effective-text full-vocabulary logits, Question hidden invariance,
non-target connections, cleanup, original FP16 rounding bounds and mass/row
tolerances are mandatory. Nothing relaxes the existing inheritance algorithm.

Commit all implementation and tests before preparing protocol.json. Run from
vlm_research, with the fixed environment and offline model resources:

```bash
python -m unittest test_attention_inheritance test_phase36 -v
python -m phase3_6.run prepare --output-root results/phase3_6_inheritance_240 \
  --model-path /fixed/model/snapshot --image-root /verified/MMEB/GQA
CUDA_VISIBLE_DEVICES='' python -m phase3_6.acceptance --model-path /fixed/model/snapshot \
  --audit-dir results/phase3_6_inheritance_240/acceptance
CUDA_VISIBLE_DEVICES=0 python -m phase3_6.run smoke \
  --output-root /absolute/master/results/phase3_6_inheritance_240
```

Then create four detached worktrees at the frozen commit and launch one worker
per worktree with CUDA_VISIBLE_DEVICES matching shard0/1/2/3. The three helper
Codex instances execute only the frozen code; no public source changes.

```bash
CUDA_VISIBLE_DEVICES=0 python -m phase3_6.run worker --shard 0 \
  --output-root /absolute/master/results/phase3_6_inheritance_240
```

Sorted string question IDs are split into four contiguous 60-image shards.
Each owns its shard directory, lock and sample checkpoints. Attempts use new
artifact subdirectories. Only complete, hash-verified samples are skipped on
restart. Foreign samples, modified sources/inputs/checkpoints and incompatible
bindings fail. A failure never counts as a completed image. No worker aggregates.

Only after all four gates pass, the master runs once at the frozen commit:

```bash
python -m phase3_6.analyze --output-root results/phase3_6_inheritance_240
```

The primary cohort excludes discovery15133706 (239); uncached187 is reported
separately; all240 and discovery1 are supplemental. The 187 subgroup means no
existing Phase3.4 task cache, not that the other52 cached images underwent
inheritance. Weak Full margin is abs<=.15; severe loss is gain<=-.5 nats.
Report six fixed conditions, left/right and Full correctness strata, absolute
full-vocabulary correct logP, tails, lambda directions, matched source contrasts,
and historical Physical K24 gain correlations. Ties are wrong. Fixed-seed5000
image-paired percentile bootstrap intervals are exploratory and unadjusted for
multiple comparisons. Flip statistics use exact two-sided paired McNemar.

Full-vocabulary normalization is retained in scoring, not two-candidate softmax.
Each row retains both candidate logits, their absolute logP and the full-vocabulary
logsumexp, plus prediction/effective-text hashes. Smoke retains full raw P-1
logits. Per-image reference NPZs and compressed per-head local audits are saved.
No full Attention matrices or redundant full-vocabulary tensors are persisted
for all240, avoiding unnecessary disk and Git payloads. All numerical invariants
are checked against the actual tensors before saving the checkpoint.

Do not train, decode, rerun Physical, expand data or merge main automatically.
If incomplete, state partial with actual coverage and do not report full-cohort
conclusions. Freeze an implementation commit before workers; later result-only
commits do not change that implementation version.
