# Phase 3.6: frozen 240-image Attention inheritance validation

## Scope

Branch: `exp/phase3_6-inheritance240`; requested baseline:
`ed666b68eef6380a8bd57852636c18bef1673b02`.
Frozen implementation: `0d323e1bb3eb6f91443854e90242150b04e9eaf1`.
The later upstream offline-Sink commit is not part of this baseline-bound
run. No original algorithm, model configuration, manifest or history changes.
Do not automatically merge this PR.

## Implementation and Verification

- New isolated phase3_6 package reuses unchanged Attention inheritance.
- Fixed six conditions, Native Full and two lambda0 controls.
- 121 CPU tests pass without skips; two real-model smoke images pass.
- Four independent worktrees / GPUs / 60-image outputs, same frozen commit.
- All 240 images completed, 2,160 score rows and 29,760 formal layer audits.
- Per-image checkpoints, actual same-image references, compressed per-head
  audit, environment, source/model/input hashes, paired bootstrap, McNemar,
  correlations, negative tails and four plots are included.
- No training, decoding, KV cache, new dataset or Physical rerun.

## Findings and Limits

Primary n=239 excludes discovery15133706. L23 improves Margin on
194-196 images and exceeds L24 in all three matched mean-Gain contrasts.
For L24-L31/lambda1 the paired difference is +0.034290 nats, 95% paired
bootstrap CI [0.026118,0.042544]. This is not selected as a new best setting;
all frozen conditions are reported.

Native Full and all L23 conditions remain 188/239 correct (78.66%); all
L24 conditions are 187/239 (78.24%). No primary Full error is recovered.
Gold absolute mean LogP falls in every condition. The 187 previously
uncached images show similar relative Margin effects but no accuracy flips.
Thus relative Margin generalization is supported within this cohort, not
general accuracy/probability improvement or a proven Physical Exit mechanism.

Cross-machine history has one Full correctness disagreement and non-bitwise
Full/cache differences, all recorded. Current same-machine no-op and causal
invariants remain exact. The 6,738 original nominal rounding warnings are
retained; all original dtype-derived rounding bounds pass. No failed attempt
or NaN/Inf/OOM is recorded. All worker GPU processes exited normally.

Review entry: `vlm_research/results/phase3_6_inheritance_240/README.md`.
Read `review.md` for six research answers, grouped/tail outcomes, historical
discrepancies and publication caveats. `report.md` remains the unchanged
machine-generated report; `aggregate.json` is the sole master aggregation.

## Publication Status

This is a ready-to-submit description, **not evidence that a PR exists**.
The server has repository Deploy Key push authorization but no GitHub CLI,
account API token or standard HTTPS account credential. Creating the PR
requires account API authentication or a browser action by the repository
owner. Main is not pushed or automatically merged.
