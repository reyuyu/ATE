# Phase 3.7B Result Index

Frozen Full/B4 paired validation on the original240 GQA ChooseRel images. Primary group:224 images not used in the Phase3.7 B4 discovery pilot. Original16 and all240 are supplementary, never mixed into the primary denominator. Left/right each112 in confirmation224 and each8 in discovery16.

Base: `f1e661fd228ec600e5a63b3016d68bd2fde64940`.
Executed code: `4f9ff439f18a4c60722ab86e225c30816bc717b0`.
Manifest binding: `81553bff73001b102bfd4b310f5f9a8d3b81a535da06ca44226555396909a686`.
Branch: `exp/phase3_7-b4-240`.

Completion is certified only by both `run/validation.json` and `summary_validation.json` with `passed=true`, correct240/224/16 coverage and matching artifact hashes. Individual checkpoint progress is not a full-cohort result. A missing or failed completion gate means partial, never an inferred completed experiment.

- [report.md](report.md): measured performance, seven research answers, tail cases and mechanism limits.
- [manifest.json](manifest.json): original source/image/model hashes, dynamic actual input mappings, source commit, frozen conditions/cohorts/statistics.
- [per_sample_results.csv](per_sample_results.csv): all240 paired rows, candidate LogPs, margins, directional preference, hidden drift and four-way Mass.
- [per_condition_results.csv](per_condition_results.csv):480 Full/B4 condition rows.
- [summary.json](summary.json), [statistics_ci95.csv](statistics_ci95.csv): image-paired Bootstrap intervals, exact McNemar, left/right and current-Full-correct/wrong subsets.
- [discovery_replication.csv](discovery_replication.csv): original16 versus this independent rerun, including archived raw-logit differences.
- `acceptance/`:147 CPU tests including18 focused tests, pip consistency and exact installed versions.
- `smoke/`, [smoke_review.json](smoke_review.json): two discovery images, exact new-versus-old implementation and raw audit checks before formal scoring.
- `run/`: unique per-image checkpoints, raw P-1 logits/attention/Mass/hidden observations, both-candidate audits, all32 Question hidden hashes, GPU/environment and parameter-byte hashes.
- [summary_validation.json](summary_validation.json): sealed file inventory, cohort counts and one scientific aggregation.
- [machine_report.md](machine_report.md), [aggregation_validation_generated.json](aggregation_validation_generated.json): unedited original generated report/gate; [publication_review.json](publication_review.json) records interpretation-only publication review and preservation of scientific artifacts.
- [numerical_audit.json](numerical_audit.json): all240 historical-versus-current Full differences, kept separate from the current paired control. Maximum absolute Margin difference is0.140625; one correctness label changes across runs/hardware. No correctness tolerance was relaxed.
- [PR_DESCRIPTION.md](PR_DESCRIPTION.md): review-ready PR body. Push only this branch; never merge main automatically.

The five PNGs show paired accuracy/flips, confirmation224 Gain distribution, left/right gains, direction-preference shift and Image/Question/BOS/other-text Mass. All cases and negative tails remain in the outputs. No raw source images, model weights, environment installation or credentials are committed.

Only physical GPU0 is exposed by UUID. L0/L1 use the unmodified Phase3.7 B4 class; every later mask and all576 image tokens are native. Extra hooks only observe. No new mask condition, layer/strength scan, training, decode, KV cache or exit is part of this experiment.

The primary endpoint is paired accuracy difference in confirmation224, not positive mean Margin Gain. All paired Bootstrap resamples are by image, never head/layer. The original preselected ChooseRel cohort does not imply unbiased GQA-population performance. Increased Image Mass is not automatically better fusion, and blocking Question does not remove all text dependence.

The intervention delays only the two layers' direct P-1->Question edges. Other earlier template-text queries may already contain Question information and remain readable by P-1. Therefore it does not enforce absence of all Question-to-answer information flow until L2, and cannot by itself establish that a globally delayed-reading architecture is preferable. Question representations and their original visual access are retained.

For reproduction, use the frozen code commit and a fresh worktree/output directory; see `phase3_7_b4_240/README.md`. For read-only artifact verification at the result commit, run from `vlm_research`:

```bash
CUDA_VISIBLE_DEVICES='' python -m phase3_7_b4_240.analyze --verify-existing
```
