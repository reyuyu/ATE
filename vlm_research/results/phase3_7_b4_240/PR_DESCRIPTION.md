# Phase 3.7B: independent224 confirmation of the original B4 mask

## Baseline and Scope

Base result commit: `f1e661fd228ec600e5a63b3016d68bd2fde64940`.
Executed frozen code: `4f9ff439f18a4c60722ab86e225c30816bc717b0`.
Branch: `exp/phase3_7-b4-240`.

This is a stacked experiment after Phase3.7. To review only the new runner/tests/results, compare against `exp/phase3_7-l0l1-pathways` (the pinned f1e661f baseline), not an older main that lacks that pilot. No main push or automatic merge is performed.

Only Native Full versus the unmodified Phase3.7 B4 definition: P-1 Query->real Question Keys blocked in L0/L1. FP16/eager LLaVA-1.5-7B, Transformers4.49.0, fixed revision, original prompt/candidates/full conditional scoring, all576 visual tokens. No other mask, strength, window, training, decode, cache or exit.

## Frozen Data and Analysis

- All240 original Phase3.2 independent images, no resampling/filtering/replacement.
- Primary confirmation224: exact image-ID complement of the discovery16;112 left and112 right.
- Original16 and all240 supplementary, separately reported.
- Manifest frozen before GPU scoring, including image/model/source hashes and real Question/P-1/BOS/candidate mappings.
- Primary endpoint: paired accuracy difference in confirmation224, image-paired20,000-resample percentile CI95, exact two-sided McNemar on discordant pairs.
- Descriptive left/right, current-Full-correct/wrong subsets, signed/absolute/tail Margin Gain, gold/wrong absolute LogP and right-left preference changes.
- Four-way Attention Mass: Image, Question, BOS, other text; both actual global Full/B4 comparison and local same-input redistribution.

## Validation and Review Artifacts

147 CPU tests passed (18 new focused tests). Two discovery-image real-GPU smoke verifies exact new-versus-original computation before formal execution; archived cross-run differences are explicitly recorded. Only physical GPU0 is exposed, and no other GPU process is stopped.

Formal completion is certified by `run/validation.json` plus `summary_validation.json`: all240 unique checkpoints, exact cohort coverage, both candidates and both conditions, finite raw vocabulary logits, zero blocked weights, correct causal/native later masks, all32 Question hidden states and final RMSNorm bitwise Full, no-op/cleanup logits exact, unchanged input/model bytes/installed Transformers and no hook leak. A missing/failed gate means partial; it is not an existing scientific result.

Read `vlm_research/results/phase3_7_b4_240/report.md` for the actual measured effect and all seven research answers, `statistics_ci95.csv` for intervals, `per_sample_results.csv` for every paired image, `discovery_replication.csv` for the original16, and the raw NPZ/compressed audits for numerical review. No conclusions are selected from interim scores or a favorable mean Margin alone.

## Completed Findings

All240 paired images completed and were audited; one scientific aggregation. Confirmation224: Full180/224 (80.36%), B4184/224 (82.14%), delta+1.7857pp, paired Bootstrap CI95[0,4.0179]pp, exact McNemar p=0.21875. Five right-labelled errors are corrected and one left-labelled correct answer is harmed. This is a favorable point estimate, not a confirmed accuracy improvement.

Primary mean/median Gain+0.309623/+0.351562; mean per-image absolute Gain0.393293;177 positive and47 negative gains, worst-0.609375. Gold/wrong full-vocabulary LogP changes+1.420711/+1.111089. Left/right mean Gains are both positive (+0.250628/+0.368617), so the improvement is not explained simply by a uniform rightward shift; overall right-left shift CI includes zero. Accuracy corrections are nevertheless concentrated on right and left Full has much higher initial accuracy.

Actual Image Mass increases by0.114185 at L0 and0.109655 at L1, but local L1 redistribution is largest to other text, not Image; BOS is small. This does not prove better fusion or a globally delayed Question-reading mechanism. Original16 mean Gain+0.130127 and archived raw logits reproduce exactly. All240: Full188 -> B4192. Historical RTX4090 Full was187; one cross-run weak-margin correctness difference is explicitly recorded, not used as the paired control.

Five figures reviewed; all CPU/GPU/raw-artifact gates pass; GPU resources released. The original generated report/gate are archived, and reviewed wording/flip tables are separately documented without changing any scientific result.

The mask delays only direct edges; indirect template-text paths remain. More Image mass is not automatically better fusion, and a rightward candidate shift is not automatically improved grounding. The preselected source cohort is not an unbiased GQA population sample. Review the accuracy endpoint, both flip directions, directional effects and negative tails together.

## PR Creation

The experiment branch is pushed only after complete gates pass. The following is a PR creation link, not an already-created PR:

https://github.com/reyuyu/ATE/compare/exp/phase3_7-l0l1-pathways...exp/phase3_7-b4-240?expand=1

Do not merge automatically. Full/B4 research stops after this240-image run; wait for the next research decision.
