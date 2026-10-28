# Phase 3.7: frozen L0-L1 pathway pilot, complete A/B results

## Scope

Base `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`; executed code `174014dac2bacb5712d2b7d6bac318ec3e77b5b2`.
Branch `exp/phase3_7-l0l1-pathways`, intended target `main`; do not auto-merge.
Only new Phase 3.7 source/tests/results, no changes to historical source/results, model weights, installed Transformers or original prompts. One GPU, no training, extra layers, decode, cache or physical exit.

## Implementation and Validation

- Reuse the existing directed additive-mask machinery and original complete-candidate scoring.
- Freeze 16 stratified images before GPU scoring; 12 eligible same-image question pairs, no fabricated questions or ROIs.
- CPU: 129/129 tests including 21 focused checks; two-image GPU smoke passes before formal execution.
- Formal: all 16 images x B0-B4 and 12 native A pairs complete; 5,120 layer audits and 256 local masked-layer/candidate audits.
- No-op and post-cleanup valid-text logits exact; all 576 visual tokens retained; only specified L0/L1 masks modified; blocked weights zero; later masks native.
- Model byte hashes before/after identical; no input/source mutation; no accepted NaN/Inf/OOM/failure.
- One scientific aggregation; all 17 figures and publication artifact hashes verified. Original log-finalization hash discrepancy and reviewed-report changes are explicitly archived/disclosed, not hidden by rerunning aggregation.

## Main Descriptive Results

| Condition | Accuracy | Mean Margin Gain | Mean per-image absolute Gain |
| --- | ---: | ---: | ---: |
| B0 Full | 8/16 | 0 | 0 |
| B1 Text->Image | 8/16 | -0.143555 | 0.545898 |
| B2 P-1->Image | 8/16 | +0.099609 | 0.307617 |
| B3 Question->Image | 8/16 | +0.007813 | 0.649902 |
| B4 P-1->Question | 8/16 | +0.130127 | 0.312256 |

No correctness flips or ties. B1 worst Gain -1.289062; signs and left/right effects can cancel. Accuracy unchanged does not imply an unused pathway. Gold absolute full-vocabulary LogP changes are reported separately from two-candidate ranking. FULL_COMPARISON.md and full_comparison.csv compare all 16 images directly against current-machine Native Full (mean Full Margin +0.662354).

A maps are nearly invariant across matched questions (mean cosine >=0.999992, entropy about 0.94-0.96). All 12 L0 P-1 visual-QK/FP32 conditional checks are exactly invariant. L1 Question has small positive paired ROI preference in 11/12 pairs, not strong spatial selectivity. Question/P-1 paths have different causal timing and effects; independent blocks do not test their joint necessity. L2+ compensation remains possible.

## Limitations and Review

Deliberately stratified 16-image mechanism pilot, not an unbiased accuracy estimate or significance claim. A excludes four left/wrong images without valid pairs. Fixed author GQA CSV has real IDs/answers/boxes, but the original Stanford JSON was not independently recovered. No new annotations or result-based filtering. Read `vlm_research/results/phase3_7_l0_l1_pathways/report.md` and the provenance/audit files before interpreting H1/H2.

The deploy key permits branch push only. No GitHub account/API credential is available to create the PR automatically. Create the PR using:

https://github.com/reyuyu/ATE/pull/new/exp/phase3_7-l0l1-pathways

This is a creation link, not evidence of an already-created PR. The completed experiment stops here pending research review.
