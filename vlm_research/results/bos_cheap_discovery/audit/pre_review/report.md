# BOS-Guided Cheap Bypass Predictor Discovery

Status: completed exploratory study; NOT independent confirmation.

Code `899222303032aace505c45250c941ecf3062fad5`. Original3000 training/calibration questions; 554 unused-image exploratory questions instead of required600. LogicalObj shortfall is explicit, no replacement, resampling or synthetic labels.
All original96000 single-layer labels reused. Old questions receive only one new native feature forward. New questions receive Full plus32 original single-layer P-1 Attention-output ablations. No LLaVA training, Decode study, joint gate or custom accelerated kernel.
True BOS mass, Attention/MLP output norms are post-computation oracle/mechanism features, excluded from online predictor inputs. All risks concern Full original top1, not answer accuracy. All policies and thresholds froze before new-image model outputs.

## Frozen Thresholds and Risk

Primary epsilon=.03 below; all .01/.03/.05 results retained in CSV. Maximum calibration coverage under observed1% risk is exploration, not a guarantee. Ordinary2000 image-cluster bootstrap intervals do not certify zero-event risk. Additional bounded-cluster simultaneous Hoeffding upper bounds assume independent image clusters and are conservative; they are not unconditional population safety guarantees. Task shortfall also prevents the requested confirmation claim.

| Predictor | Threshold | New skip coverage | Dangerous/skipped | Risk | Bootstrap95%CI | Simultaneous upper95 |
|---|---:|---:|---:|---:|---|---:|
| Layer-only | 0.02 | 12.50% | 13/2216 | 0.587% | [0.181%,1.128%] | 100.00% |
| Length-only | 0 | 0.00% | 0/0 | undefined | undefined | 100.00% |
| BOS-QK+Layer | 0.02 | 12.61% | 13/2235 | 0.582% | [0.179%,1.119%] | 100.00% |
| BOS-QK+Value | 0.02 | 12.70% | 13/2252 | 0.577% | [0.177%,1.111%] | 100.00% |
| BOS-QK+Residual | 0.02 | 14.26% | 16/2528 | 0.633% | [0.234%,1.132%] | 100.00% |
| Sampled6Keys | 0.02 | 13.66% | 15/2422 | 0.619% | [0.206%,1.127%] | 100.00% |
| Global-noLayer | 0.02 | 6.18% | 7/1096 | 0.639% | [0.182%,1.189%] | 100.00% |
| Early-separate | 0.01 | 6.25% | 5/1108 | 0.451% | [0.090%,0.993%] | 100.00% |
| Region-grouped | 0.01 | 7.61% | 4/1349 | 0.297% | [0.000%,0.678%] | 100.00% |
| BOS-cos | 0.652654 | 5.13% | 7/909 | 0.770% | [0.223%,1.354%] | 100.00% |
| BOS-score | 3.24211 | 0.01% | 1/1 | 100.000% | [100.000%,100.000%] | 100.00% |
| BOS-Mass-Oracle | 0.9 | 1.43% | 0/254 | 0.000% | [0.000%,0.000%] | 100.00% |
| Early-never-skip | 0.02 | 14.26% | 16/2528 | 0.633% | [0.234%,1.132%] | 100.00% |
| Region-threshold | 0 | 15.53% | 23/2754 | 0.835% | [0.404%,1.348%] | 100.00% |
| Layer-threshold | 0 | 11.39% | 18/2019 | 0.892% | [0.444%,1.413%] | 100.00% |

Calibration-only recommended names (never chosen from new-set performance):
`{"0.01": "Layer-threshold", "0.03": "Region-threshold", "0.05": "Region-grouped"}`

## Boundaries

Single-layer labels cannot be added to infer joint-gate harm. Joint dynamic paths and full-answer accuracy were intentionally not executed. No safe skipping, actual computation omission, end-to-end latency improvement or cross-model generality is claimed. A P-1 row is a small fraction of full Prefill. Microbenchmarks report PyTorch operations and dispatch, not a custom kernel speedup.

## Artifacts

cheap_features_per_question.csv retains train/calibration/new-exploratory labels, all32 layers and online fields. Full per-head CSV remains locally with SHA256; identical lossless gzip task partitions are in Git. Per-question raw q/k/v, sampled Keys, residuals and full/ablated logits remain in external resources with SHA256.

![figure1_risk_coverage](figures/figure1_risk_coverage.png)

![figure2_cheap_relationship](figures/figure2_cheap_relationship.png)

![figure3_cheap_relationship](figures/figure3_cheap_relationship.png)

![figure3_detail_ChooseAttr](figures/figure3_detail_ChooseAttr.png)

![figure3_detail_ChooseCat](figures/figure3_detail_ChooseCat.png)

![figure3_detail_ChooseRel](figures/figure3_detail_ChooseRel.png)

![figure3_detail_CompareAttr](figures/figure3_detail_CompareAttr.png)

![figure3_detail_LogicalObj](figures/figure3_detail_LogicalObj.png)

![figure3_detail_QueryAttr](figures/figure3_detail_QueryAttr.png)

![figure4_early_special_case](figures/figure4_early_special_case.png)

![figure5_value_residual_update](figures/figure5_value_residual_update.png)

![figure6_calibration_errors](figures/figure6_calibration_errors.png)

![figure7_threshold_comparison](figures/figure7_threshold_comparison.png)
