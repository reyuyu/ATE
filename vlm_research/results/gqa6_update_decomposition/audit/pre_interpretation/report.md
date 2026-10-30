# Native Attention / MLP Update Decomposition

Status: complete. No new ablation or Decode.

Frozen code: `881f07f7984a87303bbd4d3ce41f1e86dd9ec75c`. Baseline: `221cdde56dde53908aaa953fe26672fd33334f84`.
Model: pinned LLaVA-1.5-7B FP16/eager, Transformers4.49.0; seed20261030.

## Data and Computation

600 input-hash-selected questions,100/task, 569 unique image IDs. Same-image questions remain on one physical GPU. No performance filtering, missing-image replacement, generation or new intervention.
Each formal question uses exactly ONE native prompt forward, capturing actual decoder input, o_proj output, post_attention_layernorm input, MLP output and decoder output at its true P-1. The hooks return None; they do not replace tensors. All numerical statistics are computed in FP32 from actual FP16 captures.
Full-vocabulary logits and all five actual vectors are saved externally with SHA256. Every metric was replayed exactly. Residual additions equal the correctly FP16-rounded sums bit-exactly; FP32 sum errors are recorded, not suppressed. Undefined zero-vector cosines are null with valid counts, not imputed zero.

| Task | Questions | Unique images |
|---|---:|---:|
| ChooseAttr | 100 | 98 |
| ChooseCat | 100 | 99 |
| ChooseRel | 100 | 99 |
| CompareAttr | 100 | 100 |
| LogicalObj | 100 | 97 |
| QueryAttr | 100 | 100 |

## Regions

Layer statistics are averaged within question before region uncertainty is estimated. All intervals use10,000 whole-image paired bootstrap draws and question-weighted cluster sums/counts. The pooled600 overview equals the equal-task point mean because all tasks have100 questions; global image clustering retains across-task image sharing. Layers and heads are not independent samples.
All CIs are ordinary descriptive95% intervals, not multiplicity-adjusted significance or equivalence evidence.

| Task | Region | Attention norm | MLP norm | Residual norm | Attention ratio | MLP ratio | Net ratio | Cancellation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ChooseAttr | early | 1.30717 | 1.23712 | 2.68428 | 0.67408 | 0.74852 | 1.14730 | 0.72965 |
| ChooseAttr | middle | 10.38061 | 11.86336 | 27.70430 | 0.40114 | 0.45822 | 0.56658 | 0.66679 |
| ChooseAttr | deep | 14.03639 | 46.00712 | 137.45029 | 0.10280 | 0.30116 | 0.33599 | 0.79440 |
| ChooseCat | early | 1.32039 | 1.24280 | 2.67922 | 0.67485 | 0.75051 | 1.14873 | 0.72983 |
| ChooseCat | middle | 9.98444 | 11.58463 | 27.28974 | 0.39010 | 0.45155 | 0.55270 | 0.66446 |
| ChooseCat | deep | 14.05283 | 43.46798 | 130.97679 | 0.10774 | 0.30369 | 0.33689 | 0.78473 |
| ChooseRel | early | 1.33044 | 1.21714 | 2.65155 | 0.68487 | 0.71201 | 1.11958 | 0.72867 |
| ChooseRel | middle | 10.22632 | 12.18403 | 26.10193 | 0.41471 | 0.48665 | 0.58889 | 0.66074 |
| ChooseRel | deep | 12.22495 | 48.42413 | 127.21876 | 0.09592 | 0.34310 | 0.37443 | 0.81918 |
| CompareAttr | early | 1.33299 | 1.21476 | 2.68885 | 0.68231 | 0.73277 | 1.13663 | 0.72471 |
| CompareAttr | middle | 9.16395 | 12.10020 | 25.73340 | 0.37796 | 0.48523 | 0.56131 | 0.65832 |
| CompareAttr | deep | 13.19167 | 44.94681 | 125.23869 | 0.10502 | 0.33300 | 0.36203 | 0.80261 |
| LogicalObj | early | 1.30255 | 1.24426 | 2.69791 | 0.66941 | 0.76192 | 1.15712 | 0.73115 |
| LogicalObj | middle | 9.15204 | 11.93170 | 25.75745 | 0.38733 | 0.48431 | 0.58201 | 0.67715 |
| LogicalObj | deep | 12.59845 | 43.76363 | 123.60081 | 0.10340 | 0.31794 | 0.35248 | 0.79834 |
| QueryAttr | early | 1.34037 | 1.26263 | 2.67710 | 0.68189 | 0.75047 | 1.15030 | 0.72736 |
| QueryAttr | middle | 9.33960 | 11.70795 | 24.43107 | 0.40291 | 0.49341 | 0.58629 | 0.66127 |
| QueryAttr | deep | 10.75678 | 43.20449 | 110.85893 | 0.09679 | 0.35381 | 0.38250 | 0.81891 |
| ALL | early | 1.32232 | 1.23645 | 2.67982 | 0.67790 | 0.74270 | 1.14328 | 0.72856 |
| ALL | middle | 9.70783 | 11.89531 | 26.16965 | 0.39569 | 0.47656 | 0.57296 | 0.66479 |
| ALL | deep | 12.81018 | 44.96903 | 125.89071 | 0.10194 | 0.32545 | 0.35739 | 0.80303 |

## Correctness

Numerical audit: `{"actual_vectors": 96000, "formal_full_forwards": 600, "fp16_rounded_residual_sums_all_exact": true, "historical_joins": 19200, "layers": 19200, "max_historical_full_logit_difference": 0.0, "max_metric_replay_error": 0.0, "max_sum1_fp32_error": 0.03125, "max_sum2_fp32_error": 0.0625, "passed": true, "questions": 600, "raw_metrics_replayed_exact": true}`.
CPU tests precede four-GPU Smoke. Same-run unhooked/observed/unhooked logits must be bit-exact. Historical Full uses the original frozen cross-run tolerances, with actual differences saved per question. All32 layers and160 nodes are required. Input hashes, parameter versions and hook cleanup are checked. Formal collection performs600 forwards total; Smoke alone adds a before/after unhooked reference.
See cpu_validation.json, smoke/, collect/, execution_manifest.json and failure_inventory.json.

## Historical Joins and Figures

Historical BOS/Delta LogP join by task/question/layer, cross-checked against the original checkpoint. A6 distinguishes old500 sensitivity from current100 updates and also shows the matched100 old sensitivity. All sample-level correlations use only strict matched questions. Signed Delta LogP is NOT negated. Per-layer and within-task/layer-centered correlations are descriptive and have no causal p-value.

![A1_update_norms](figures/A1_update_norms.png)

![A2_update_ratios](figures/A2_update_ratios.png)

![A3_update_directions](figures/A3_update_directions.png)

![A4_net_cancellation](figures/A4_net_cancellation.png)

![A5_bos_update_ratio](figures/A5_bos_update_ratio.png)

![A6_ratio_sensitivity](figures/A6_ratio_sensitivity.png)

![A7_six_task_overview](figures/A7_six_task_overview.png)

## Interpretation

See research_findings.md for the measured contrasts. Norms, directions and cancellation are observational; they cannot establish Attention redundancy, causal MLP compensation, safe whole-layer skipping or acceleration. This experiment concerns native P-1 Prefill only in one model, not Decode or cross-model generality.
