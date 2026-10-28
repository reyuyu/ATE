# Phase 3.6 Reviewed Interpretation

## Scope and Evidence

Completed on 2026-10-29, Asia/Shanghai. All 240 frozen Phase 3.2 images were
run once in four disjoint 60-image shards, sorted by string question ID.
All workers used commit `0d323e1bb3eb6f91443854e90242150b04e9eaf1`.
The master aggregated only after all four complete gates passed.

There are six research conditions, Native Full, and two lambda0 controls:
2,160 score rows and 29,760 formal per-candidate layer audits. The two-image
smoke adds 248 audits, separate from the formal denominator. No training,
decoding, KV cache, Physical rerun, additional dataset or post-hoc tuning.

Primary analysis excludes discovery image 15133706 (n=239). The separately
reported 187 images had no Phase 3.4 task Attention cache. Their own fresh
Full L23/L24 references were captured, as for the other 53 images. A cached
image is not automatically a previously intervened image. All 240 images
and the single discovery case are supplemental, not substitutes for n=239.

| ID | Source | Target layers, inclusive | Lambda |
| --- | --- | --- | ---: |
| C1 | Full L23 | L24-L31 | 0.5 |
| C2 | Full L23 | L24-L31 | 1 |
| C3 | Full L24 | L24-L31 | 0.5 |
| C4 | Full L24 | L24-L31 | 1 |
| C5 | Full L23 | L25-L31 | 1 |
| C6 | Full L24 | L25-L31 | 1 |

All log quantities below are natural logarithms. Correctness is Margin>0;
ties count as wrong. Full-vocabulary normalization, original prompt and
complete candidates are retained. These frozen candidates each contain one
token; the runner checks this fact rather than truncating their sequences.

## Answers to the Six Research Questions

1. **Not confined to 15133706 at the Margin level.** C1/C2/C5 improve Margin
   in 194/195/196 of 239 images, respectively. All paired mean-gain intervals
   are positive. The 187-image subgroup independently has positive means and
   150/151/152 positive-gain images. This is evidence on this frozen cohort,
   not proof of an unrestricted capability improvement.
2. **L23 exceeds L24 in matched mean Gain**, for all three frozen pairings.
   Paired differences and intervals are below. It does not win on every image
   and is not uniformly safer by every metric.
3. **There is a descriptive left/right asymmetry.** Left mean Gain exceeds
   right mean Gain in all six conditions, much more strongly for L24. Native
   Full is already 115/120 correct on left versus 73/119 on right. Label and
   initial correctness are imbalanced; these summaries do not isolate a
   causal directional bias. L23 is less asymmetric, not bias-free.
4. **Accuracy harm is absent for L23 here, not all harm.** L23 preserves all
   188 initially correct primary images, but lowers Margin on 7/13/8 of
   them in C1/C2/C5. Each L24 condition harms one initially correct image.
   None of the six conditions recovers any of the 51 primary Full errors.
5. **Gain correlates strongly with historical Physical K24 Gain**, but does
   not establish an identical mechanism. Primary Pearson is 0.839-0.877 and
   Spearman 0.820-0.854; these use historical Physical baselines versus
   current same-machine inheritance baselines. Hardware differences and
   the shared task/margin structure limit causal interpretation.
6. **The broad accuracy/absolute-probability claim is not supported.**
   The relative Margin phenomenon generalizes within this frozen cohort,
   and L23 is better than L24 on the specified paired comparisons. However,
   the primary accuracy does not increase, initially wrong images worsen
   on average, and mean absolute gold LogP falls in every condition.
   Do not promote this into a generally useful correction or an explanation
   of Physical Exit. Stop automatic expansion here. Any narrower mechanistic
   follow-up needs a new explicitly authorized protocol.

## Primary Results: 239 Images

Native Full: 188/239 correct (78.6611%). L23 accuracy remains 188/239;
L24 becomes 187/239 (78.2427%). Each six-condition exact paired McNemar
two-sided p-value is 1.0; there is no evidence of accuracy improvement.
The paired accuracy-difference percentile interval is [0,0] for L23 and
[-1.2552,0] percentage points for L24. A degenerate empirical bootstrap
interval is not a population guarantee of zero harm.

| ID | Mean Gain [95% paired CI] | Median | P5 / P95 | Min | Positive / negative / zero |
| --- | --- | ---: | --- | ---: | --- |
| C1 | +0.049572 [+0.043246,+0.055864] | +0.050781 | -0.039062 / +0.125391 | -0.093750 | 194 / 35 / 10 |
| C2 | +0.100876 [+0.087996,+0.113788] | +0.097656 | -0.070703 / +0.258594 | -0.191406 | 195 / 41 / 3 |
| C3 | +0.032868 [+0.027115,+0.038523] | +0.031250 | -0.046875 / +0.101562 | -0.105469 | 182 / 54 / 3 |
| C4 | +0.066586 [+0.054802,+0.078077] | +0.066406 | -0.093750 / +0.207031 | -0.222656 | 183 / 53 / 3 |
| C5 | +0.091642 [+0.079808,+0.103475] | +0.089844 | -0.062500 / +0.246094 | -0.152344 | 196 / 36 / 7 |
| C6 | +0.066586 [+0.054802,+0.078077] | +0.066406 | -0.093750 / +0.207031 | -0.222656 | 183 / 53 / 3 |

| Matched pairing | Mean L23-L24 Gain [95% paired CI] | L23 wins / losses / ties |
| --- | --- | --- |
| C1-C3 | +0.016704 [+0.012846,+0.020545] | 161 / 70 / 8 |
| C2-C4 | +0.034290 [+0.026118,+0.042544] | 156 / 79 / 4 |
| C5-C6 | +0.025056 [+0.017585,+0.032558] | 147 / 86 / 6 |

## Direction and Initial Correctness

| ID | Left / right mean Gain | Left / right accuracy | Full-correct / Full-wrong mean Gain | Weak-Margin mean Gain |
| --- | --- | --- | --- | ---: |
| C1 | +0.057064 / +0.042017 | 95.8333% / 61.3445% | +0.066032 / -0.011106 | +0.017578 |
| C2 | +0.111589 / +0.090074 | 95.8333% / 61.3445% | +0.133768 / -0.020374 | +0.040365 |
| C3 | +0.060091 / +0.005416 | 95.8333% / 60.5042% | +0.049057 / -0.026808 | -0.010742 |
| C4 | +0.122005 / +0.010701 | 95.8333% / 60.5042% | +0.099838 / -0.055990 | -0.020833 |
| C5 | +0.103092 / +0.080095 | 95.8333% / 61.3445% | +0.122029 / -0.020374 | +0.030599 |
| C6 | +0.122005 / +0.010701 | 95.8333% / 60.5042% | +0.099838 / -0.055990 | -0.020833 |

Stratum sizes are left120/right119, Full-correct188/Full-wrong51,
weak abs(Full Margin)<=0.15 n=12. Weak accuracy is 8/12 for Full and L23,
7/12 for L24. Per-stratum intervals and counts are in aggregate.json.

For lambda0 -> 0.5 -> 1 at L24-L31, L23 has 191 both-positive, 34
both-negative, 2 both-zero and 3 opposite-sign images. Its nondecreasing
count is 196 and nonincreasing count 41, with all-zero images in both.
L24 has 182 both-positive, 53 both-negative, 2 both-zero and no opposite-sign
images; corresponding monotonic counts are 185 and 55. Zero/nonzero cases
need not fall in the strictly both-positive/both-negative categories.
Stronger lambda amplifies negative as well as positive effects. No best
lambda is selected after seeing these results.

## Absolute Gold Probability and Physical Correlation

| ID | Mean gold LogP change [95% paired CI] | Gold LogP increases /239 | Physical Pearson [95% CI] | Spearman [95% CI] |
| --- | --- | ---: | --- | --- |
| C1 | -0.129813 [-0.135743,-0.123702] | 0 | 0.8773 [0.8428,0.9041] | 0.8544 [0.8075,0.8893] |
| C2 | -0.219082 [-0.231301,-0.206418] | 3 | 0.8701 [0.8344,0.8981] | 0.8453 [0.7971,0.8814] |
| C3 | -0.112226 [-0.118330,-0.106207] | 1 | 0.8389 [0.7992,0.8724] | 0.8196 [0.7640,0.8617] |
| C4 | -0.223978 [-0.236323,-0.211669] | 1 | 0.8466 [0.8088,0.8786] | 0.8274 [0.7718,0.8687] |
| C5 | -0.185814 [-0.197643,-0.174005] | 6 | 0.8660 [0.8284,0.8954] | 0.8396 [0.7889,0.8773] |
| C6 | -0.223978 [-0.236323,-0.211669] | 1 | 0.8466 [0.8088,0.8786] | 0.8274 [0.7718,0.8687] |

Mean LogP is not the arithmetic mean probability. Positive Margin change
can result from the wrong candidate losing still more probability; it is
not evidence that the gold candidate gains absolute probability.

## Previously Uncached 187 Images

Native Full is 151/187 (80.7487%). All six conditions retain exactly this
accuracy, with no correct/wrong flips. All three paired source differences
have positive mean-gain intervals: C1-C3 +0.016732 [0.012408,0.021035],
C2-C4 +0.034634 [0.025400,0.043638], C5-C6 +0.025109 [0.016753,0.033527].

| ID | Mean Gain [95% paired CI] | Positive /187 | Mean gold LogP change |
| --- | --- | ---: | ---: |
| C1 | +0.052118 [+0.044661,+0.059408] | 150 | -0.129302 |
| C2 | +0.106054 [+0.090825,+0.120906] | 151 | -0.217793 |
| C3 | +0.035386 [+0.028910,+0.041987] | 149 | -0.112359 |
| C4 | +0.071420 [+0.058112,+0.084977] | 150 | -0.224058 |
| C5 | +0.096528 [+0.082512,+0.110170] | 152 | -0.183796 |
| C6 | +0.071420 [+0.058112,+0.084977] | 150 | -0.224058 |

The subgroup is left96/right91, not rebalanced or resampled. Full left/right
accuracy is 91/96 and 60/91. Complete left/right and correctness strata,
lambda directions, physical correlations, paired statistics and tails are
saved for this cohort in aggregate.json.

## Supplemental 240 and Discovery Case

Full is 188/240 (78.3333%). C1/C2/C5 are 189/240 (78.75%): each additional
correct prediction is **only the discovery case**. C3/C4/C6 are 187/240
(77.9167%). These supplemental numbers must not replace primary accuracy.

| ID | Mean Gain, all240 | Positive /240 | Margin on 15133706 |
| --- | ---: | ---: | ---: |
| C1 | +0.049593 | 195 | +0.023438 |
| C2 | +0.100944 | 196 | +0.085938 |
| C3 | +0.032731 | 182 | -0.031250 |
| C4 | +0.066276 | 183 | -0.039062 |
| C5 | +0.091683 | 197 | +0.070312 |
| C6 | +0.066276 | 183 | -0.039062 |

Discovery Full Margin is -0.03125. Its originally studied L24-L31 Margin
pattern reproduces, but the n=1 bootstrap is degenerate and not inferential.

## Tail Losses and Correctness Harm

No sample reaches the pre-frozen severe Gain<=-0.5 threshold. This does
not mean there are no adverse effects. Worst primary image is 16312698
(right): C1 -0.093750, C2 -0.191406, C3 -0.105469, C4 -0.222656,
C5 -0.152344, C6 -0.222656. Worst ten per condition and cohort, with full
scores, are saved in aggregate.json; every negative point remains in plots.

Image 04132006 (right) has current Full Margin +0.015625. C3 changes it to
0 (tie, wrong); C4/C6 to -0.0234375. It is the only primary accuracy harm.
This same weak sample is the historical/current Full disagreement below.

## Correctness and Numerical Audit

- 121 CPU regression tests pass with no skips; original inheritance tests
  pass 7/7. Exception cleanup is tested as well as normal cleanup.
- All 240 formal samples have exact same-machine lambda0 effective-text
  full-vocabulary logits, exact Question hidden invariance, candidate P-1
  agreement, final Native Full restoration, unchanged input tensors and
  third-party source, and complete hook removal.
- All 29,760 layer audits pass the unchanged original algorithm checks:
  per-head target query only, other queries/nonvisual entries bitwise
  unchanged, finite nonnegative values, exact rounded intervention formula,
  visual Mass, row sums and original dtype-derived FP16 rounding bound.
- Maximum row-sum error is 0.00036209821701049805; maximum visual-Mass
  absolute error is 0.00019592046737670898. Every actual head passes the
  existing absolute/relative Mass tolerance; no tolerance was enlarged.
- **6,738 audits exceed the old nominal conditional-L1 warning threshold.**
  They remain explicitly recorded, not deleted or relabeled as zero warnings.
  The already-validated original FP16 dtype-derived bound passes in every
  case. The distinction is inherited from the original single-case method,
  not a threshold fitted to this batch.
- No failed attempt, NaN, Inf, OOM or input contamination is recorded. All
  four model processes exited normally; no other GPU job was terminated.
- C4/C6 have identical full prediction-logit hashes and candidate scores
  for all 240 images. This is expected: at native L24 the same-image L24
  reference is its own distribution before any earlier intervention.
  The two pre-specified windows remain separately reported, not removed.

## Historical Full and Cache Differences

Historical environments record RTX4090 / Python3.11.18; this server uses
four RTX3090 / Python3.11.15. The ten pinned packages, model revision and
canonical file hashes are fixed; FP16/eager, deterministic algorithms and
TF32-disabled settings are reused unchanged. Exact historical input token
and candidate mappings and original image hashes are verified.

None of 240 historical Full candidate LogP pairs is bitwise identical to
current Full. Maximum absolute Full Margin difference is 0.140625. One
correctness difference occurs: 04132006 historical -0.0078125 versus
current +0.015625. All 53 historical cache comparisons are non-bitwise;
maximum L23/L24 raw Attention element difference is 0.011021614074707031.
Per-image discrepancies and cache hashes are retained in aggregate.json
and per-image audits. GPU architecture and Python patch differences are
documented, but their individual numerical contributions were not isolated;
do not claim a proven cause.

Historical values are not silently substituted into current baselines and
no tolerance is fitted to make them agree. Each reference comes from its
own current Full. Same-machine no-op, causal and local invariants remain
bitwise or original-bound mandatory. Physical K24 values remain explicitly
historical; no Physical computation is presented as a current rerun.

## Statistics and Artifact Integrity

Bootstrap uses 5,000 fixed-seed image-level paired resamples; heads and layers
are never independent statistical samples. Intervals are exploratory and
not multiplicity adjusted. Near-zero FP16 outcomes and tied scores require
caution. Each frozen condition is reported; there is no per-image best
lambda or post-hoc condition selection.

The original report.md and summary_validation.json are unmodified generated
outputs. This review is a separate interpretive artifact. The recursive
publication audit additionally binds these review/publication files,
per-image references, logs, manifests and environment records. It does not
replace the frozen protocol or constitute a second scientific aggregation.
