# Research Readout: Six GQA Tasks, Single-Layer Attention Ablation

Status: **complete**, 2026-10-30, Asia/Shanghai. This readout supplements the
immutable, automatically audited [full report](report.md); it does not replace
or modify the frozen protocol, completed results, or their original hashes.

## Scope and Coverage

- Model: LLaVA-1.5-7B, revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`,
  FP16/eager, Transformers 4.49.0. Frozen code:
  `934335e78c8f3d1f11836dc815ded3d88b8b0884`; seed `20261030`.
- Original GQA Balanced validation, task-filtered without a correctness
  prefilter. Each task has 500 questions; 80 per task were independently
  hash-selected from those 500 before any GPU output.
- Total: 3,000 questions, 2,409 unique image IDs and 2,409 unique image SHA256s.
  No different image IDs share an image SHA256. Cross-task image overlap and
  per-GPU counts are recorded in [data_manifest.json](data_manifest.json).
- Main: 3,000 Full and 96,000 independent single-layer conditions. Generation:
  480 Full and 15,360 single-layer conditions. No questions or conditions are
  missing, substituted, or excluded on the basis of results.
- Only the original Prompt P-1 `self_attn.o_proj` output at one layer is zeroed.
  MLP and other queries remain native. Generation later uses native Decode
  with each condition's own cache. This is not attention-edge knockout,
  whole-layer skipping, or an implementation of faster inference.

## First-Token Sensitivity

The table selects the minimum mean Delta LogP over the 32 measured layers
within each task. These are descriptive extrema of this cohort, not an
independent validation of a selected layer. Delta LogP always tracks the same
native Full Top-1 token, including when that token belongs to a wrong answer.

| Task | Questions / images | Most sensitive layer | Mean Delta LogP [ordinary 95% CI] | Mean relative probability change | Top-1 flips / 500 |
|---|---|---|---|---|---|
| ChooseAttr | 500 / 481 | L14 | -0.05655 [-0.06738, -0.04585] | -4.81% | 34 |
| ChooseCat | 500 / 471 | L14 | -0.05958 [-0.07050, -0.04955] | -5.16% | 18 |
| ChooseRel | 500 / 455 | L15 | -0.07173 [-0.08262, -0.06142] | -6.28% | 35 |
| CompareAttr | 500 / 490 | L16 | -0.03310 [-0.03983, -0.02714] | -3.03% | 5 |
| LogicalObj | 500 / 477 | L14 | -0.01443 [-0.02035, -0.00871] | -1.23% | 6 |
| QueryAttr | 500 / 485 | L14 | -0.03697 [-0.04172, -0.03225] | -3.49% | 14 |

All six task-level minima occur at L14-L16, with different effect sizes.
This supports nonuniform *first-token output-update sensitivity* in this
model/cohort; it is not a statement about every head, every query, or Decode.

Layers closest to zero by **absolute mean** Delta LogP are ChooseAttr L3,
ChooseCat L10, ChooseRel L27, CompareAttr L5, LogicalObj L21, and QueryAttr L9.
This metric can hide cancellation: QueryAttr L9 has mean Delta LogP about
-0.000086, yet changes the Top-1 token on 10/500 questions. Small average
change is not sufficient evidence of redundancy or safe removal.

Complete signed effects, medians, JS and ordinary confidence intervals for
all 192 task-layer combinations remain in
[first_token_layer_summary.csv](first_token_layer_summary.csv).

## BOS Mass and Necessity

Native BOS mass is low at L0-L1 and already high at several early and late
layers. Its task-level peak is at ChooseAttr L27 (0.86635), ChooseCat L5
(0.86154), ChooseRel L25 (0.92237), CompareAttr L25 (0.86734), LogicalObj L27
(0.86112), and QueryAttr L25 (0.92531). Thus peak BOS mass is not universally
late, nor does its position identify the most sensitive layer.

Correlations below use **negative Delta LogP** as sensitivity: a larger value
means a larger loss of the native Full token's log probability.

| Task | Across-layer Pearson / Spearman | Within-layer-centered Pearson / Spearman |
|---|---|---|
| ChooseAttr | -0.092 / -0.095 | -0.011 / -0.029 |
| ChooseCat | -0.187 / -0.225 | -0.030 / -0.042 |
| ChooseRel | -0.124 / -0.280 | -0.042 / -0.037 |
| CompareAttr | -0.167 / -0.277 | +0.001 / +0.007 |
| LogicalObj | -0.183 / -0.349 | -0.028 / -0.041 |
| QueryAttr | -0.200 / -0.309 | -0.009 / -0.006 |

There is a weak negative across-layer linear association, but little
within-layer linear or rank association in these data. High BOS mass alone
does not provide strong evidence of low causal necessity. Layer depth,
residual updates and task differences can confound visual alignment; repeated
image/layer observations are not independent. These are descriptive
correlations, with no p-value or causal claim.

## Exploratory Full-Answer Generation

These are the predefined 80 questions/task from the main cohort, **not an
independent Confirmation set**. The following table aligns generation results
with each task's main sensitivity minimum above, rather than selecting a
different accuracy-favorable layer. All 32 layers remain in the CSV/figure.

| Task | Full | Ablated layer | Ablated accuracy | Delta pp [ordinary paired 95% CI] | Correct to wrong | Wrong to correct |
|---|---|---|---|---|---|---|
| ChooseAttr | 80.00% | L14 | 81.25% | +1.25 [-5.06, +7.69] | 3 | 4 |
| ChooseCat | 90.00% | L14 | 85.00% | -5.00 [-10.13, -1.23] | 4 | 0 |
| ChooseRel | 85.00% | L15 | 78.75% | -6.25 [-12.66, 0.00] | 6 | 1 |
| CompareAttr | 83.75% | L16 | 82.50% | -1.25 [-3.80, 0.00] | 1 | 0 |
| LogicalObj | 95.00% | L14 | 91.25% | -3.75 [-8.75, 0.00] | 3 | 0 |
| QueryAttr | 85.00% | L14 | 85.00% | 0.00 [-5.00, +5.00] | 2 | 2 |

- The largest observed task/layer accuracy loss is ChooseRel L15: -6.25 pp,
  comprising six harmed and one corrected question. This is exploratory,
  not a multiplicity-adjusted inferential claim.
- A zero net change can hide harm: QueryAttr L14 harms two and corrects two.
- A lower native-token probability need not harm answer accuracy. LogicalObj
  L16 improves this subset from 95.00% to 98.75% (three corrections, zero harms;
  ordinary paired CI for the +3.75 pp difference: [0.00, +8.75]). The native
  Full token may itself be wrong. This example is not evidence of a generally
  beneficial intervention or a basis for selecting an operational layer.
- Actual Full-correct/wrong subgroup curves use these 80 complete Full
  generations only. In particular LogicalObj has only four Full-wrong
  questions, so its subgroup effects are highly uncertain.

## Validation, Statistics and Time

- Ten CPU tests passed before execution. Four same-question GPU Smoke runs
  passed the frozen tolerances; native/observed/cached/No-op logits were
  bit-exact on each card, and intervened cached generation matched the
  independent native `generate` reference.
- All 99,000 Full/ablated full-vocabulary FP32 vectors were replayed. Maximum
  replay difference for every recorded numerical metric was **0.0**.
- All 15,840 generated conditions were rescored. All 33 generation Prefill
  vectors per selected question exactly matched that question's main result.
- All 3,548 files listed by the original completion manifest passed the
  subsequent hash check. Frozen source verification passed. Four GPU workers
  finished without runtime failures, OOM, skipped questions or leaked hooks.
- One initial CPU attempt failed because SciPy was absent. Before code freeze,
  the dependency was replaced with tested NumPy Pearson/tie-aware Spearman;
  the failed attempt remains recorded. No GPU tolerance was relaxed.
- All intervals use 10,000 paired image-cluster bootstrap draws, retaining
  question weighting for unequal cluster sizes. They are ordinary descriptive
  95% intervals across 192 primary task-layer comparisons, **not** family-wise
  confidence or evidence of equivalence when they include zero.
- Longest worker compute wall time, excluding model load: main 6,028.95 s
  (100.48 min), generation 1,345.17 s (22.42 min), Smoke 11.21 s. Worker times
  overlap and must not be added across the four cards.
- All four GPUs were released after generation. No training, additional
  intervention, Caption restart or follow-up experiment was executed.

## Artifact Index

- [Automatically audited report and seven PNG figures](report.md)
- [All PNG/PDF figures](figures/)
- [First-token per-question results](first_token_per_question.csv)
- [First-token layer summaries](first_token_layer_summary.csv)
- [BOS per-question and per-head results](bos_mass_per_question.csv)
- [BOS layer summaries](bos_mass_layer_summary.csv)
- [BOS descriptive correlations](bos_sensitivity_correlations.csv)
- [Complete-generation paired answers and token IDs](full_generation_80_per_question.csv)
- [Complete-generation layer summaries](full_generation_80_layer_summary.csv)
- [Actual Full-correct/wrong subgroup summaries](first_token_by_full_generation_group_summary.csv)
- [Numerical replay](numerical_audit.json), [completion hashes](completion.json),
  [failure inventory](failure_inventory.json), [execution binding](execution_manifest.json)

External raw FP32 NPZ files, with paths/SHA256 in per-question checkpoints:
`/data/lk/ATE-resources/paper_rq1_single_layer_gqa6/raw/`.
They are not committed as large model/data artifacts.
