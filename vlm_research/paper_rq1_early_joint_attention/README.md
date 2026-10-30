# Early Joint Attention Ablation

Independent worktree from `f469eedbc960a0a1cd73661723e70c1ff46a6b1f`.
Only this new package, its tests and its new results directory may change.

## Pre-Registered Core

Exactly S0--S10 in `protocol.py`; no optional L14--L16 reference.
Every condition performs its own actual prefill and own-prefix greedy generation.
All target layers are hooked simultaneously. Only Attention `o_proj` outputs change;
MLP, LayerNorm, residuals, parameters and other layers remain native.
This still computes Attention; no speedup or actual computation skipping is claimed.

Model/prompt/processor/official primary GQA scoring are inherited unchanged.
PyArrow data inspection uses AFAC only; frozen inference environment is unchanged.
Image IDs and exact SHA256 in all prior available experiment manifests/configs are
conservatively excluded, including reserved/not-executed images. SHA/content identity
is not a claim of perceptual near-duplicate removal.

Available independent cohort before any new GPU output:

| Task | Requested | Frozen |
|---|---:|---:|
| ChooseAttr | 500 | 500 |
| ChooseCat | 500 | 353 |
| ChooseRel | 500 | 500 |
| CompareAttr | 500 | 6 |
| LogicalObj | 500 | 0 |
| QueryAttr | 500 | 500 |

1,859 questions / 1,453 images, two additional image-disjoint Smoke questions.
There is no exposed-data fallback or synthetic supplementation. The absent
LogicalObj and six-question CompareAttr prevent a complete six-task confirmation.
The pooled score is the achieved question-weighted mixture, not a balanced3000 score.

Statistics freeze before GPU scoring: 200,000 image-cluster paired bootstrap draws;
ordinary95% and separate two-sided Bonferroni95% families:8 pooled primary comparisons,
2 pooled auxiliary comparisons,60 taskwise comparisons,6 planned pooled contrasts.
Contrasts are S1-S5, S2-S6, S3-S7, S4-S8, S2-S1 and S6-S5. Direction is literal first
minus second. Preserve whole-image cluster sum/count ratios, not averaged question CIs.
Pooled NI requires corrected lower bound strictly above -1pp. Taskwise -3pp is only
a heterogeneity risk screen. No global simultaneous inference across these families.
Degenerate zero-discordance bootstrap intervals do not establish zero population risk.

Use physical GPUs1/2/3 only; GPU0 belongs to other work. All conditions for a given image
use one deterministic shard, atomic immutable condition checkpoints and a worker lock.
Commit collector and data first, then freeze execution binding, then real Smoke on all
three GPUs. Any correctness gate failure blocks collection, without tolerance changes.
Supplementary POPE/ScienceQA/TextVQA checks are not part of this core execution and must
remain explicitly marked unexecuted; no general cross-task safety claim is allowed.

## Commands

From `vlm_research`, using the pinned Python:

```bash
python -m paper_rq1_early_joint_attention.data
python -m paper_rq1_early_joint_attention.run cpu
# Commit collector/tests/config/manifests.
python -m paper_rq1_early_joint_attention.run freeze
python -m paper_rq1_early_joint_attention.launch smoke generation
python -m paper_rq1_early_joint_attention.analyze
```

Source hashes bind collector files, shared dependencies and tests. Postprocessing may
be implemented while the frozen collector runs, but collector/shared code must not change.
Failed attempts stay in `failures/`; valid immutable checkpoints are never overwritten.
