# Attention Update vs Text Source: Three Depth Regions

Independent branch: `codex/attention-text-region-ablation`.
Frozen GPU code: `131f6bf82aa3bb3f25967f10ad1dbc3567fc4fa2`.

- GQA: 600 original questions, six tasks x100, 600 distinct image SHA256s.
- COCO: the original 60 images, prompts and five reference captions.
- All 4,620 generations are new, with Full and six conditions paired on the
  same assigned physical GPU. No historical Full scores are substituted.
- L0 stays native. Early=L1-L10, Middle=L11-L20, Late=L21-L31.
- GQA intervenes at original P-1 and true Decode; COCO intervenes at true
  Decode only, leaving the entire Prefill native.

## Outputs

- `report_zh.md`: Chinese protocol, measured summaries and interpretations.
- `gqa_summary.csv`, `gqa_task_accuracy.csv`, `gqa_per_question.csv`.
- `coco_summary.csv`, `coco_per_image.csv`.
- `same_region_comparisons.csv`: Text OFF minus Attention OFF, paired by image.
- `coco_object_action_terms.csv`, `coco_object_action_change_counts.csv`:
  descriptive noun/verb lemma changes, NOT grounded error/hallucination scores.
- `qualitative_notes_zh.md`: manual inspection of six original images and
  all seven captions, including failure cases and counterexamples.
- `figure_gqa_accuracy`, `figure_coco_cider`, `figure_coco_spice`,
  `figure_cross_task_regions`, `qualitative_six_coco`: PDF and PNG.
- `protocol.json`, `sample_manifest.json`, CPU/GPU validation, checkpoints,
  execution logs, `delivery_audit.json`, `gpu_release.json`, artifact hashes.

Accuracy and its changes use percent / percentage points. CIDEr and SPICE use
the original official raw scales. COCO relative drop is
`100 * (Full - intervention) / Full`; a negative value means a point gain.
Bootstrap uses entire image SHA256 clusters. CIDEr intervals condition on the
original 60-image reference corpus document frequencies, as in prior work.
All comparisons are exploratory, with ordinary uncorrected 95% intervals.

`physical_gpu` is the actual generation device. All conditions for an image
share it. First-token LogP Loss tracks Full's Top1, not the gold answer.
First divergence is zero-based among generated tokens; `-1` means identical.
For COCO, a changed sequence can only first diverge at index 1 or later.
Token trigram repetition is a syntactic screening flag, not a quality metric.

Attention OFF zeros target-query `o_proj` outputs. Text OFF hard-masks text
keys before Softmax and retains BOS/Image/current Self. Neither is Image
source OFF, whole-block skipping, physical pruning or a measured speedup.
