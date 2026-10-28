# Phase 3.7: L0-L1 Early Fusion and Information Pathways

**Research status: completed.** Frozen descriptive stratified pilot: 16 B images; 12 real same-image A question pairs.
Baseline `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`; frozen implementation `174014dac2bacb5712d2b7d6bac318ec3e77b5b2`.
Only one GPU; original FP16/eager LLaVA-1.5-7B / Transformers4.49.0. No training, decode, cache, physical exit or extra layer scan.

## Frozen Sampling and Provenance

Within each gold left/right x historical Full correct/wrong stratum, sort abs(Full Margin),qid; split into four contiguous rank bins (numpy.array_split). In each bin prefer a geometry-valid same-length same-P-1-token real pair, then any valid pair, then nearest bin median abs Margin, then qid. One image per bin. No current model outcome used.
Original frozen CVPR-author GQA ChooseRel CSV rows only; same image, distinct question and subject object2 ID, positive visible target ROIs with distinct patch sets. Choose minimum prompt-length difference, then qid. No rewritten questions, labels, boxes, padding or resampling.
Fixed CVPR 2025 author CSV at c9a1111dac4baa175f67716fab473789320187c0, with original GQA IDs/answers/object boxes; not independently verified against original Stanford annotation JSON. Official download page connection timed out during preparation.
The fixed source has 49 geometry-valid pair images among the historical240. The stratified pilot has 12 pairs, 12 with identical prompt length/P-1 position/token. Missing pairs remain missing; no invented replacement.
Historical outcomes are used only for stratification; this is not an unbiased population estimate. Current Native Full is the paired baseline. No statistical-significance claim or post-hoc best condition.

## Experiment B

| Condition | Accuracy | Harmed / recovered / ties | Mean / median Gain | Minimum Gain | Mean absolute gold LogP change |
| --- | ---: | --- | --- | ---: | ---: |
| B0 | 0.5000 | 0 / 0 / 0 | +0.000000 / +0.000000 | +0.000000 | +0.000000 |
| B1 | 0.5000 | 0 / 0 / 0 | -0.143555 / -0.187500 | -1.289062 | +2.958215 |
| B2 | 0.5000 | 0 / 0 / 0 | +0.099609 / +0.121094 | -0.488281 | +1.110692 |
| B3 | 0.5000 | 0 / 0 / 0 | +0.007813 / -0.001953 | -1.218750 | +1.163806 |
| B4 | 0.5000 | 0 / 0 / 0 | +0.130127 / +0.046875 | -0.652344 | +1.477492 |

B1: all valid post-image Text queries -> Image keys; B2: P-1 -> Image; B3: true Question -> Image; B4: P-1 -> true Question. Only L0/L1. P-1 is before the teacher-forced answer, not the answer token itself.

### Left / Right

| Condition | Left accuracy / mean Gain | Right accuracy / mean Gain |
| --- | --- | --- |
| B0 | 0.5000 / +0.000000 | 0.5000 / +0.000000 |
| B1 | 0.5000 / -0.633789 | 0.5000 / +0.346680 |
| B2 | 0.5000 / -0.034668 | 0.5000 / +0.233887 |
| B3 | 0.5000 / -0.340332 | 0.5000 / +0.355957 |
| B4 | 0.5000 / +0.000488 | 0.5000 / +0.259766 |

## Experiment A

Per real query and head normalize visual probabilities, then average queries within Question, then heads. P-1 is one query. JS is sqrt(Jensen-Shannon divergence), natural log, no pseudocount; cosine compares pooled spatial maps. Raw visual Mass is separately head/query averaged.
B: compare each token to its own Full then average Question token cosine distances and change norms. A: mean-pool real Question token hidden states separately per question, then compare pooled vectors; never align different questions by absolute token index. P-1 is a single vector. Capture decoder block outputs (after residual/MLP).
Each heatmap labels raw visual Mass separately from conditional spatial distribution. Both subject boxes are shown, with shared scales for all raw panels and all conditional panels within a pair.

| Layer / Query | Mean spatial JS distance | Mean cosine | Q1 / Q2 visual Mass | Q1 / Q2 normalized entropy | Mean paired own-vs-other ROI preference |
| --- | ---: | ---: | --- | --- | ---: |
| L0_question | 0.000515847 | 0.999998 | 0.829093 / 0.829071 | 0.955899 / 0.956034 | -9.07552e-05 |
| L0_answer | 2.10263e-05 | 1 | 0.699736 / 0.699668 | 0.956697 / 0.956697 | +3.35643e-06 |
| L1_question | 0.00121251 | 0.999992 | 0.437931 / 0.437127 | 0.941093 / 0.941183 | +0.000570886 |
| L1_answer | 0.000555436 | 0.999998 | 0.413467 / 0.413473 | 0.945649 / 0.945593 | -5.6431e-05 |

ROI preference = half the sum of (own target enrichment - other question target enrichment) for Q1 and Q2. Enrichment divides conditional ROI probability by ROI patch fraction; box overlaps and empty background are retained, not invented. Values are descriptive, not a validated test of task selection.
All 12 matched pairs pass the L0 P-1 input / visual-input / visual-QK and FP32 visual-only distribution exact checks. Any nonzero spatial discrepancy from normalized stored FP16 probabilities is not evidence against the structural L0 invariance.

## Seven Research Questions

1. Task specificity: L0 P-1 spatial selectivity cannot depend on changed Question content under the matched conditions, as checked directly. Real Question queries differ lexically, so their L0 map differences alone do not isolate task-dependent visual reasoning. L1 can incorporate prior-layer context; use paired ROI preference and heatmaps, not JS difference alone, to assess alignment. Strong universal spatial-selection claims remain unsupported by this small pilot.
2. Early all-Text visual blocking B1: mean Gain -0.143555, median -0.187500, harmed/recovered 0/0. This is the measured local causal effect, not evidence about removing visual tokens.
3. P-1 direct image reading B2 versus Question image reading B3: mean Gain +0.099609 versus +0.007813. Compare absolute per-image changes and hidden drifts as well as signs; a larger beneficial mean is not necessarily a larger required information contribution.
4. Early P-1 -> Question B4: mean Gain +0.130127, harmed/recovered 0/0. Other text keys and all later layers remain accessible. This does not test absence of all textual dependence.
5. Support for early global fusion: diffuse/high-entropy maps, matched L0 P-1 spatial invariance and broad redistribution are compatible with early fusion, but entropy alone is not proof of global semantic integration. Contrast each actual result with the figures and numeric tables.
6. Contradictory evidence must be retained: positive task-aligned ROI preference at either layer, sharp maps, adverse per-image mask effects, and differences between Question/P-1 pathways would limit an undifferentiated global-fusion claim. Worst losses and every paired sample remain in CSV/summary; no result-based filtering.
7. L2-and-later compensation is possible: every later layer still has native masks and all visual tokens. The present design does not intervene on compensation or isolate alternative text routes. Lack of accuracy harm does not establish that a pathway has no function.

Attention association, local causal intervention, Hidden change and final prediction are distinct evidence. Hidden drift is not information quality; B2 is not a Question-only visual route, B3 is not a visually empty Question, and B4 is not removal of all text. No automatic extension is authorized.

## Audit and Numerical Differences

Recorded B attention audits: 5120; local masked-layer/candidate audits: 256. CPU and two-image GPU smoke gates passed before the formal pilot. No-op valid-text logits and post-cleanup Native Full are exact.
All masks reuse the existing clone/-inf implementation. Blocked target attention is zero; causal/padding constraints and nonempty legal rows hold; non-target queries are locally exact; L2-L31 masks and all 576 visual tokens are retained. Softmax redistribution of unblocked keys is expected and recorded.
Full before/after model parameter+buffer byte hashes match; input/parameter versions and installed Transformers files are unchanged. Exception cleanup is CPU-tested. No failure is silently counted as complete. Historical Full Margin differences are recorded per image; they are never substituted for current paired Full or fitted with new tolerances.

## Artifacts

[Per-image B scores and drift/mass metrics](per_sample_b.csv), [per-pair A metrics](per_pair_a.csv), [complete descriptive summary and worst cases](summary.json), [frozen manifest](manifest.json).
Raw observations, complete candidate mappings, compressed audits, source/model/input hashes and environment are under pilot/ and smoke/. Reproduce with the frozen implementation commit, not a later result-only commit.

![Paired Gain](paired_margin_gain.png)
![Margins versus Full](margin_vs_full.png)
![Early hidden drift](early_hidden_drift.png)
![Local redistribution](local_mass_redistribution.png)
![Actual group Mass](full_vs_mask_group_mass.png)

![A image 2340966](heatmaps/02599393.png)
![A image 1987](heatmaps/04468301.png)
![A image 2363550](heatmaps/01851960.png)
![A image 2360994](heatmaps/00223311.png)
![A image 2355857](heatmaps/10541771.png)
![A image 2392833](heatmaps/05862856.png)
![A image 2342950](heatmaps/03539611.png)
![A image 2412075](heatmaps/16312698.png)
![A image 2343687](heatmaps/1351224.png)
![A image 2344578](heatmaps/1228576.png)
![A image 2363722](heatmaps/02159371.png)
![A image 2355633](heatmaps/11283646.png)
