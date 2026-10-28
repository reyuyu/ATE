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

| Condition | Accuracy | Harmed / recovered / ties | Mean / median Gain | Minimum Gain | Mean signed gold LogP change (full vocabulary) |
| --- | ---: | --- | --- | ---: | ---: |
| B0 | 0.5000 | 0 / 0 / 0 | +0.000000 / +0.000000 | +0.000000 | +0.000000 |
| B1 | 0.5000 | 0 / 0 / 0 | -0.143555 / -0.187500 | -1.289062 | +2.958215 |
| B2 | 0.5000 | 0 / 0 / 0 | +0.099609 / +0.121094 | -0.488281 | +1.110692 |
| B3 | 0.5000 | 0 / 0 / 0 | +0.007813 / -0.001953 | -1.218750 | +1.163806 |
| B4 | 0.5000 | 0 / 0 / 0 | +0.130127 / +0.046875 | -0.652344 | +1.477492 |

B1: all valid post-image Text queries -> Image keys; B2: P-1 -> Image; B3: true Question -> Image; B4: P-1 -> true Question. Only L0/L1. P-1 is before the teacher-forced answer, not the answer token itself.

Gold LogP change above is a signed difference in the gold candidate's absolute, full-vocabulary log probability, not the absolute value of that difference. Its increase must not be confused with a better gold-versus-wrong ranking: accuracy is unchanged for every condition.

### Direct Native Full Comparison

| Condition | Correct / N | Mean final Margin | Mean Gain vs Full | Mean per-image absolute Gain | Mean gold absolute LogP |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 Native Full | 8/16 | +0.662354 | 0 | 0 | -17.169756 |
| B1 Text->Image | 8/16 | +0.518799 | -0.143555 | 0.545898 | -14.211541 |
| B2 P-1->Image | 8/16 | +0.761963 | +0.099609 | 0.307617 | -16.059063 |
| B3 Question->Image | 8/16 | +0.670166 | +0.007813 | 0.649902 | -16.005950 |
| B4 P-1->Question | 8/16 | +0.792480 | +0.130127 | 0.312256 | -15.692263 |

Full and every intervention are paired on the same image, question, candidate tokens, model and scoring protocol. [FULL_COMPARISON.md](FULL_COMPARISON.md) lists all 16 images side by side; [full_comparison.csv](full_comparison.csv) includes exact Margin/Gain, gold/wrong absolute LogP and correctness flags. These are reformatted completed scores, not new experiments. The existing margin_vs_full.png plots every B1-B4 final Margin against its own Native Full; the diagonal means no change.

The inherited complete-candidate scoring protocol appends one answer token. B1 blocks every valid post-image text query, including that future teacher-forced answer query. This query cannot causally influence P-1, Question or the scored first-answer prediction. B1's local all-target-query Mass average includes it; the separately reported Question/P-1 Mass and hidden metrics do not. A uses prompt-only Prefill. Do not compare A and B outputs bitwise across their different sequence lengths.

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

1. **Does task-specific spatial selection already exist?** The pooled maps are extremely similar across questions at both layers (mean cosine >=0.999992; normalized entropy about 0.94-0.96). At L1, Question has positive paired own-versus-other ROI preference in 11/12 pairs, but the mean effect is only +0.000571 enrichment units; at L0 it is -0.000091 with 4/12 positive. L1 P-1 preference is negative on average (-0.000056, 3/12 positive). This is a weak descriptive hint of emerging Question-specific alignment at L1, not evidence of strong spatial selection or proof that selection begins only at L1. All 12 matched L0 P-1 visual-QK/FP32 conditional distributions are exactly invariant. Tiny differences in conditional maps reconstructed from stored FP16 probabilities are rounding, not task selectivity. L0 Question maps can also change simply because their query words differ.
2. **Does removing all early Text->Image reading harm answers?** B1 leaves accuracy at 8/16 with no flips or ties, but its mean/median Margin Gain is -0.143555/-0.187500, with losses in 11/16 images. The worst loss is -1.289062 at qid 04468301 (Full Margin 3.507812 -> 2.218750). Thus there is measurable confidence/ranking-margin damage without a measured accuracy loss in this pilot. This is a local two-layer causal intervention, not deletion of visual tokens or proof of a population effect.
3. **Which direct image pathway has the larger effect?** Mean per-image absolute Margin change is 0.649902 for B3 (Question->Image), versus 0.307617 for B2 (P-1->Image); medians/signs alone conceal this difference. Their signed mean Gains are +0.007813 and +0.099609, respectively. B3 has stronger descriptive final-margin sensitivity but mixed benefit/harm, not a demonstrated greater necessary information contribution. Causal timing is consistent: B3 leaves L0 P-1 exactly unchanged, then changes it at L1 (mean cosine distance 0.017853); B2 leaves Question states bitwise unchanged, including the final layer. Alternative text routes remain available.
4. **Is early P-1->Question important?** B4 changes P-1 at L0/L1 (mean cosine distances 0.016781/0.057112), and final margins (mean absolute Gain 0.312256; signed mean +0.130127; worst -0.652344). The path therefore modulates representations and predictions, but its necessity for correct answers is not established: all 16 answers retain their original correctness, other text keys remain legal, and later layers still read Question. Question states are bitwise Full under B4, as required by causal ordering.
5. **What supports relatively global early fusion?** High-entropy, nearly question-invariant pooled spatial maps and substantial native visual Mass are compatible with H1's relatively global early reading. Blocking B1 removes mean local native Mass of 0.820482 at L0 and 0.727837 at L1, redistributed over legal text keys; the intervention changes hidden states and some final margins. Structural L0 P-1 invariance is an implementation check, not independent empirical proof of semantic fusion. Entropy and Hidden Drift do not measure information quality.
6. **What contradicts or limits the hypotheses?** The small but consistently positive L1 Question ROI preference argues against an absolute claim of no task specificity. B3's sizeable per-image effects, persistent final representation changes, and severe losses on particular images limit a claim that early visual paths are irrelevant. Directional effects also cancel: B1 mean Gain is -0.633789 for left versus +0.346680 for right; B3 is -0.340332 versus +0.355957. These are descriptive directional shifts in an intentionally stratified sample, not established population bias. H2's claim about not needing image and Question simultaneously remains undecided: B2 and B4 were blocked separately, never jointly, and other text routes were never removed.
7. **Can L2 and later compensate?** Yes, this remains possible because all 576 visual tokens and native masks are preserved in L2-L31. The interventions leave nonzero final P-1 hidden change norms (B1/B2/B3/B4 means 23.4056/6.7939/15.3957/7.9446), so early changes are not simply erased; such drift is not direct proof of harm or information loss. The design does not isolate compensation or distinguish it from redundant paths and margins that remain on the same side of zero. No additional layer or combination experiment was run.

Attention association, local causal intervention, Hidden change and final prediction are distinct evidence. Hidden drift is not information quality; B2 is not a Question-only visual route, B3 is not a visually empty Question, and B4 is not removal of all text. No automatic extension is authorized.

### Coverage and Interpretation Limits

Experiment B includes all 16 frozen images, four per historical left/right x correct/wrong stratum. Experiment A includes only 12 valid pairs; the four historical left/wrong images have no eligible pair and are not replaced. Its composition is therefore not balanced like B. Both questions use the same ChooseRel task family and often share much of their text; conclusions do not cover arbitrary question types. Pooled maps can hide head-specific selectivity. Pooled Question hidden differences also contain lexical differences and cannot isolate visual information quality.

Annotations, original object IDs, answers, boxes, image hashes, actual prompt token IDs and dynamic mappings are preserved in manifest.json and annotation_provenance.json. The pinned author-provided GQA subset is reliable recorded provenance, but independent recovery of the original Stanford JSON was not completed; this limit is not presented as verified official-JSON ground truth. No question, answer or ROI was synthesized to fill gaps.

This pilot is compatible with relatively diffuse early reading and different Question/P-1 roles, while showing weak L1 Question alignment and nontrivial causal effects. It neither proves H1/H2 nor justifies a broader performance claim. Stop here for research review rather than extending the conditions.

## Audit and Numerical Differences

Recorded B attention audits: 5120; local masked-layer/candidate audits: 256. CPU and two-image GPU smoke gates passed before the formal pilot. No-op valid-text logits and post-cleanup Native Full are exact.
All masks reuse the existing clone/-inf implementation. Blocked target attention is zero; causal/padding constraints and nonempty legal rows hold; non-target queries are locally exact; L2-L31 masks and all 576 visual tokens are retained. Softmax redistribution of unblocked keys is expected and recorded.
Full before/after model parameter+buffer byte hashes match; input/parameter versions and installed Transformers files are unchanged. Exception cleanup is CPU-tested. No failure is silently counted as complete. Historical Full Margin differences are recorded per image; they are never substituted for current paired Full or fitted with new tolerances.

CPU acceptance passed 129/129 tests (including the 21-test focused suite), followed by the two-image real-GPU smoke and the complete 16-image run, on GPU0 only. No NaN, Inf, OOM or failed sample was accepted. Maximum absolute current-versus-historical Full Margin difference is 0.0312500000000018; no tolerance was loosened to hide this difference. Tiny negative computed cosine distances of order 1e-17 on bitwise-identical vectors are floating-point roundoff; the corresponding change norms and explicit bitwise checks remain zero/exact.

### Publication Finalization

Scientific aggregation ran once. The original generated summary gate was written while aggregation_console.txt was still open; its final SUMMARY_COMPLETE line was printed afterward, changing that log's hash only. The unchanged original gate is retained as aggregation_validation_generated.json and the unedited generated report as machine_report.md. This reviewed report adds interpretation without changing scores, samples, plots or raw observations. summary_validation.json is finalized after log closure; publication_validation.json verifies these disclosed publication-only differences and all original scientific artifact hashes. This is not a rerun, a new condition or a silent repair of results. The aggregation runner's open-stdout hash timing limitation is recorded rather than modifying its frozen source after the experiment.

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
