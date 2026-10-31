# Native First-Answer Attention Mass

This directory contains a Full-only observation experiment on the exact Figure 1
cohort: six GQA tasks, 500 questions each, 3,000 questions and 2,409 distinct image
SHA256 clusters. The original cohort blob is bound in `execution_manifest.json`.
No ablation, training, decoding, or answer teacher forcing is performed here.

## Measurement

One native FP16/eager Full forward per question captures all 32 layers. Read-only
hooks observe the last original prompt Query (P-1), which predicts the first
answer Token. They return `None` and never replace intermediate model outputs.
The two-question GPU smoke compares every effective output logit exactly and
checks the captured reductions against the native Attention matrices.

Each head's post-softmax distribution is partitioned into four mutually
exclusive, exhaustive groups: Image, BOS, real Question, and Prompt Template.
Prompt is the sum of real Question and Prompt Template. Answer-format instructions
belong to Prompt Template, not Question. Token offsets are verified against the
original rendered input; image expansion and P-1 are not hard-coded indices.
All sums and head averages use FP64 CPU accumulation after lossless promotion
from native FP16 weights. We do **not** renormalize native row sums. The existing
native FP16 row-sum bound, 0.002, is unchanged.

`layer_summary.csv` includes the across-question sample standard deviation
(ddof=1). The confidence bands are ordinary pointwise 95% percentile intervals
from 10,000 whole-image cluster bootstrap draws, seed 20261031. Means are
question-weighted; heads and layers are not independent samples. These are not
simultaneous confidence bands across all layers, sources, or tasks.

## Figures and Captions

**Figure X. Layer-wise Attention Mass at the First-Answer Query.** Native Full
P-1 Attention, averaged over heads and the 3,000 frozen GQA questions. Image,
Prompt (real question plus template), and BOS are exhaustive disjoint sources.
Shading shows image-cluster-bootstrap pointwise 95% intervals. Gray bands mark
L0-L1, L2-L5, and L14-L16. Attention allocation is descriptive evidence, not a
test of whether a layer can safely be skipped.

**Figure Y. Representative Visual Attention Maps Across Depth.** Four examples
chosen by image/question clarity before GPU inference from a pre-frozen,
hash-selected pool of two examples per task. Panels show native P-1 head-averaged
raw patch probabilities at L0, L3, L15, and L31. The scale is shared across layers
within each example, not across different examples. The 24x24 patch grid is
mapped back through the actual processor resize and center crop. Outside-crop
pixels are unobserved, not zero-Attention observations. A dotted boundary marks
the crop. Full answer strings are reused from the matched native Figure 3 Full
generation checkpoints, not newly generated or teacher-forced here.

The `_paper` version is a compact 7.2-inch-wide rendering of the identical
four cases, probability maps and shared scales. It changes only layout, line
wrapping, and the abbreviated label `M = Image mass`.

`supplement_cases_conditional` uses visual-internal probabilities
`patch_probability / image_mass` instead. It isolates spatial allocation from
the amount of Attention paid to Image. Its four layers still share one scale
within each case. It must not be presented as raw Attention Mass.

## Files

- `per_question_layer.csv`: 96,000 source-Mass observations.
- `layer_summary.csv`: overall and six-task means, SD, and pointwise intervals.
- `case_pool.json`, `showcase_selection.json`: pre-inference case selection.
- `case_patch_attention.csv`: raw and visual-conditional 24x24 probabilities at
  the four display layers for all 12 frozen candidates.
- `input_partitions.json`: per-question real Question positions and sample hashes.
- `checkpoints/`: immutable, individually bound results; safe resume support.
- `aligned_layer_sensitivity.csv`, `mass_sensitivity_correlations.csv`: exact
  joins to Figure 1's original first-Token ablation results. Correlations compare
  the 32 layer means descriptively; layers are not independent replicates.
- `cpu_validation.json`, `gpu_smoke.json`, `completion.json`,
  `analysis_validation.json`: validation, coverage, environment, and hashes.
- `gpu1_console.txt`: execution log; only physical GPU1 is used.
- `postprocess_compatibility.json`: narrowly scoped serialization regression
  checks. The frozen writer hashed integer layer-map keys, which JSON converts
  to strings. `finalize` restores these key types only when checking the original
  digest. The 3,000 checkpoints, numeric values and frozen collection code remain
  unchanged; altered values still fail their original digest checks.

## Reproduction

Run from `vlm_research` with the pinned local research Python environment.
The prepare/freeze stages deliberately reject inconsistent existing manifests;
do not overwrite frozen inputs when resuming a run.

```bash
python -m first_answer_mass.run prepare
# Review the outcome-blind candidate contact sheet and freeze four showcase IDs.
python -m first_answer_mass.run cpu
# Commit source, test records, partitions, and pre-inference selections.
python -m first_answer_mass.run freeze
CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false python -u -m first_answer_mass.run worker
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m first_answer_mass.finalize
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m first_answer_mass.paper_render
```

The worker refuses to load a model on an occupied GPU1. A completed run exits
without allocating a GPU. Partial results are not a full-cohort conclusion.
