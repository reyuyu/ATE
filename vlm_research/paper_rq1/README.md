# Paper RQ1: Layer and Generation-Time Attention Necessity

This is an independent, inference-only experiment. Historical experiments and
main are not modified. All layer numbers are zero-based. Bypass means zeroing
the current query's Attention `o_proj` output, not physically skipping compute.
MLP, other queries, weights and positions remain native.

## Frozen Design

The executable preregistration is `protocol.py`. Seed: `20261029`.
Discovery targets 2,000 questions; independent Confirmation targets 5,700.
Official GQA Balanced validation replaces neither annotations nor selection
with the old `correct_question` subset. COCO uses official 2014 validation,
not the Karpathy test split; results are subset Pilot evidence, not published
leaderboard scores. TextVQA is image-only without external OCR tokens.

Before GPU work, `refine.py` resolves published VG/COCO/Flickr/OpenImages aliases,
then byte/RGB duplicates. The original selection draft is retained; exclusions
and deterministic input-only replacements are recorded in the final selection
plan. ScienceQA has no publisher cross-source photo IDs, so its cross-source
coverage is byte/RGB only. This does not guarantee perceptual deduplication.
Discovery/Confirmation are separated by image, and six POPE questions sharing
one image are always kept together. All image/source SHA256, prompts, question
positions and shards are frozen before model scoring.

Discovery probes reached native steps 0, 1, 4, 8, 16, including an EOS prediction
but never a post-EOS step. Every window deep-clones the same native history;
its changed cache is discarded. All native step probability distributions are
saved in external NPZ artifacts, with hashes in committed per-question results.
The frozen prompt and native token IDs preserve replayable native prefix state;
large per-layer KV tensors are transient, never mixed between interventions.

Window selection averages JS within dataset/reached-step cells and then equally
weights available cells. Fixed starts 0, 9, 20, 29 are supplemented with the
two lowest remaining and two highest remaining starts; ties use smaller start.
Confirmation is not read or generated before this selection is frozen.
Confirmation has one Full and 24 independent greedy generations per question.
Each condition owns its own causally updated KV history.

Short answers use the pinned official/publisher protocols. Caption results use
real PTBTokenizer, CIDEr and SPICE; missing Java/scorer artifacts fail closed.
Image-cluster paired bootstrap uses 50,000 replicates. Marginal intervals and
familywise Bonferroni bounds are both reported (144 primary, 48 SPICE secondary
comparisons). Noninferiority margins: 0.01 short score, 0.02 raw CIDEr, 0.01 SPICE.
An exact zero-negative-event resolution guard prevents degenerate bootstrap
intervals from automatically proving noninferiority with too few images.
Caption bootstrap conditions on complete-cohort CIDEr IDF rather than refitting
IDF in every bootstrap sample. These choices, including the guard, were fixed
before any GPU experimental results; the initial non-GPU-ready selection draft
predates these final protocol clarifications and is retained for provenance.

## Execution

Use the pinned environment at
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
Run modules from `vlm_research`:

```bash
python -m paper_rq1.inventory
python -m paper_rq1.acquire
PYTHONPATH=/data/lk/ATE-resources/benchmarks/phase4_0/tools python -m paper_rq1.data plan
python -m paper_rq1.data fetch
PYTHONPATH=/data/lk/ATE-resources/benchmarks/phase4_0/tools python -m paper_rq1.refine
python -m paper_rq1.data freeze
python -m paper_rq1.scorers
python -m paper_rq1.tests
# Commit the final source and manifests before execution freeze.
python -m paper_rq1.run freeze
python -u -m paper_rq1.launch
```

The coordinator checks all four physical GPU UUIDs without interrupting existing
tasks. Identical frozen sources run on GPU0-3 with one model copy each, independent
shards/checkpoints/output paths. Four smoke passes gate Discovery; complete
Discovery gates frozen windows; complete Confirmation gates final analysis.
No failed gate is converted into a score. Checkpoints validate execution/sample
bindings and source/model/environment consistency before resume.

Results: `vlm_research/results/paper_rq1_attention_time/`.
External images, full-vocabulary NPZ files and scorer binaries:
`/data/lk/ATE-resources/paper_rq1/`. They are not committed as Git blobs.
Only real complete results may populate the final paper figures. A failed or
incomplete study must be labelled partial; no fabricated values or equivalence
claims. Figure 4 includes conditional descriptive associations, not causal
explanations. This experiment cannot establish real inference acceleration or
cross-model generality.
