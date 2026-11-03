# GQA3000 Visual Writing: Keep L0-L15 vs Keep L0-L19

## Frozen Experiment

- Base: `afddd2a9972e6decddbb535f4b524225e52eba0e`.
- GPU runner: `90be4a467c246b1a62adf937d35544812804bf60`.
- Exactly 6,000 new complete generations: 3,000 questions x two conditions.
- Historical All-Writer: 2,611/3,000 correct; never substitute native Full.
- P-1 and all real Decode queries cannot read original Image Keys at any layer.
- Only prefill R-to-Image permissions vary: 16 vs 20 allowed layers.
- GPU2 and GPU3; original image clusters and historical GPU ownership preserved.

## Statistics

The primary set has 2,640 questions / 2,169 image clusters, excluding image SHA256
overlap with the historical 240-image exploration. All 3,000 questions / 2,409
clusters are supportive. The primary set is not claimed to be independent of all
earlier research on this cohort.

The three comparisons and quality rules are frozen in `protocol.json`.
There are 20,000 whole-image paired bootstrap replicates, retaining question
weighting. For the two candidates against All-Writer, a Bonferroni familywise
two-sided 95% interval uses per-comparison 97.5% intervals (1.25/98.75 percentiles).
Only a corrected lower bound strictly greater than -1 pp passes the primary
quality rule. The -2 pp rule is supplementary. Keep20 minus Keep16 has an
ordinary two-sided 95% interval as a predeclared secondary comparison.

## Artifacts

- `report_zh.md`: actual results and the predeclared recommendation.
- `accuracy_summary.csv`, `paired_comparisons.csv`, `task_accuracy.csv`.
- `per_question.csv`: 9,000 rows, clearly distinguishing 3,000 historical baseline
  answers from 6,000 new answers, with generated Token IDs and EOS diagnostics.
- `negative_cases.csv`, `error_transitions.json`: all harms and corrections.
- `quality_comparison.pdf/png`, `task_comparison.pdf/png`.
- `historical_all_writer.json`, `baseline_reuse_audit.json`: original records
  and per-record archive/checkpoint hash provenance.
- `raw_checkpoints_shard0.tar.gz`, `raw_checkpoints_shard1.tar.gz`: complete new
  checkpoints. Loose copies remain local for resume but are not duplicated in Git.
- `checkpoint_hashes.json`, `raw_archive_manifest.json`: raw evidence hashes.
- `protocol.json`, exact `sample_manifest.json`, `input_bindings.json`,
  `analysis_sets.json`, `shards.json`: frozen model/data/condition/statistical bindings.
- `cpu_validation.json`, `analysis_tests_validation.json`, `validation.json`,
  per-shard smoke/environment/progress/completion records and closed worker logs.
- `completion.json`, `gpu_release_audit.json`: final completeness and resource audit.

Softmax hard masks do not physically skip attention computation or prune KV.
The allowed writer-layer count is not a measured speedup.
