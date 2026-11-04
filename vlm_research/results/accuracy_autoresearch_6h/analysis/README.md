# Statistical Reading Rules

`09_final/` is the frozen-candidate evaluation. Its `ni_comparison_predeclared` field identifies the only noninferiority comparisons: the selected architecture versus Native on two datasets, separately under H0/H1, family size four. All four fail to support the prespecified -1pp margin.

Earlier DEV/CHECK CSVs are retained byte-for-byte because CHECK file hashes were frozen before FINAL. Some early summaries contain the legacy numerical field `ni_minus1_supported` without a preregistration flag. Those values are exploratory calculations only, **not confirmatory noninferiority claims**. They were not used in the selection rule. The authoritative selection uses worst-dataset point loss, a predeclared 0.25pp tie band, then matched-sequence latency and KV.

Other pairwise/task/DEV/CHECK intervals are exploratory. In particular, the `corrected_*` columns for arbitrary pairwise rows do not define an additional preregistered hypothesis family. A CI spanning zero does not prove equality or absence of harm.

H0 and H1 are separate generated answer bindings. `final_quality_speed.png/pdf` uses H1 quality and H1 speed consistently. `supportive_full_cohort.csv` includes DEV/CHECK and is not a new confirmation set. CSV files retain the standard `csv` writer's CRLF line endings; no frozen result was reformatted to satisfy a whitespace checker.
