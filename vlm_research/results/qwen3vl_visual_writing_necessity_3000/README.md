# Qwen3-VL Visual Writing Necessity — Execution Records

The resumed formal experiment uses physical GPU0 and GPU3, with one
model and 1,500 questions per GPU. All questions of an image cluster stay on the
same GPU. The target is 3,000 questions, 2,409 image clusters, 38 conditions and
114,000 independently cached complete greedy answers.

- Frozen generation code: `04f520914ea37d3808fe9360d6ffe62ace5ef373`.
- Protocol: `protocol.json`; original native Qwen inputs: `input_bindings.json`.
- CPU regression: 28 tests passed. Statistics/figure regression: 4 tests passed.
- Both GPUs passed all three frozen real Smoke cases, including strict native
  no-op equality and the Drop-L35 structural negative control.
- Live progress: `shard0/progress.json` and `shard1/progress.json`.
- Checkpoints are immutable and local; verified raw archives are produced after
  complete collection, avoiding 114,000 loose files in Git.
- The internal image-disjoint subset has 2,640 questions; it is not a new
  independent external test cohort.

Until `completion.json` reports COMPLETE, any available data are partial and
must not be interpreted as complete 3,000-question results. That file is written
only after all 114,000 answers have been independently
decoded, officially rescored, audited, summarized and archived, and all paper
figures generated. The supervisor pushes only after this completion gate passes.

The earlier `resource_preflight.json` and `preflight_report_zh.md` describe the
previous GPU-busy attempt; they are historical records, not the current status.
