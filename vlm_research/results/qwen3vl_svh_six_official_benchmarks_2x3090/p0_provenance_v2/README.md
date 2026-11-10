# P0 Provenance V2

This directory is a new static audit. It does not replace the parent P0/P1 blockers.
Start with `report_zh.md`, `six_task_protocol_matrix.csv`, and `evidence_index.md`.

`STATIC_AUDIT_COMPLETE` never means the author protocol, Physical numerical gate,
or GPU experiment passed. Plan B is explicitly non-author-official and not executed.

CPU-only reproduction from the existing archives (no source downloads required):

```bash
python3 /data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090/p0_provenance_v2/extract_paper.py
python3 /data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090/p0_provenance_v2/check_datasets.py
python3 /data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090/p0_provenance_v2/build_matrix.py
python3 /data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090/p0_provenance_v2/test_static_provenance.py
```

The input inventories require the existing read-only `/data` resources.
`collect_sources.py` is an optional small-source retriever, not an evaluation runner.
HTTP403/404 and PDF timeout limitations are recorded; missing sources remain unresolved.
No weights, raw images, environment packages, or performance results are added.
Archived upstream source is included only for provenance; its imports are not executed.
