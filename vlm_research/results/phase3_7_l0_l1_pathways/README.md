# Phase 3.7 Result Index

Completed descriptive pilot: 16 frozen B images, five conditions (80 rows), 12 real same-image A pairs (48 layer/query rows). Only GPU0 and L0/L1 were used. No further research run is authorized by this result.

- Start with [report.md](report.md): measured outcomes, seven research answers and limitations.
- [manifest.json](manifest.json): frozen selection, actual token/ROI mappings, sources, model/environment hashes, image SHA256 and configuration binding.
- [per_sample_b.csv](per_sample_b.csv), [per_pair_a.csv](per_pair_a.csv), [summary.json](summary.json): scores, attention/hidden measurements and descriptive statistics.
- [FULL_COMPARISON.md](FULL_COMPARISON.md), [full_comparison.csv](full_comparison.csv): each image's Native Full and B1-B4 scores side by side, plus overall final Margin and absolute gold LogP.
- `acceptance/`: 129-test CPU acceptance, pip consistency and pinned environment.
- `smoke/` and `pilot/`: completed checkpoints, compressed per-candidate audits, native/intervention observations, full model byte-hash checks and environment records. Raw images/model weights are not committed.
- `heatmaps/` and the five top-level PNGs: all 17 requested figures, retaining adverse samples.
- [annotation_provenance.json](annotation_provenance.json): pinned author CSV provenance and explicit original-JSON verification limitation.
- [aggregation_validation_generated.json](aggregation_validation_generated.json), [machine_report.md](machine_report.md): unmodified original aggregation gate/report.
- [summary_validation.json](summary_validation.json), [publication_validation.json](publication_validation.json): finalized artifact gates and disclosed log/report finalization differences.
- [publication_review.py](publication_review.py): CPU-only publication integrity verification; does not run or reaggregate experiments.
- [PR_DESCRIPTION.md](PR_DESCRIPTION.md): review-ready PR body. Deploy-key push does not confer API permission to create a PR.

Baseline: `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`.
Actual executed code: `174014dac2bacb5712d2b7d6bac318ec3e77b5b2`.
Branch: `exp/phase3_7-l0l1-pathways`.

Run reproduction at the executed code commit in a fresh worktree/output directory, using the frozen manifest/resources and the instructions in `phase3_7/README.md`. The runner intentionally rejects a later result-only HEAD or an already-aggregated output directory. For read-only publication verification from the result commit, run from `vlm_research`:

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python results/phase3_7_l0_l1_pathways/publication_review.py
```
