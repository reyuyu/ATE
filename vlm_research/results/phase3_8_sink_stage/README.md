# Phase 3.8 Results Index

Completed:16/16 frozen images, C0-C4 only, physical GPU1 only.
This is a mechanism pilot, not an overall accuracy benchmark.

- [Report with all eight research answers and reviewed interpretation](report.md)
- [Per-image scores, complete candidates and absolute LogP](per_sample_scores.csv)
- [Native raw Attention masses and actual vector statistics](native_vector_statistics.csv)
- [All conditions' post-layer/final-norm hidden drifts](hidden_drift.csv)
- [Frozen16-image manifest, selection rule, Sink/ROI and resource/source hashes](manifest.json)
- [Machine-readable grouped outcomes and historical Full differences](summary.json)
- [CPU regression acceptance:121 tests, no skips](tests/validation.json)
- [Two-image smoke gate](smoke/validation.json)
- [Formal16-image gate and actual full weight-tensor hashes](run/validation.json)
- [Finalized master output hashes](summary_validation.json)
- [Transparent report/log finalization audit](artifact_finalization_audit.json)
- [Execution and authorization provenance](execution_notes.md)
- [Recursive publication audit](publication_validation.json)
- [Ready-to-submit PR description](PR_DESCRIPTION.md)

Frozen code: d0da03a848b8825941ec9f4d5df2461f87c6b0bf.
Baseline: e89c8acce6e674ecfbd2c8acad57faf71bf4d547.
Branch: exp/phase3_8-sink-stage.

Each run/sample_QUESTION_ID contains a checkpoint, numerical/audit JSON and
raw-vector/logit/Attention NPZ. Smoke files are separate and not added to the
formal outcome denominator. Model weights, image files, environments and
credentials are not committed.

To reproduce verification or compatible resumption, use a separate worktree
at the frozen code commit. The protocol intentionally rejects later
result-only commits. Do not rerun the completed scientific master summary.
No existing algorithm, Phase2-3.7 result, frozen historical Manifest, model
configuration, GPU0 worktree or third-party installed source was edited.

PR API authorization is unavailable with the installed Deploy Key alone.
After branch publication the user may create the PR at:
https://github.com/reyuyu/ATE/compare/main...exp/phase3_8-sink-stage?expand=1
This comparison URL is not evidence of an already-created PR. No main push
or automatic merge is performed. The pilot stops after publication.
