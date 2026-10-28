# Phase 3.6 Results Index

Research status: **completed, 240/240 images**. Publication status: experiment
branch can be pushed; PR creation requires GitHub account API authorization,
which is not provided by the repository Deploy Key. No merge is authorized.

- [Reviewed interpretation and caveats](review.md)
- [Unmodified machine-generated report](report.md)
- [Per-image, per-condition scores: 2,160 rows](per_sample.csv)
- [Frozen cohorts, statistics, paired bootstrap, tails and correlations](aggregate.json)
- [Master completeness and artifact hashes](summary_validation.json)
- [Frozen protocol, source hashes, model hashes and original data bindings](protocol.json)
- [CPU regression acceptance: 121 tests, no skips](acceptance/validation.json)
- [Real-model two-image smoke acceptance](smoke/validation.json)
- [Four-worker independent completion audit](worker_completion_audit.json)
- [Worker allocation and GPU UUIDs](coordination.json)
- [Ready-to-submit PR description](PR_DESCRIPTION.md)
- [Final recursive publication integrity audit](publication_validation.json)

Each `shards/shard_N` directory contains its own manifest, console log,
environment record, 60 checkpoints and per-image reference NPZ / compressed
head-level audit files. Smoke also retains raw prediction logits.

Frozen implementation commit: `0d323e1bb3eb6f91443854e90242150b04e9eaf1`.
Original baseline: `ed666b68eef6380a8bd57852636c18bef1673b02`.
Protocol binding: `e3e32e45f36f1842f8a5904a73bca2d1f89295aeced959fafdc8d34f2311588a`.

For verification or resumption, use an isolated worktree at the **frozen
implementation commit**, not a later result-only commit or a merged main:
the protocol deliberately rejects different code commits and source files.
Do not rerun the completed master aggregation; verify its existing hashes.
Model weights, image files, environments and credentials are not committed.
Original history, manifests, model configuration and shared algorithms are
unchanged. Four GPU processes exited normally and released their allocations.
