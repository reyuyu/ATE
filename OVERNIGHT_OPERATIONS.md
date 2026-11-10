# Qwen3-VL Consecutive Quality and Re-access Experiments

Branch: `codex/qwen3vl-svh-quality-then-svhr-search-2x3090`.

Worktree: `/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090`.

Python: `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`.

The current supervisor runs in tmux session `ate-qwen-quality-svhr-20261111`.
It has a single-owner flock. Do not start another copy or delete lock files.

## Active Sequence

1. Frozen original-Eager Native/SVH structure smoke on both UUIDs.
2. 35,346 paired questions in independent WAL SQLite checkpoints.
3. CPU rescore, complete-group coverage, image-cluster bootstrap and local commit.
4. Pre-frozen image-disjoint Discovery/Validation/Held-out cohorts.
5. Fourteen compatible functional candidates; Early8 and Middle24 are inapplicable.
6. Top-three validation; final choices frozen before held-out outcomes.
7. Independent selective-KV physical port and held-out tests, only after its smoke.
8. Actual cap32 timing including extra visual-KV construction, with matching-sequence subset.
9. If the physical choice passes structure and free disk permits, its frozen full run.
10. Final CPU report, local aggregate-only commit and GPU release audit.

Strict numerical failures remain failures. Quality mode is
`RELAXED_NUMERICAL_QUALITY_ONLY`; it is not author-official reproduction,
full-59 numerical validation, or a proof of acceleration. Original vision Eager,
BF16, pixels, prompts, token budgets and scorers are unchanged. OOM requests retain
their planned denominators; MME pairs and MMBench Circular groups stay complete.

## Inspect

```bash
tmux capture-pane -pt ate-qwen-quality-svhr-20261111
tail -n 10 vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench/worker0.log
tail -n 10 vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench/worker1.log
/data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_overnight.report --partial
```

The last command performs CPU-only partial rescore. Partial image-ordered results
are not an unbiased full-benchmark estimate. After the first stage is terminal,
it switches to a read-only status view and does not overwrite frozen results.

## Stop and Resume

Create `vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench/STOP_USER` to request
a checkpointed stop at the next request boundary. This does not signal external
GPU jobs. A structure/resource failure writes a separate STOP record and does not
auto-retry indefinitely. Do not remove a failure marker without investigation.

The supervisor command, when no existing owner remains, is:

```bash
/data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u -m vlm_research.qwen3vl_overnight.pipeline
```

Resume validates protocol/source/input/model bindings and never overwrites a
completed prediction. Existing stage selections must match exactly. Leave the
frozen execution modules unchanged while workers are active.

## Storage and Privacy

The two result roots are under `vlm_research/results/`:
`qwen3vl_ate_hf_native_svh_sixbench` and `qwen3vl_svhr_reaccess_layer_search`.

`overnight_report_zh.md` starts with the two requested tables. Raw predictions,
gold, tokens and full input bindings remain local in compressed SQLite payloads
with SHA256. Only explicit aggregate artifacts are auto-committed locally. There
is no push, PR, main update, model/image copy, dataset expansion or environment
installation. The disk safety floor is 2 GiB; reaching it stops writes rather
than deleting historical data.
