# Formal Execution Notes

- Base: `2654e80a4b9448dd864ae6c726998e4931bf772a`.
- Branch: `codex/query-role-attention-boundary-240-formal`.
- Worktree: `/data/lk/ATE-worktrees/query-role-attention-boundary-240-formal`.
- Unchanged Python environment: `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
- Only physical GPU1 UUID `GPU-b08489af-0a8a-eb48-55ba-8346733af417` is visible to
  the GPU process as cuda:0. Preflight checks UUID, processes and free memory again.
- Old Partial and repair result directories are immutable; their file hashes are
  recorded in the new protocol. Manifest and input mapping copies preserve bytes.
- The task text contains a truncated/mistyped Manifest hash. The actual 64-digit
  hash from the requested base commit is preserved and recorded in protocol.json;
  no samples or input data are regenerated.

## Versioned Changes

Only the new `query_role_formal/` module and new result directory are added.
The repaired role generator/controller and all historical experiment source files
remain unchanged. Formal calls explicitly pass `logits_to_keep=1`.

The already-passed repair smoke is reused with only its output destination changed.
It does not rerun the LM-head root-cause investigation. Historical A parity uses
the original `supplement_generate` E controller, with a call-scoped model prehook
that supplies the supported `logits_to_keep=1` argument. Raw effective FP16 logits,
full token sequences and the legacy FP32 prediction traces must match exactly.
The prehook neither replaces model forward nor changes Attention, MLP or Cache.

## Stages

Set `PYTHONPATH=.:vlm_research`; CPU stages hide CUDA. Run:

1. `python -m vlm_research.query_role_formal.run cpu`
2. Commit source files and CPU acceptance, then run the `freeze` stage.
3. With only the physical GPU1 UUID exposed, run the `gpu` stage. This rechecks
   all three frozen smokes and historical A parity before collecting 960 new
   formal generations. Smoke outputs never enter formal checkpoints/statistics.
4. After the model process exits, hide CUDA and run
   `python -m vlm_research.query_role_formal.analyze`.

Formal checkpoints are immutable atomic per-condition JSON files; a question is
complete only when all four are present. Resuming requires identical protocol,
source/model/input bindings and verified raw arrays, followed by fresh smoke.
First-token arrays are stored outside Git under
`/data/lk/ATE-resources/experiments/query_role_attention_boundary_240_formal/`.

All 960 records are replayed for text, scoring, flips, signed loss, divergence,
role events and cache positions before the analyzer marks COMPLETE. Statistics
reuse the original image-cluster paired bootstrap (20,000 draws, seed 20261102),
summary functions and plotting helpers. No noninferiority or speedup is claimed.
