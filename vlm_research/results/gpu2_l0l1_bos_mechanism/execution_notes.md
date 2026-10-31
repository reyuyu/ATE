# Execution Notes

- Repository: reyuyu/ATE; independent branch codex/gpu2-l0l1-bos-mechanism.
- Base: e495adf4fd93337a3638981bbb66c2bbac81242d.
- Frozen successful collection code: b9a85a11360b3135bc7dc88c6c522e45e271537d.
- Only physical GPU2 UUID GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 was exposed.
- The existing generation/scoring/cache engines and AttentionStep observer were reused.
  EarlyRows extends the observer only for layers0..5 and the original P-1 row.
- All3,000 original samples, prompts, 576 image positions, BOS and P-1 mappings
  were reused unchanged. Historical Full/E/D checksums are in reuse_bindings.json.
- Full/E/D complete answers and native first-token LogP were reused. No full
  reference forward per question is needed for the new LogP-loss metric.
- Only L0L1_OFF free generation is new: greedy, max_new_tokens32, EOS2,
  condition-owned Prefill/KV/prefix; o_proj output zero only at original P-1
  and subsequent Decode queries. Other Prefill queries and MLP remain normal.
- Two necessary GPU smoke cases replayed old Full answers and first logits,
  checked all-step No-op logits and initial-cache fingerprints, every layer's earlier Prefill queries,
  all MLP calls, independent HF masked generation, original-P-1 capture logits,
  actual BOS cached K/V, native restoration, clean hooks and unchanged inputs.
- The12 primary cases were selected before any outcomes with seed20261031.
  At most3 post-hoc failures are separately selected by the frozen illustrative
  rule; they are not added to the primary mechanism analysis.
- Capture uses one fresh cached Prefill per condition, never an EOS Decode step.
  Q-projection outputs are the actual module outputs; selected Q is rotated by
  the official apply_rotary_pos_emb with the actual position embeddings.
  BOS K/V come directly from the resulting DynamicCache, not re-projections.
  Only six attention rows per head and small selected vectors are retained.

## Commands

Run from the worktree's vlm_research directory, using
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.

CPU: `CUDA_VISIBLE_DEVICES='' python -m l0l1_bos.run prepare`, then
`CUDA_VISIBLE_DEVICES='' python -m l0l1_bos.run cpu`.
After committing sources: `CUDA_VISIBLE_DEVICES='' python -m l0l1_bos.run freeze`.

GPU: `python -u -m l0l1_bos.run worker` with
`CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`MPLCONFIGDIR=/data/lk/ATE-resources/matplotlib`.
Console is gpu2_console.txt; checkpoints are atomic and individually bound.

After worker exit: `CUDA_VISIBLE_DEVICES='' python -u -m l0l1_bos.analyze`.
Inference uses only GPU2; aggregation and plotting use CPU. No other GPU process
is stopped or modified, and main is never pushed.

## Pre-score Failure Audit

The first start at code40ff76a37d42fb1c41a7ba02b6de38ddad08fe46 failed before
Smoke and before any formal score: the existing TASKS object is a dictionary,
not a numerically indexed list. The command exited and released GPU2.
The fix converts its keys to a list and adds a CPU regression test.
The old CPU log, execution binding and GPU console are retained under
attempts/01_pre_smoke_task_index; the traceback is retained under failures/.
The corrected31-test CPU acceptance, new frozen binding and successful GPU
Smoke were required before formal collection. No tolerance was relaxed.
