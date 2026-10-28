# Paired Full / B4 / C2 Evaluation

This is the evaluation explicitly authorized after Phase 4.0 preparation.
All eight frozen manifests (six units; POPE has three strategies) are reused:
2,620 questions, without resampling or prediction-based filtering.

- B4 reuses the original `PathwayMask` in the pinned Phase 3.7 worktree.
- C2 reuses the original `zero_query` in the pinned Phase 3.8b worktree.
- Original modules and legacy B4 dependencies are SHA256 checked.
- Only the original prompt P-1 is intervened upon. Full candidate sequences
  are teacher-forced; later candidate-prediction queries are not newly edited.
- Generation intervenes on the first full prefill, preserves the real native
  cache, then removes all hooks before subsequent native decode calls.
- The B4 cache adapter temporarily presents cache-free kwargs only to the
  legacy validation/mask builder, then restores the actual empty cache and
  flags before attention executes. Mask/position/shape checks are unchanged.
- The cache adapter, original-algorithm equivalence, exact no-op logits,
  generation and exception cleanup have dedicated real GPU smoke gates.
- Every sample saves Full/B4/C2 atomically together and compares all causal
  prefix hidden hashes across all 32 layers. Resume rejects changed bindings.
- Full-vocabulary raw logits are saved locally as NPZ. The final aggregator
  independently recomputes all candidate and generated-token LogP from them.
- Official-style generated scores and diagnostic candidate scores are separate.
- Statistical intervals resample actual image SHA256 clusters, not heads or
  independent questions when multiple questions share an image.

Output: `vlm_research/results/cross_dataset_b4_c2/`.
The optional official GQA Balanced entry is blocked and not part of this run.

`launch.sh` is the durable service entrypoint. It checks CPU and GPU gates,
runs four read-only workers at one frozen source version on distinct physical
GPU UUIDs, waits for all workers, validates complete coverage, aggregates and
pushes only this experiment branch. Never pushes main or merges a PR.
Failure leaves partial checkpoints and does not publish a complete report.
Raw logits stay local; JSON/CSV audits, console logs, figures and report are
committed. Do not edit Python worker source during an active run.

To inspect: `systemctl --user status ate-cross-dataset-b4-c2.service` and
`journalctl --user -u ate-cross-dataset-b4-c2.service`.
Per-worker `progress.json`, `validation.json`, `failure_*.json` and
`console.txt` are the authoritative state, not this static README.
