# Exploratory C3 quality and speed experiment

All commands run from the new worktree root, on the unchanged local Qwen environment.

1. `python -m vlm_research.qwen_quality_speed.prepare` (CPU only; run once before predictions).
2. `python -m vlm_research.qwen_quality_speed.tests`.
3. Commit source/protocol; freeze imported source files and the existing bridge binary in `source_manifest.json`.
4. With a freshly verified idle GPU UUID, run `CUDA_VISIBLE_DEVICES=<uuid> python -m vlm_research.qwen_quality_speed.run <rank> smoke` for each rank.
5. Both rank Smoke completion files must say PASSED before either quality worker can run. Replace `smoke` with `quality`, then `benchmark`, using the same frozen ranks and UUIDs. A failure is saved and raised; no fallback precision, cache, or kernel search exists.
6. On the existing plotting environment run `python -m vlm_research.qwen_quality_speed.analyze` and `python -m vlm_research.qwen_quality_speed.verify_delivery`.

Stage checkpoints are atomic, keyed by question/condition/repeat, and tied to input, image, model, protocol, source, rank and UUID. Existing records are verified and reused exactly. A changed source/protocol cannot silently resume an old request. The field `source_manifest_sha256` hashes canonical JSON of the source manifest (including its frozen Git commit); protocol/sample/model-manifest fields hash the actual file bytes. CSV indices and the final artifact index hash actual file bytes.

The Original path is the inherited, unchanged `qwen3vl_c3v2_physical.engine.Execution C`. The Repaired path copies the existing three zero-key repairs and fixed-Lt policy with identical tensor operations, omitting read-only probes when audit is disabled. This is validated against the original debug repair implementation and against audit-enabled execution in the 20-question Smoke. No repair candidates are searched.

The timed common loop includes input transfer, request setup, actual vision/language forwards, token selection, EOS handling and context/plan cleanup. Image loading/processing, model loading, text decoding and recording are outside timing. Fixed 16 means one Prefill prediction and 15 incremental Decode calls, independent of EOS. Every timed token sequence must match its finite-checked untimed reference. Memory reports PyTorch allocated peak plus live/capacity KV; reserved allocator memory is history dependent.
