# Frozen C1 / C2 Cross-Dataset Comparison

Reuse the exact2620questions and four physical GPU/shard assignments in
`codex/cross-dataset-b4-c2@ad73b00947b40084326c0cde75298ad8afbcb1f0`.
Do not modify any historical source/result or main. This new package subclasses
the existing prefill lifecycle, adding only original Phase3.8 `zero_query` to
the L2/L3/L4 Attention `o_proj` outputs at fixed original prompt P-1.
All other tokens/weights/residuals stay unchanged. Remove every intervention
hook at the end of the first prefill; native cached decode is never retargeted.

Stages: `python -m cross_dataset_c1_c2.run --stage freeze|smoke|decide|run`.
`--shard 0..3` binds the same physical3090as the historical worker; expose that
GPU UUID only. Every launch checks it is empty and never kills any other task.
Freeze CPU-tested code before any GPU run. Smoke on each shard selects the first
frozen question per unit plus first local multi-token ChooseAttr and local
maximum-choice ScienceQA, input-based only, at most10cases/shard.

Smoke checks original C1 bitwise equivalence, no-op full logits, empty-cache
prefill consistency, cached generation no-op, local non-target exactness, normal
and exception hook cleanup. Recompute Full/C2 on all frozen smoke cases and
compare raw full-vocabulary logits, generated tokens, candidate LogP, scores,
prefix hashes, software versions, GPU UUID and entire loaded model hashes.
Global decision reuses historical Full/C2 only if ALL four gates are exact.
Otherwise rerun all2620samples with same-device Full/C1/C2 pairing, never mix.

Formal C1 checks causal earlier prefix hashes against corresponding Full at
every layer and every candidate/prefill. Preserve complete candidate sequences
and the Phase4 actual Question/P-1 mappings and original generation settings.
Atomic per-sample checkpoints reject changed code/config/NPZ/historical hashes.
NPZ raw logits stay on disk; push JSON/CSV audit/scores/logs/report only.

Aggregation requires all four shards and before/after weight audits, verifies
saved C1 logits independently, joins only frozen baseline IDs, and reports
generated scores separately from candidate diagnostics. Fixed20,000replicate
image-SHA cluster paired bootstrap gives95%CIs for C1/Full, C2/Full and C1/C2.
Direct C1/C2 also receives Bonferroni simultaneous familywise95%CIs over8units.
Mean absolute generated score change describes perturbation size, not whether
the path is useful. No post-hoc tuning or equivalence claim from non-significance.

Output: `vlm_research/results/cross_dataset_c1_c2/`.
No training, layer scans, extra conditions, model/cache algorithm changes,
or extra datasets. Push the independent branch, never merge main.
