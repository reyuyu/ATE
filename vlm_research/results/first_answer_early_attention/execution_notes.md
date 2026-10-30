# Execution Notes

- Base: `c25001e135408ad72d9f797da03265464b01f7b1`.
- Frozen collector: `7b9b033e149ea03c4583fe95734761d607519d42`.
- Branch: `codex/first-answer-early-attention`; independent worktree.
- No historical source or result is modified. GPU0 is not used.
- Exact original 3,000 samples and Figure3 image-to-GPU1/2/3 assignment are reused.
- Figure3 Full free generations, tokens and scores are reused, not regenerated at scale.
- Each question additionally receives one native cached Prefill forward to obtain its current-hardware first-token reference distribution. Its Top1 must match the reused Full first token.
- The existing P-1-only hook is used for every condition. C/D/E compose it with the existing Decode-only hook, whose active time is disjoint. The historical all-Prefill `Combined` mode is deliberately not used.
- Every generation owns its Prefill, actual generated prefix and KV Cache. All conditions retain the original 32-new-token safety cap; truncations are scored and reported, not excluded.
- First-token LogP Loss uses the native Full Top1 token, FP32 log-softmax, natural log units; positive means lower probability, negative means higher probability. It is not an accuracy metric or a gold-token score.
- CPU: 38 checks. GPU: two fixed real inputs per worker; all five conditions and native HF generation; no-op all-step logits exact; all other Prefill queries exact in all32 layers; all32 MLPs execute; input/parameter versions and hooks audited. Cross-GPU smoke results must be exact before opening the formal gate.
- Primary statistics: 200,000 whole-image-cluster paired bootstrap replicates, seed20261031, question-weighted accuracy; ordinary95% and five-comparison Bonferroni-corrected95% intervals. Noninferiority requires the corrected lower bound strictly above -1pp.
- Six separately prespecified auxiliary contrasts: B-A, D-C, C-A, D-B, E-C, E-D; family6 correction. Taskwise intervals are descriptive within-task family5, not a global30-test guarantee.
- Initial startup failed before any scoring because a task dictionary was sliced as a list. All three processes exited; the one-line fix and a CPU regression check were committed before restarting. Failed manifests, console logs and tracebacks are retained in `attempt1_startup_failure/` and `failures/`.
- No training, new data, additional windows or claims of actual computational speedup.

## Commands

Run from this worktree's `vlm_research` directory with the pinned `ate-vlm-research` Python:

```bash
python -m first_answer_attention.run cpu
# Commit the collector sources before freezing.
python -m first_answer_attention.run freeze
# Run rank0/1/2 in separate processes with their frozen GPU1/2/3 UUIDs.
python -u -m first_answer_attention.run worker --rank 0
python -u -m first_answer_attention.run worker --rank 1
python -u -m first_answer_attention.run worker --rank 2
# Only after all three smoke.json files exist:
python -m first_answer_attention.run gate
# Only after all three completion.json files validate:
python -m first_answer_attention.analyze
```

Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false` and one `CUDA_VISIBLE_DEVICES` UUID per worker. Frozen JSON and checkpoint bindings reject incompatible resume. Do not edit collector sources while workers execute.
