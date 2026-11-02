# Execution and Validation Notes

- Requested historical base: `bd171312fd0489e8bc027744b7a976367da11668`.
- Isolated worktree: `/data/lk/ATE-worktrees/query-role-attention-boundary-240`.
- Branch: `codex/query-role-attention-boundary-240`. No main edits or merges.
- Only physical GPU1, UUID `GPU-b08489af-0a8a-eb48-55ba-8346733af417`, is exposed to the GPU process. Its logical device is `cuda:0`.
- Existing environment: `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
- Read all five requested historical reports. Their output-ablation and source-hard-mask evidence are not treated as interchangeable.
- Samples and processor-derived token IDs were frozen before any GPU prediction; 240 unique image SHA256 values, 40 questions per task. Three smoke samples use shortest/longest/median prompt lengths, not correctness.
- All model files were SHA256-checked against the original frozen LLaVA manifest before execution. The same files are rehashed at final CPU validation.

## Preformal Validation Attempt

The first implementation commit was `71c42d75e7`. Its GPU process stopped during the first smoke sample, before any formal checkpoint existed.

The validation harness attempted to compare HF `generate` logits of shape `[1, vocab]` with the custom prefill's `[prompt_length, vocab]` matrix. Installed Transformers 4.49.0 automatically uses `logits_to_keep=1` in native generation. The full generated token sequences had already matched before the shape assertion failed.

Only the comparison alignment was corrected: native HF and custom generation compare their actual last-query prediction vectors exactly. Full-versus-No-op and Full replay still compare every returned valid prefill/decode row exactly. No numeric tolerance, model, precision, prompt, sampling rule, layer set, or experimental condition was changed.

The failure, original environment, log, protocol, and CPU acceptance are preserved in `failures/` and `preformal_attempt_71c42d75e7/`. A CPU regression for the shape distinction was added; all 11 tests passed. Sources were committed and the protocol was refrozen before restarting GPU smoke. The formal collection code commit is `3744af84e8` (full SHA in protocol.json).

## Attempted and Planned Stages

Commands use the foreground, without background polling, timers, additional GPU experiments, or subprocess termination. Stages 1-3 completed; stage 4 stopped during smoke; its formal matrix and stage 5 were NOT executed:

1. `python -m vlm_research.query_role_attention.run prepare` with CUDA hidden.
2. `python -m vlm_research.query_role_attention.run cpu` with CUDA hidden.
3. Commit code and immutable samples; `python -m vlm_research.query_role_attention.run freeze` with CUDA hidden.
4. `python -m vlm_research.query_role_attention.run gpu` with only the fixed GPU UUID exposed. One model replica runs smoke, then the four-condition matrix serially.
5. After the GPU process exits, `python -m vlm_research.query_role_attention.analyze` with CUDA hidden. Official GQA conversion/scoring, checkpoints, raw vectors, paired bootstrap, figures, model hashes and GPU release are checked here.

All active collection sources are immutable after freeze; scalar outputs, input maps, sources and checkpoints are committed. First-token full-vocabulary arrays are stored only under `/data/lk/ATE-resources/experiments/query_role_attention_boundary_240/`, with paths and SHA256 values in the checkpoints and validation JSON. They are not uploaded to GitHub.

Correction for the actual partial outcome: no formal checkpoints or raw-logit arrays were produced. The preceding sentence describes the implementation's intended storage, not completed collection.

## Mandatory Stop

The second smoke attempt used the refrozen `3744af84e8` implementation. After aligning prediction-query shapes, `torch.equal` still failed for native-HF versus custom-Full effective logits. Their complete token sequences had matched. The failure is preserved as `failures/gpu_1793626706075986460.json` and `gpu_execution.txt`.

This is before any I/R/A ablation or the Full/No-op checks could be completed. The process exited; no tolerance or precision was changed and no further GPU attempts were made. Native `logits_to_keep=1` versus all-position LM-head computation is a possible FP16 arithmetic-shape explanation, not an established diagnosis. The actual differing vectors/max difference were not saved by this assertion and are not invented here.

Final status is PARTIAL / validation gate failed / 0 of 960 formal generations. The separate CPU partial-delivery script records this status, verifies immutable artifacts and model files, and audits GPU release. It does not fabricate scores, intervals or figures.

The scope is only Full / I-Bypass / R-Bypass / A-Bypass at L2-L5 on this 240-question exploratory cohort. No physical skipping, training, source mask, extra window, model, or follow-up experiment is implemented.
