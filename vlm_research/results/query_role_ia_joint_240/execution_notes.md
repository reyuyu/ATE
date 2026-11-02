# Experiment four execution notes

Base: `8d12f8c05f2b246d029ec4d9ffe45d1f857fa947`.
Branch: `codex/query-role-ia-joint-240`.
Worktree: `/data/lk/ATE-worktrees/query-role-ia-joint-240`.

Only the new `query_role_ia_joint/` module and `results/query_role_ia_joint_240/`
are added. Historical source, records, weights and raw arrays remain immutable.
The new engine is the historical role engine with only the joint condition's
position union added. The existing strict generator and hook implementation are
retained in a separate module; historical modules are never monkeypatched.

Only I+A produces new formal checkpoints (240 total). Full, I and A reuse 720
fixed checkpoints from the base commit after raw-array SHA256, input, token/text,
score and delivery-manifest validation. Three frozen smokes additionally compare
the new path to the unchanged historical generator at all steps and to the saved
first logits/tokens, always with logits_to_keep=1 and strict torch.equal.

The historical repair Recorder/compare functions are reused with their output
directory redirected to this experiment only. No tolerance, model, mask, QKV,
MLP or residual changes are introduced. Joint prefill targets are exactly the
sorted union of original 576 I positions and original P-1; decode targets are
each actual single query, including the EOS prediction. R is not directly edited;
downstream R states are not asserted equal to Full.

CPU-hidden-CUDA stages: `python -m vlm_research.query_role_ia_joint.run cpu`, then
commit source/CPU acceptance and run `...run freeze`. Freeze performs full model
SHA256, copies original manifest/mappings byte for byte, selects an actually idle
GPU and records source/model/input/dependency bindings. GPU stage rechecks that
UUID immediately before loading a single model; all work is serial. A failed
gate saves PARTIAL and stops. No background polling or external process killing.

After the GPU process exits, hide CUDA and run
`python -m vlm_research.query_role_ia_joint.analyze`; then use `...analyze --check`
for a read-only replay and saved-file hash comparison. Analysis checks full model
SHA256 again and records GPU release. Raw arrays live outside Git under
`/data/lk/ATE-resources/experiments/query_role_ia_joint_240/`.

Primary contrast joint-Full and interaction joint-I-A+Full are frozen before GPU
generation, along with joint-I, joint-A, paired harm/correction and unique harm
(Full/I/A correct, joint wrong). All use the same 20,000 image-cluster draws with
seed 20261102. No equivalence/noninferiority margin is asserted. Scientific A/B/C
interpretation remains descriptive; statistical non-significance is not safety.

The supplied manifest string is 59 characters. The unchanged historical file's
complete SHA256 is `258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c`.
No sample replacement, prompt change or answer-based filtering is performed.
