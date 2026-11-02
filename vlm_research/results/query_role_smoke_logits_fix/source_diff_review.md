# Source Review

Base: `094e1d7916d1e18351bffaf950c68280ba29bc85`.

## Execution Change

`query_role_attention/engine.py` gains `logits_to_keep`, defaulting to 1,
passed directly to the supported LLaVA forward interface. Diagnostic value 0
retains the historical all-position LM-head computation. No output-only slicing
is used as a substitute for matching LM-head computation.

`RoleBypass`, roles, layers L2-L5, target rows, QKV, attention, residuals, MLP,
positions, cache logic, model files, processor inputs and scoring are unchanged.
The execution default is a new version, not the previously frozen version.

## Independent Validation

`query_role_smoke_fix/` adds a CPU/freeze/GPU entrypoint with no formal collection
loop. It first diagnoses only frozen question 00150746. Strict same-shape
comparisons must pass before the original three frozen smoke questions run.
Native versus all-position computation is an explicitly diagnostic cross-shape
comparison, not an equivalence gate. All matched-shape gates use `torch.equal`.

The recorder reads actual model forward arguments, actual LM-head inputs and raw
FP16 logits without altering them. It removes its hooks on exceptions. Comparison
JSON is saved before strict assertions, with compact differing prediction vectors.
No full attention matrices are captured.

The historical partial result directory, sample manifest, failure logs and old
protocol are immutable. Their complete file hashes are bound in the new protocol.
The old execution entrypoint remains bound to its OLD source hashes and must not
be restarted under the changed sources. Resuming formal collection would require
an explicit new execution freeze and authorization; this repair runs smoke only.

## Reproduction

Use the unchanged `ate-vlm-research` Python environment, `PYTHONPATH=.:vlm_research`.
With CUDA hidden, run `python -m vlm_research.query_role_smoke_fix.run cpu`, commit
sources, then run the `freeze` stage. The GPU stage requires exactly the frozen
physical GPU1 UUID in `CUDA_VISIBLE_DEVICES`, checks idle status again, and exits
after diagnosis plus three frozen smokes. There is no formal generation stage.
