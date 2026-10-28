# Execution Notes

## Boundary and Provenance

- New isolated worktree `/data/lk/ATE-worktrees/phase38b-c2-confirm240`;
  branch `exp/phase3_8b-c2-confirm240`.
- Before implementation, remote main was independently checked as
  `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`, and the completed Phase3.8 branch
  as `2415b69264d20c5da1c47c18bdb0b5adcafa69a7`.
- All85 artifacts listed in the original Phase3.8 publication manifest were
  independently SHA256-verified. No older sources or results were changed.
- All four GPUs were checked; physical GPU1 was idle with no compute process
  before smoke and formal launch. No other process was terminated, and no other
  worktree was edited. This task uses one worker, physical GPU1 only.
- Frozen code commit `210e99baa06dcf4a86cb6326d8f03e3330e94d7c` contains all
  runner, analysis, tests and protocol documentation before any confirmation
  GPU result was obtained. No source edits during execution or outcome tuning.
- Binding `f1d6159e751b6c0eda2db7d8b8024b2144f373321d5a5434c155cde6d48e0cbf`.

## Development and Gates

- Initial development tests found a synthetic statistics fixture missing the
  Gold/Wrong absolute LogP fields. The fixture was corrected before freeze;
  experimental data, intervention, thresholds and tolerances were not changed.
- CPU acceptance:132 tests, no failures/errors/skips; fixed packages and pip
  check passed. Historical tests use only a process-local default model path
  adapter to find the identical local snapshot. No environment files changed.
- Smoke IDs were fixed before freeze:02474907(left),00784767(right). Both
  reproduce original Phase3.8 C0/C2 raw prediction logits and Margins exactly.
- A separate optional SciPy cross-check was unavailable because SciPy is not
  installed in the fixed environment. No package was installed or upgraded.
  Exact paired-binomial arithmetic has explicit CPU reference tests; further
  independent standard-library validation is preserved in audit artifacts.
- Actual loaded model-state tensor hashes match before and after smoke. Formal
  validation separately checks its own complete state before/after, not merely
  parameter names or versions. Disk model files are SHA256-checked each launch.

## Publication

- stdout/stderr are captured outside the recursively hashed result directory
  until their generating process exits. Completed logs are copied into `logs/`
  before final publication. No live-console self-hash is used.
- Confirmation224 is the sole primary confirmation cohort. Discovery16 and
  pooled Full240 are reported separately, not counted as new independent data.
- B1 is offline right-plus1.0 only; no extra GPU condition, calibration fit,
  new sample, layer search, training, token deletion or decoding is authorized.
