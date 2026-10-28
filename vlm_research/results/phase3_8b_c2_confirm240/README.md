# Phase 3.8b Result Index

Primary cohort:Confirmation224,112left/112right, independent of the16-image
C2 discovery pilot. Discovery16 and pooled Full240 are separate descriptive
summaries. Only C0/C2 are GPU interventions; B1 is a fixed offline score bias.

Frozen implementation:`210e99baa06dcf4a86cb6326d8f03e3330e94d7c`.
Branch:`exp/phase3_8b-c2-confirm240`. Original Phase3.8 and main are untouched.

## Files

- `report.md`:Chinese research report answering the ten requested questions.
- `summary_224.json`:primary paired confirmation statistics, including separate
  left/right results, bootstrap intervals, exact McNemar and contingency tables.
- `summary_16.json`, `summary_240.json`:discovery reproduction and pooled
  descriptive results; not extra independent confirmation.
- `manifest.json`, `experiment_config.json`:immutable cohorts, original sample
  metadata, image hashes, sources, model/environment and execution binding.
- `per_sample_scores.csv`:complete C0/C2 scores; strict positive Margin only.
- `offline_bias_baseline.csv`:fixed right+1.0 candidate Margin counterfactual;
  no fabricated absolute model LogP for this offline operation.
- `direction_and_absolute_changes.csv`:per-image DeltaD and genuine C2 changes
  in left/right, Gold/Wrong full-vocabulary LogP.
- `historical_full_differences.csv`:current native versus old RTX4090 values,
  diagnostic only, never used as the paired C2 baseline.
- `run/sample_<qid>/`:atomic checkpoint, sample scores/mappings/audits and both
  candidates' raw full-vocabulary prediction logits for C0/C2.
- `tests/`, `smoke/`, `run/validation.json`:ordered CPU/smoke/formal gates.
- `logs/`:completed stdout/stderr. Summary stdout is archived only after exit.
- `plots/`:all224 confirmation images, negative extremes retained.
- `validation.json`:independent original-scorer rescoring and local-edit audits.
- `independent_*audit.json`, `frozen_input_integrity_audit.json`:supplementary
  statistics and immutable-input cross-checks.
- `publication_validation.json`:closed-file hashes from the frozen summarizer.
- `final_publication_audit.json`:final packaging hashes, including any later
  closed console logs and this index. It excludes only its own self-hash.
- `execution_notes.md`:scope, provenance, development/gates and operation notes.

Code/protocol documentation is at `../../PHASE38B.md`. Checkpoint resume
requires the original frozen code commit and matching configuration/environment.
Do not run prepare over existing results. A fresh full reproduction should use
an isolated clone with this experiment branch at the frozen code commit and
the same verified local resources. No model weights or image copies are
published in Git; existing fixed-resource SHA256 bindings identify them.
