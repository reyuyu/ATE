# Phase 3.8: GPU1-only L2-L4 sink-stage mechanism pilot

Base main: e89c8acce6e674ecfbd2c8acad57faf71bf4d547.
Frozen implementation: d0da03a848b8825941ec9f4d5df2461f87c6b0bf.
Experiment branch: exp/phase3_8-sink-stage. Do not auto-merge main.

## Scope and Implementation

New isolated hook/runner/test/summary files reuse the original model loader,
input mappings, candidate scorer, resource and cleanup helpers. No shared
algorithm, completed Phase2-3.7 result, historical Manifest, Transformers
installed source or GPU0 worktree is modified.

Only16 deterministic Phase3.4 main images (8 left,8 right) and C0-C4 are
tested. Every image retains576 visual tokens, original prompt/positions and
frozen Sink/ROI. C1/C4 bypass projected Attention output at P-1 in L2-L4 /
L5-L7; C2 bypasses MLP output at P-1 in L2-L4; C3 modifies only its
pre-softmax row to read BOS. Actual residual equations and local scopes
are audited; no layer is skipped wholesale and no global forward patched.

CPU121/121 pass, two-image GPU smoke passes before16 formal images. Actual
state_dict tensor hashes agree before/after each GPU stage. Both candidates,
all32 causal preceding-text states, Question, L1, input tensors, cleanup
and exact final Native Full are checked. Physical GPU1 UUID is recorded;
no other GPU task is killed. Logs, vectors, full-vocabulary logits, CSV,
four figures, selection/protocol and all resource/source hashes are saved.

## Actual Pilot Findings

- Full12/16; C1 Attention bypass12/16, mean Gain-0.352051 nats.
- C2 MLP bypass16/16, mean Gain+0.409424, recovering four right-label errors.
  All16 gold absolute LogPs increase, but7 individual Margins decrease;
  left/right gains are asymmetric. This is not population ability evidence.
- C3 BOS-only12/16, mean Gain-0.703857; answer decisions survive but
  numerical behavior is not Full-equivalent.
- C4 L5-L7 Attention bypass13/16, mean Gain-0.827881. Accuracy and mean
  Margin rank the two windows differently; no universal safety conclusion.
- C1 early relative hidden drift falls from mean0.653491 at L4 to0.084891
  after final RMSNorm. This is compatible with later compensation, not
  evidence that early cross-token interaction or tokens are unnecessary.

Report answers all eight questions and distinguishes Attention distributions,
actual residual-update vectors, sublayer interventions and final decisions.
The simple MLP-necessity / passive stable-stage interpretation is not
established, and the BOS stabilizing mechanism is not uniquely identified.

## Auditable Publication

Results entry: vlm_research/results/phase3_8_sink_stage/README.md.
The original machine report/summary manifest are preserved. A final console
completion line was appended after its original hash was recorded; the
known one-line difference and editorial report review are sealed explicitly
in artifact_finalization_audit.json, without changing measurements, source
or numeric tolerances. No failed formal attempt is hidden.

No expansion to48/240, new conditions/windows, Sink-strength scan, Physical
Exit, training or decoding. Stop after this pilot. API PR authentication
is unavailable; this ready-to-submit description and comparison link are
the authorized fallback, not a claim that a PR already exists.
