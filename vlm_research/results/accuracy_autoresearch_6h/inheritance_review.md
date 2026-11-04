# Inherited Evidence and Search Boundaries

This work starts from `3b46d36dc80e46fde778c6fc93191c8e33f24122`.
The six-hour clock starts at 2026-11-05 04:31:15 +08:00 and never resets.

## Read and Reused

- FullJoint V2 engine, runner, quality CSV, checkpoints and frozen input specifications.
- `644834d3e22194b4982758dabd59d3e479ef8ce3`: semantic search report and phase7 CPU budget table. Always Question + complete ASSISTANT was excluded for its 55.76% average receiver budget, not for a measured accuracy failure. The old 50% cap is not imposed here.
- `b13a21b07e84a0095731d10dc1e8854f514a7877`: GQA3000 All-Writer 87.0333%, Keep-0-19 87.0667%, Keep-0-15 86.5%. These are functional writing conditions, not evidence of current physical-path speed.
- `b483cdc524ce568fdcff5a871b1608b4aeecb322`: receiver cross-dataset report, VQAv2 input specifications, annotations and 500-question cohort. All-Writer itself can lose answers even with 32 writing layers; a later handoff is therefore a hypothesis, not a promised repair.
- The old space/newline/boundary searches are not repeated. Conditional-15 is historical context, not a new threshold search. Legacy Image+Answer bypass is not executed.

## New Execution Semantics

All non-Native candidates prohibit original P-1 and every real Decode query from accessing original Image Keys. L2-L5 Image and all pre-P-1 text queries retain Attention updates. Only the last-query and/or Decode Attention branch can be physically bypassed. Dense references, original RoPE positions and independent text-only persistent KV are retained.

First search matrix: All-Writer versus unconditional Question + ASSISTANT, crossed with (a) no answer bypass, (b) P-1 plus Decode bypass, and (c) Decode-only bypass. The fixed initial exit is before L20. The six hypotheses/configurations and each later promotion are stored before their own generation jobs.

The source commit is `65693faf28` for initial execution; runtime file SHA256 values in `execution_freeze.json` are the authoritative immutable binding. Analysis and dispatch code may evolve without changing GPU execution code.

## Data Discipline

Frozen input-only whole-image split: GQA DEV 244 / CHECK 606 / FINAL 2150; VQAv2 120 / 180 / 200. The historical quick60 and known smoke images are DEV. No FINAL answers are inspected before candidate freeze. All samples are historical research cohorts, not an independent external test.

Historical V2 regeneration audit: 60 questions x Native/Fast-P2/C3-V2 = 180 exact input, first-logit-hash and complete-token matches. No historical latency is used as a measurement of a new configuration.
