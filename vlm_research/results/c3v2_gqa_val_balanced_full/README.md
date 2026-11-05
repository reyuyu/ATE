# Complete local GQA val_balanced: paired C3-V2 evaluation

Target: 132,062 questions, two conditions, 264,124 complete greedy generations.
Do not interpret this file or the download inventory's COMPLETE as completed scoring.
Only a validated completion.json with status COMPLETE establishes completion.

Conditions:

- C3-V2: original All-R writing L0-L19, visual Hidden exit before L20,
  P-1/Decode no original Image access, physical answer Attention bypass L2-L5.
- C3-V2+L11-L24: same R permission and bypass, P-1 and true Decode Image
  access at L11-L24, visual Hidden exit before L25, only fourteen layers retain Image KV.

Local data is complete but uses MMEB images; original Stanford resolution identity
has not been established. This is validation, not hidden test/testdev.

Run from the worktree with the existing ate-vlm-research Python environment:

```bash
python -m vlm_research.gqa_balanced_c3.tests
python -m vlm_research.gqa_balanced_c3.prepare
python -u -m vlm_research.gqa_balanced_c3.run --driver
python -m vlm_research.gqa_balanced_c3.analyze
```

The inference commit, sources, samples, model and protocol must be frozen before
the driver. Each GPU uses its own transactional SQLite checkpoint. Never replace
or delete records on resume. No latency benchmark or new intervention is included.
