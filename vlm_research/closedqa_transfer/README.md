# Fixed Closed-Answer Transfer

Frozen ScienceQA-IMG500 and POPE COCO Random/Popular/Adversarial300 each.
Exactly five conditions; no search, training, constrained decoding or resampling.

Results: `vlm_research/results/ate_closedqa_frozen_transfer/`.

Use the existing pinned LLaVA environment. Each rank exposes one physical GPU:

```bash
CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 python -m vlm_research.closedqa_transfer.benchmark --rank 0
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 python -m vlm_research.closedqa_transfer.benchmark --rank 1
CUDA_VISIBLE_DEVICES= python -m vlm_research.closedqa_transfer.analyze
CUDA_VISIBLE_DEVICES= python -m vlm_research.closedqa_transfer.analyze --finalize
```

Check device availability before launching. The two GPU commands run concurrently;
do not run a second copy of a rank. Each image cluster belongs to one rank.

`run.py` is the immutable original quality runner. Its initial timing metadata
wrapper had a duplicate `condition` keyword; its STOP records remain published.
`benchmark.py` fixes only that serialization and is the supported execution entry.
It uses the same frozen generator, manifest, configuration and scorer, with a
separate timing-entry hash. Original quality checkpoints were not regenerated.

`execution_freeze.json` binds quality execution. `performance_entry_freeze.json`
binds timing execution. Each checkpoint has a payload SHA and input/model binding.
Identical POPE inputs may reuse a same-GPU prediction with explicit provenance;
all task questions remain separately scored. Timing is never reused from history.

CPU checks:

```bash
CUDA_VISIBLE_DEVICES= python -m vlm_research.closedqa_transfer.tests
CUDA_VISIBLE_DEVICES= python -m vlm_research.closedqa_transfer.analysis_tests
```

Do not overwrite frozen manifests, protocols or checkpoints to start a new study.
Resources and original model files are shared read-only under `/data`.
