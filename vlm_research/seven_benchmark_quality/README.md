# Seven-Benchmark Native / Physical C3 Quality

## Test-dev data alignment, 2026-11-07

The new paper registry is `vlm_research/benchmarks/official_registry.json`.
Its `vqav2_testdev` entry uses exactly 107,394 official VQA v2 test-dev2015
questions and hidden gold. The seven source splits now total 260,068 questions.
The old val2014 source remains available as `vqav2_val2014_diagnostic` but is
not in the paper queue. No historical freeze or checkpoint is migrated.
The hidden-answer adapter never loads an answers file or computes local Accuracy.
Offline export is `python -m vlm_research.benchmarks.vqav2_submission --help`.
Only data preparation was authorized for this alignment; do not launch inference.

## Historical execution scope

Historical user scope (2026-11-06): GQA and VQAv2 were deferred. The active
dispatcher is
`vlm_research/results/seven_benchmark_native_physical_quality/execution_scope/schedule.py`.
It runs only POPE, ScienceQA-IMG, MMBench, TextVQA and MM-Vet, reusing the exact
existing checkpoints. Do not use the original seven-benchmark launch command
to resume this scope; it would also schedule the two deferred datasets.

This runner reuses the unchanged original LLaVA engine from the completed
MME/SEED quality experiment. It does not measure speed or alter the algorithm.
Each worker owns one UUID-pinned GPU and keeps one resident model. Every image
SHA cluster belongs to exactly one worker, including cross-benchmark duplicates.
Native and Physical requests have independent caches and identical input tensors.

The historical seven-source freeze contained 367,028 questions and required
734,056 generations. Its queue is not the new test-dev paper protocol.
Full input manifests, image SHA checks and model file SHA checks
are frozen before GPU execution. Each benchmark passes three input-selected
smoke cases on each GPU before either shard starts formal scoring.

Qwen is explicitly stopped: the later Vision SDPA experiment at
`53e9cc1f52481de338ab1a3149a41b63eaa5d162` failed its predeclared numerical
gate and never established worst-input Native/Physical viability. This runner
does not weaken that gate or substitute a different Qwen algorithm/backend.

MMBench uses the pinned official VLMEvalKit exact-matching option extractor
and circular all-rotation hit criterion, not candidate likelihood ranking.
MM-Vet predictions are generated but await an authorized official judge;
no paid API is called and no score or confidence interval is fabricated.

## Historical Commands (Not Run For Data Alignment)

Run from `/data/lk/ATE-worktrees/seven-benchmark-native-physical-quality` with
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`:

```bash
python -m vlm_research.seven_benchmark_quality.tests
python -m vlm_research.seven_benchmark_quality.prepare
python -m vlm_research.seven_benchmark_quality.seal
python -m vlm_research.seven_benchmark_quality.launch
```

Prepare/seal are one-time operations; never repeat them over a running or
completed freeze. Launch rejects a second live supervisor, occupied GPUs,
modified sources, invalid checkpoint bindings and recorded unresolved failures.
SQLite transactions commit each condition separately. An interrupted worker
validates every existing payload digest and its sample/protocol binding before
resuming. A failed worker stops only its paired owned worker, not external jobs.

Results are under `vlm_research/results/seven_benchmark_native_physical_quality`.
Progress and release audits are per worker. Completed benchmarks receive an
independent scorer/digest check and 20,000 image-cluster paired bootstrap draws.
No partial split is labeled COMPLETE. Disk availability is checked per question
and must stay above 20 GiB. No logits or image arrays are checkpointed.
