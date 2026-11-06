# Seven-Benchmark Native / Physical C3 Quality

This runner reuses the unchanged original LLaVA engine from the completed
MME/SEED quality experiment. It does not measure speed or alter the algorithm.
Each worker owns one UUID-pinned GPU and keeps one resident model. Every image
SHA cluster belongs to exactly one worker, including cross-benchmark duplicates.
Native and Physical requests have independent caches and identical input tensors.

The seven source splits contain 367,028 questions. LLaVA requires 734,056
generations. Full input manifests, image SHA checks and model file SHA checks
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

## Commands

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
