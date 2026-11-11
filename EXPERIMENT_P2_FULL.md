# P2 Full and GQA Official Type Audit

Independent branch: `codex/qwen3vl-p2-full-gqa-type-audit`.
No changes to the historical SVH engine, original predictions, shared environments,
main, or GitHub remotes. The inherited strict Physical numeric status is FAIL_RETAINED.

## Frozen Generation

Initial generation source commit: `d78ba24c47`.
Base prompt-study commit: `d073f887d3e13634df2400b45cf710a81287f5b4`.

The exact P2 suffix is read from the historical protocol, never reconstructed by
paraphrase. The old source files and all new generation files have SHA256 gates in
`p2_full_protocol.json`. Historical Validation prediction reuse has a separate
independent raw-annotation, prompt, token-decode, and scorer audit.

Each physical GPU holds one model and executes both methods on its image shard.
All new generation uses the unchanged frozen BF16/Eager engine and original cap128.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest -v test_full test_analysis
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python freeze_full.py
CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u run_full.py --shard 0
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u run_full.py --shard 1
```

Do not refreeze an already running experiment. To resume, keep the protocol and
manifest byte-identical and use the existing per-shard checkpoints.

The hardware alignment addendum was frozen before ScienceQA scores were read.
Four historical Validation questions share an image with newly generated questions
but their prior GPU differed from the new image shard. They are re-executed once
on the new shard (eight method requests). The old records remain read-only, and
canonical selection always prefers the aligned record regardless of score.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python align_validation.py --freeze
CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417 PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python align_validation.py --shard 0
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python align_validation.py --shard 1
```

Run alignment only after that shard has finished and its GPU is actually free.
All generated results have atomic compressed SQLite records with uncompressed
payload hashes. Duplicate keys cannot overwrite a prior record.

## CPU Analysis

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_gqa.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_full.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python finish_report.py
```

The GQA audit uses original official testdev annotations, not labels inferred from
question wording. Historical val cohorts are separately bound to their original
Parquet by question ID, image ID, question, gold, and locally verified image SHA.
Its Arrow reader uses the existing video-model environment in a CPU-only child
process; nothing is installed into the frozen Qwen environment.

All intervals use 20,000 paired whole-image bootstrap resamples with
question-weighted estimates. Circular groups are scored before image resampling.
Heuristic diagnostics and descriptive post-hoc counterexample selection are
separately labeled. No uncorrected subtype interval is a confirmatory claim.

Natural-EOS timings are recorded for every quality request. The fixed32 timing
table explicitly reuses exact frozen P2 probes from the prior experiment, with
its original two-input-per-task scope and three measured repeats. It is not a
new all-cohort speed benchmark or evidence of operator acceleration.
