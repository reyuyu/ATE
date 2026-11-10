# FastV-K3 + llava-SVH

Independent executor based on `e281754ef17b3accc09d6938b5b9f020048b2aec`.
Historical engines and the official FastV source stay unchanged.

## Validation Status

CPU tests use randomly initialized, 32-layer tiny Llama with 576 image positions.
They are **not** evidence of GPU/model-level numerical acceptance or benchmark accuracy.
The task must stay partial until real-image GPU Smoke and all 35,256 predictions pass.

## Execution

Use the frozen local Python environment from this worktree:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.fastv_svh.tests

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.fastv_svh.preflight
```

Commit executable code before GPU launch. Obtain actual idle UUIDs with `nvidia-smi`.
Then explicitly launch one-card Smoke, followed by the two-card data shards:

```bash
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.fastv_svh.run \
  smoke --gpus GPU-EXPLICIT-IDLE-UUID

/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.fastv_svh.run \
  driver --gpus GPU-EXPLICIT-IDLE-UUID-A GPU-EXPLICIT-IDLE-UUID-B

CUDA_VISIBLE_DEVICES= /data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.fastv_svh.analyze
```

These commands do not select an occupied GPU or wait indefinitely for one.
The driver refuses to start without a bound GPU Smoke and accepted datasets.
The explicit `resume.py` schedule preserves all 35,256 requested rows while running
the five verified benchmarks (32,882 rows); all 2,374 MME rows remain deferred for
the missing reference-image bytes. This is never marked six-benchmark COMPLETE.
Exact MME image restoration requires a new, preserved protocol freeze; do not edit
the manifest or acceptance JSON underneath an existing freeze to bypass the gate.
Resume uses the same protocol, partition, source hashes, physical GPU and SQLite rows.
A failure leaves `STOP.json`; investigate it rather than silently deleting failures.

## Selector And Cache

- L2 scoring uses actual normalized SVH L2 hidden, a separate full-shape Q/K/QK
  computation, native causal softmax, and the unchanged official top-k AST.
- Scoring has no V/O residual injection. Its extra matrix FLOPs and GPU event time
  are recorded explicitly; the score-only reference is used only during Smoke.
- The selector freezes original visual positions once before L3. All text and
  original RoPE positions survive 576 -> 192 -> 0 visual Hidden compression.
- Only text KV is persisted; layers L2-L5 allocate no persistent KV. Decode uses
  in-place text-buffer append and never copies the complete historical cache.
- No-op/Native and disabled-pruning/SVH gates retain strict `torch.equal`.
- The executor has one coordinated forward/context, not stacked FastV/SVH hooks.

No GitHub push, PR, weights/images publication, efficiency claim, or parameter search
is performed by these entry points. Missing historical row-level controls are never
reconstructed from aggregate scores.
