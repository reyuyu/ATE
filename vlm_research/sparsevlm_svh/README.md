# Fixed SparseVLM-prefix + SVH handoff

All executable changes live in this worktree. Frozen SVH/FastV, official SparseVLM
source and shared model/environment remain unchanged. The official `einops==0.6.1`
pin is installed only in `.runtime-deps`; no shared package is upgraded.

The explicit chain waits for `ate-fastv-svh-sixbench-3090.service` to finish, checks
both shard completions, runs the authorized whole-MME supplement with the unchanged
FastV executor, independently rescores all six benchmarks, then gates SparseVLM on
six real images. Only a passing gate permits the new fixed 35,256-question matrix.
There is no idle-GPU polling, alternate device hunt or automatic restart on failure.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.sparsevlm_svh.prepare

# Commit source and frozen receipts before starting the explicit handoff.
/data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.sparsevlm_svh.handoff \
  --gpus GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 \
         GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6
```

SparseVLM nominal192 is `[300,200,110]` plus the original recycling and clustering,
after L2/L6/L15. Text-rater selection and pruning bodies are compiled from the pinned
official AST. Score-only paths observe the actual normalized layer input and the
actual SVH mask; they do not inject an answer residual or open answer image access.

RoPE is deliberately hybrid: real retained patches keep their previous anchors,
merged visual slots use image-start plus stage-local ordinal, and nonvisual text
always keeps its original absolute positions. Visual causal order follows the
physical sequence, with new merged tokens after selected tokens. The protocol does
not claim identity with standalone SparseVLM's compact-position execution.

Outputs and raw SQLite checkpoints stay under `/data`. The FastV supplement and
complete combined scores are in the prior result directory; SparseVLM outputs and
lightweight copies of its predecessor summary are in the new result directory.
Only local commits are made; no GitHub push, PR, main edit or data upload occurs.

## Explicit recovery of the completed FastV generation

The original driver failed only during CPU aggregation. All 32,882 original
record checksums pass. One MMBench row stored the sample digest after the pinned
official scorer lowercased its mutable option dictionary. `checkpoint_audit.py`
requires exact replay of that side effect and exact score agreement; it never
rewrites a checkpoint or accepts unrelated sample changes. New requests pass a
deep copy to the same scorer, preserving its output and the canonical manifest.

The failed, GPU-unlaunched handoff receipts are archived under `revisions/`.
The explicit recovery commands are:

```bash
/data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.sparsevlm_svh.prepare --recover-failed-unlaunched
# Commit the revised source and immutable receipts before continuing.
/data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.sparsevlm_svh.handoff --recover-completed-predecessor \
  --gpus GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 \
         GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6
```

Recovery first independently verifies and rescores every completed FastV row.
Only then may the authorized MME supplement start. The original service failure
and all original prediction bytes remain preserved as evidence.
