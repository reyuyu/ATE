# Which Text Tokens Need to See?

Single-GPU LLaVA-1.5-7B FP16/eager experiment, native `logits_to_keep=1`.
No output bypass, cache pruning, training, or physical speedup implementation.
Only pre-softmax edges to original image keys change, at all 32 decoder layers.
P-1 and every real decode query are closed in every matrix condition.

## Frozen Token Policy

The lexical selector takes only question text, prompt offsets and token groups.
Core candidates are question words outside the explicit functional stoplist.
All subwords in a word are inseparable, including in the matched controls.
The final semantic set is the largest feasible whole-group subset whose token
count also permits exact question-tail and R-tail suffixes without splitting a
word. A deterministic left-to-right DP breaks ties. Every budget reduction is
recorded; no labels, images, margins, or outputs affect the choice.

`Drop-Semantic` removes the final selected semantic set, not extra unselected
lexical candidates. Scaffold excludes all core candidates and reports deficits.
Random controls shuffle whole groups using three fixed seeds keyed only by the
question text; a feasible exact-budget subset is chosen without label access.
This is a constrained whole-word random control, not uniform independent tokens.

## Execution

From this worktree root, using the original research environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:vlm_research /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.semantic_visual_writers.run cpu
```

Commit all source files. Then freeze (validates model content, original input
mapping, all historical All-Writer records and the 40 old image-swap pairs):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:vlm_research /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.semantic_visual_writers.run freeze
```

Commit the frozen artifacts. Set `CUDA_VISIBLE_DEVICES` to the exact UUID stored
in `results/semantic_visual_writers_240/protocol.json` before the GPU stage:

```bash
PYTHONPATH=.:vlm_research OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -u -m vlm_research.semantic_visual_writers.run gpu
```

The entrypoint refuses occupied devices and never waits or kills another job.
Strict real-GPU Smoke and all 40 No-Writer image-independence pairs are gates.
One model executes all conditions serially with independent native cache/prefix.
Per-condition atomic checkpoints support exact-source resumption; raw first
logits and No-Writer swap traces remain outside Git, with content hashes.
Preserve failure records and stop on failed validation or OOM, without changing
numerical tolerances, image resolution, precision, or writer rules.

After the GPU process exits, record the GPU release audit. Aggregate once on CPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:vlm_research OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.semantic_visual_writers.analyze
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:vlm_research OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.semantic_visual_writers.analyze --check
```

`--check` replays all 2,800 records, raw arrays, statistics and CSV serialization
without changing completed results. The three random seeds are averaged within
each image, not treated as independent samples or selected after seeing accuracy.
Only ordinary exploratory intervals and the seven prespecified primary-family
Bonferroni intervals are used. No new layer/token policies are run after this task.
