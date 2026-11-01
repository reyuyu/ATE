# InternVL3 Layer-Stage Pilot

This module adds only the frozen 120-question protocol. It does not modify the
prior InternVL smoke, vendor model code, LLaVA/Qwen environments, or old results.

Run commands from the repository root using the existing isolated environment:

```bash
PY=/data/lk/深度学习/.envs/ate-internvl3/bin/python
$PY -m unittest vlm_research.internvl3_stage.tests -v
$PY -m vlm_research.internvl3_stage.prepare
```

`prepare` intentionally refuses to overwrite an existing frozen protocol.
`cpu_validation.json` records the test outcome before GPU execution. The checked
in protocol binds every runtime/analysis Python source and input manifest.

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=/data/lk/ATE-resources/cache/internvl3/modules
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
$PY -u -m vlm_research.internvl3_stage.run smoke
$PY -u -m vlm_research.internvl3_stage.run run
```

Both modes stop if physical GPU1 has an existing compute process. The formal
mode requires a passing smoke bound to the same protocol. Checkpoints are atomic
and bound to both the frozen protocol and the exact sample; complete records are
not regenerated. An interrupted partial sample replays its official native
generation and verifies exact identity before filling only missing conditions.

CPU-only plotting uses the existing environment that already has Matplotlib,
without installing anything or loading another GPU model:

```bash
OPENBLAS_NUM_THREADS=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python \
  -m vlm_research.internvl3_stage.analyze
```

Aggregation refuses partial/missing/extra records. Image-cluster bootstrap uses
20,000 fixed-seed draws, question weighting and ordinary percentile 95% CIs.
Head/layer observations are never treated as independent images. Historical
LLaVA `delta_logp` is OFF minus Full; comparisons explicitly negate it to match
this experiment's Full minus OFF convention.

Only native first-token logits are stored as small per-sample Safetensors outside
Git for audit. Attention observers immediately reduce the current query row to
five source masses; no full attention matrices are serialized.
