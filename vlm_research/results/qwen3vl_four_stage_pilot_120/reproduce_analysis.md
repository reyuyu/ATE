# CPU Analysis Reproduction

Inference ran only on physical GPU1 at code commit
`56aecd386968c9f431282b7197eedfcfdb81dbd8`.
The frozen `protocol.json` binds every inference source, sample and model.
The final result commit adds only completed outputs and CPU comparison plotting.
Do not rerun model forwards to reproduce plots.

From the experiment worktree, with the existing ATE CPU plotting environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_four_stage_pilot.analyze
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_pilot_cross_model_plot
```

The second command is required: historical LLaVA Prompt includes Self, so the
cross-model Figure3 must compare Prompt+Self unions on both sides. The original
Qwen Figure1 and mass tables continue to keep Self separate. No inference values
are altered. `comparison_partition_compatibility.json` documents the distinction.

The first command creates a base report. The committed `report_zh.md` additionally
contains the final human curve reading and scope limitations; preserve it when
regenerating the base report. The three predeclared depth probe ranges are not
fitted stage boundaries. All intervals are descriptive ordinary 95% intervals,
not simultaneous inference across 36 layers or independent confirmation.

Raw vocabulary logits and per-head masses remain outside Git in the directory
bound by `protocol.json`. The per-question JSON/CSV outputs retain their paths
and SHA256. Their integrity is checked in `post_run_validation.json`.
