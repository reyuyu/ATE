# Execution Notes

- Worktree: /data/lk/ATE-worktrees/early-source-delay-240.
- Branch: codex/early-source-delay-240; base c67bc7d1a9d0603927b1f18b52777d790f4d1d85.
- Frozen code: 10c028a94fb93f2a04de17fee14130b922a575e2.
- Python: /data/lk/深度学习/.envs/ate-vlm-research/bin/python.
- Only physical GPU2 UUID GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 exposed.
- Environment and weights are the same fixed revision as the original complete
  answers; all shared sources and existing historical artifacts remain unchanged.

Commands run from vlm_research with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
CPU commands use CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=4, MKL_NUM_THREADS=4,
OPENBLAS_NUM_THREADS=4:

1. python -m early_source_delay.run prepare
2. python -m early_source_delay.run cpu
3. After committing code/protocol/CPU acceptance: python -m early_source_delay.run freeze
4. python -u -m early_source_delay.run worker
5. After GPU worker exit: python -u -m early_source_delay.analyze

GPU command additionally uses CUDA_DEVICE_ORDER=PCI_BUS_ID,
CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6,
CUBLAS_WORKSPACE_CONFIG=:4096:8,
MPLCONFIGDIR=/data/lk/ATE-resources/matplotlib.
Console is gpu_console.txt; Smoke must pass before formal scoring.

Only the frozen240 selected questions are processed. Full complete answers are
reused. A single native cached Prefill per question is used for row/vector
measurement; all new intervention rows are captured during their real first
generation Forward. The two Smoke cases additionally replay Full as a correctness
control; this is not a rerun of the old16-case pilot or3,000-question Full cohort.

## CPU Attempts

The first cleanup test mocked self_attn.forward and was rejected by the existing
signature guard. Its CPU report and traceback are retained under attempts/01
and failures/. Injection was moved to q_proj and a separate signature-guard test
added. No production guard was disabled.

A later CPU refresh after final source edits correctly refused to overwrite a
different completed CPU-validation binding. The earlier successful report was
archived in attempts/02_pre_final_source_cpu; the refusal traceback was retained.
The final33-test suite was rerun successfully against all frozen sources.
No tolerance or inference condition changed between these validation attempts.

Results, figures, reports and verification records are result artifacts added
after collection; they do not modify the frozen running Python implementation.
