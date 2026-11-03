# Qwen3-VL Keep L0-L21: One Joint Writing Condition

This experiment adds only Keep-L0-L21 on the frozen 3,000 GQA questions.
All-Writer and No-Writer are independently verified historical records from
commit `96a66a30bc81db9ffe5c3ed77716673a81aaaa01`; No-Writer is reference only.

The sole primary comparison is Keep minus All-Writer. The prespecified margin is
-1 pp, evaluated using the one-sided 95% lower bound from 20,000 image-cluster
paired Bootstrap replicates, seed 20261104. Questions remain equally weighted.

Read `completion.json` for final completeness, `paired_comparison.csv` for the
primary result, and `report_zh.md` for the research conclusions. No complete
result should be claimed before completion says COMPLETE.

Each shard retains atomic, immutable per-question checkpoints. If collection is
interrupted, explicitly run `vlm_research.qwen3vl_keep_l0_21_resume` with the pinned
Qwen Python. It first refuses live jobs or busy devices, verifies existing
checkpoints, and archives only attempt metadata before starting the unchanged
frozen worker. Both GPUs must pass fresh Smoke again. It never modifies old
scientific checkpoints or automatically waits for a free GPU. Already complete
generation must proceed to CPU analysis, not be generated again.

The parent single-layer experiment and its frozen source/results are untouched.
