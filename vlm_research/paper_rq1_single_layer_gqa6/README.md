# Six-Task GQA Single-Layer Attention Output Ablation

Independent branch/worktree; no historical experiment or Caption worker is
modified or resumed. This is not a reproduction of the reference paper's
source-target edge knockout. Only one layer and original Prompt P-1's entire
Attention o_proj output is zeroed; MLP and all other queries remain native.

Seed20261030; pinned LLaVA-1.5-7B revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, FP16/eager/Transformers4.49.0.
The executable preregistration is `protocol.py`.

Original GQA Balanced validation is task-filtered without model-correctness
selection: ChooseAttr, ChooseCat, ChooseRel, CompareAttr, LogicalObj, QueryAttr.
Each has500 input-only hash-selected questions and an independently hash-frozen
80-question exploratory complete-generation subset. Same-image questions stay
on one GPU; actual per-task shard sizes and cross-task overlaps are reported.
No selected missing image may be replaced. All six types must have500 questions.

Main: Full +32 independent single-layer forwards per question. Track the SAME
Full Top-1 token using FP32 probability arithmetic. Full and changed vocabulary
logits are saved externally with hashes, enabling complete numerical replay.
Native BOS positions are located by actual tokenizer BOS ID, not hard-coded0;
post-softmax per-head masses and native row-sum audits are retained.

Generation: Full +32 single-layer greedy answers, max32tokens/EOS. Intervention
only on original Prefill P-1, once. Later Decode is native, with that condition's
own intervened cache. All33 Prefill vectors must exactly match the paired main
experiment. The scorer is the verified official GQA primary equality after the
pinned LLaVA submission converter, not the full auxiliary GQA evaluation suite.

Statistics use10,000 whole-image paired bootstrap draws and question-weighted
cluster sum/count ratios. Ordinary95% CIs are descriptive across192 primary
comparisons, not multiple-comparison significance evidence. The80 subset is
exploratory, not independent Confirmation. BOS correlations are not causal;
zeroing a computed Attention output is not real skipped computation or speedup.

## Run or Resume

Use `/data/lk/深度学习/.envs/ate-vlm-research/bin/python` from `vlm_research`.
Data preparation additionally needs the existing isolated PyArrow path:

```bash
PYTHONPATH=/data/lk/ATE-resources/benchmarks/phase4_0/tools python -m paper_rq1_single_layer_gqa6.data
python -m paper_rq1_single_layer_gqa6.tests
# Commit final source, data and CPU acceptance before execution freeze.
python -m paper_rq1_single_layer_gqa6.run freeze
CUDA_VISIBLE_DEVICES='' python -u -m paper_rq1_single_layer_gqa6.launch
```

Resume uses the same final launch command; atomic per-question checkpoints
validate source/config/input/raw hashes. Four matching real GPU Smoke results
gate main, complete main gates exploratory generation, and complete generation
gates unique CPU aggregation/figures/full-logit replay. Failure is recorded and
blocks new stages; samples are never silently skipped. Never loosen frozen
tolerances to make a failed check pass. GPU preflight refuses occupied cards.

Results: `vlm_research/results/paper_rq1_single_layer_gqa6/`.
External full-vocabulary NPZ: `/data/lk/ATE-resources/paper_rq1_single_layer_gqa6/`.
Images/model/scorer sources are reused read-only, not copied into Git. Expected
raw storage is about13GB before compression; reserve at least30GiB.

Reference: Zhang et al., *Cross-modal Information Flow in Multimodal Large
Language Models*, CVPR2025, arXiv2411.18620. The author dataset README is pinned
locally under the existing Phase4.0 resources; task labels follow its six
categories, but this study samples the original non-correctness-filtered GQA
validation instead of its author-filtered CSVs.
