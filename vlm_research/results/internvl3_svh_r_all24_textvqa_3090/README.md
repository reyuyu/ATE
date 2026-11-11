# All24 visual reaccess except L2-L5

Exactly one user-requested configuration: KEEP=L0,L1,L6..L27 (24 layers),cutoff28.
Same frozen128 historical TextVQA Image-only questions/order.
128 NEW formal quality answers plus1664 previous Main predictions reused,1792 total.
No new timing/control/other windows/GQA/OCRBench/training/Windows transfer.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED preserved.

## Meaning and source

P-1 and real Decode may read originalImageKV in every nonSKIP layer.
L2-L5 answerAttention/QKV/QK/AV/o_proj are physically skipped, Norm/MLP/Residual retained.
R->Image still prohibited fromL20. This condition retains these two SVH restrictions relative to Native.
VisualHidden exists in all28 language blocks, exits AFTER L27 before tokenwise finalNorm.
There is NO Prefill language-block visual-row saving.
Actual originalImageKV persists in24 KEEP layers,256 visualKV each; SKIP cache empty.
Original RoPE/logical positions retained. Inherited engine is byte-identical to parent, only KEEP differs.
Parent Commit 1407275cbdffca5d4a23cbf8b8d233002e2840dd, all92 inherited Python files preserved; all96 as-run files frozen BEFORE GPU.
analysis_source_manifest.json binds postprocessing separately.
CPU official28layerQwen2 Float64,1prefill+8decode; independent cache/position/storage/mask/cleanup checks.
GPU3short/medium/long Dense/Physical Smoke and Native/No-op strict validLogits check precede128 formal answers.
Small-sample numerical observations do not lift the prior global numerical waiver.

## Model, input and provenance

OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
BF16/Eager,Torch2.6.0+cu124,TF4.49.0,Python3.11.8;batch1 greedy cap32,naturalEOS,logits_to_keep1.
Single448tile256visual,nativeChat,scorer-aligned short instruction,no OCR/Gold auxiliary input.
All131 prepared native token/pixel/prompt/roles bindings CPU-verified again.
GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, idle before loading, same GPU as previous physical windows.
Native/R0/R6/R14 original predictions still historical oldGPU, explicitly labeled.
Repeatedly used128 historical questions,NOT independent Holdout/noninferiority confirmation.
Official TextVQA ten-answer scores independently rederived via leave-one-out calculation for all1792 records.
Paired whole-image-SHA bootstrap5000,ordinary95%CI,uncorrected multiple comparisons.
24-layer and8/9-layer conditions have DIFFERENT budgets.

## Runtime and CPU audit

RUN=/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-all24-textvqa-3090
PREPARED=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
ExtraDeps=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps (isolated timm0.9.16)
Model=/home/featurize/ate-local-models/InternVL3-8B
All heavy resources localSSD; full raw evidence persisted on work.

Set ATE_DEEP_RUN=RUN,ATE_SEVEN_RUN=RUN,PYTHONPATH=RUN/code:ExtraDeps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_all24_textvqa.analyze then .report for CPU-only reaudit.
Official reference at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py;
identical uploaded reference in references/textvqa_eval.py.
A fresh GPU run needs separate authorization and new protocol/output/source bindings; do not rerun completed inference for reporting.

## Evidence

per_question.csv contains all1792 rawanswers/Gold/Tokens/scores/source recordSHAs.
paired_soft_changes.csv has13 comparisons; per_question_changes.csv has every paired change/harm.
kv_layer_audit.csv has1792x28 actualTensor shape/dtype/length/Live/Storage layouts.
physical_cutoff_audit.full.json.gz is lossless fullSmoke shapes/positions/probabilities; rawJSON stays server_only.
NormalEOS KV means are NOT fixed32Token KV@32; no new speed numbers are claimed.
server_only/predictions.sqlite is a complete1792-row SQLite backup,not partialWAL.
Raw audits/logs remain persistent; no weight/image/preparedtensor/dependency/cache/key upload.
artifact_manifest.json inventories files except itself/SHA256SUMS/post-push receipt.
SHA256SUMS covers the artifactmanifest and excludes itself.
server_only/git_publish_receipt.json records immutable GitHub HTTPS per-file bytes/SHA and SSH remote branchhead agreement.
No main change/forcepush/PR/old result overwrite/automatic next experiment or server shutdown. GPU released.
