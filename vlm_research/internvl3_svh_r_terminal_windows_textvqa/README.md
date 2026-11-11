# Terminal late-window TextVQA follow-ups

Exactly accepted candidates: L20-L27 (8 layers, cutoff28), L19-L27 (9,cutoff28), L21-L26 (6,cutoff27).
Same frozen128 historical TextVQA Image-only questions/order.
384 NEW formal quality answers plus1280 prior Main records reused;1664 total.
No new timing/control/other windows/GQA/OCRBench/training/Windows transfer.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED preserved.
Any exact Dense/Physical agreement on three Smoke examples is a small-sample observation, not a new global equivalence claim.

## Parent and source

Parent Commit 84790bb7ce9904126054cfff39ef54d39a76f25a.
All82 inherited Python files preserved byte-for-byte in the new run. All88 as-run files frozen BEFORE GPU.
New engine.py is an independent copy with one terminal-cutoff28 exit branch plus explicit audit metadata.
engine_delta.patch shows the entire change. No old frozen engine is overwritten.
R->Image remains blocked fromL20. Answer image read only KEEP. Original logical RoPE/cache positions retained.
Answer QKV/QK/AV/o_proj skipped only L2-L5; all Norm/MLP/Residual retained.
NonKEEP persistentKV text only, SKIP cache empty, zeroScratch transient/originally aligned/masked.
cutoff28 keeps real image Hidden in ALL28 Transformer blocks, removes256 image rows only AFTER L27 before tokenwise final norm.
It has ZERO Prefill language-block visual-row saving. cutoff27 removes beforeL27 as previously.
CPU old-vs-new regression oncutoff26/27 verifies torch.equal logits and identical KV positions.
CPU Float64 target tests use official28layerQwen2,1prefill+8decode.
GPU3short/medium/long Dense/Physical Smoke each candidate and Native/No-op strict equality precede formal generation.

## Model and machine

GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, idle checked before loading. Same card as preceding new windows.
Native/R0/R6/R14 original quality records still historical oldGPU; labeled as such.
OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
BF16/Eager Torch2.6.0+cu124 TF4.49.0 Python3.11.8; batch1 greedy, max32 naturalEOS,logits_to_keep1.
NativeChat single448tile256visual and scorer-aligned short instruction. No OCR text/Gold in prepared model inputs.
All131 prepared tensor/native prompt/token/roles bindings CPU-verified again.
These are repeatedly used historical questions and user-selected exploratory follow-ups, NOT independent Holdout.
Paired whole-image-SHA bootstrap5000, ordinary95%CI, uncorrected multiple comparisons.
Different6/8/9 budgets cannot be conflated; cutoffs also move with last kept layer+1.

## Runtime

RUN=/home/featurize/ate-runs/internvl3-svh-r-terminal-windows-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-terminal-windows-textvqa-3090
PREPARED=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
ExtraDeps=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps (isolated timm0.9.16)
Model=/home/featurize/ate-local-models/InternVL3-8B
Model/images/preparedtensors remain localSSD, logs/checkpoints persist on work.

## CPU-only audit

Set ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN, PYTHONPATH=RUN/code:ExtraDeps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_terminal_windows_textvqa.analyze then .report.
Official reference lives at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py;
uploaded identical reference in references/textvqa_eval.py.
Official ten-answer scorer independently checked with separate leave-one-out calculation on all1664 records.
A NEW GPU run needs separate authorization and new protocol/source/output; do not rerun this completed job to recreate report.

## Evidence

per_question.csv includes all1664 answers/Gold/Tokens/scores/source recordSHAs.
paired_soft_changes.csv contains33 paired comparisons; per_question_changes.csv includes all harm/improvement cases.
kv_layer_audit.csv has1664x28 real tensor shape/dtype/length/Live/Storage layouts.
physical_cutoff_audit.full.json.gz is lossless fullSmoke shapes/positions/probabilities; rawJSON stays server_only.
NormalEOS KV means are explicitly NOT fixed32 Token KV@32 or latency; no new speed was measured.
server_only/predictions.sqlite is a complete1664-row SQLite backup, never partialWAL.
Raw audit/logs stay persistent; no model/image/preparedtensor/dependency/cache/key upload.
artifact_manifest.json inventories files except itself/SHA256SUMS/post-push mutable receipt.
SHA256SUMS includes the artifact manifest, excludes itself.
server_only/git_publish_receipt.json records Commit and immutable GitHub HTTPS per-file SHA plus SSH branchhead verification.
No main change/forcepush/PR/old-result overwrite. GPU released, server kept running, no automatic next experiment.
