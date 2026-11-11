# Deep7 L20-L26 follow-up

New user-requested fixed window only: keep=L20..L26, cutoff=27.
R->Image permissions still stop at L20. Original logical RoPE positions retained.
This is an exploratory follow-up on the same frozen historical B128, NOT an independent Holdout.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED remains.

## Counts and sources

128 new formal quality generations. No new speed, GQA, OCRBench, training or other windows.
816 real prior records reused: Main128 x6 (Native/R0/R6/R14/Deep6/Deep8) + Exit27_NoLateRead x48.
Parent Commit c5a1036abccc1ecf2d77d1b187bde55d64b8231a.
Deep6/Deep8/Exit27 and the new Deep7 used the same current GPU; four older baselines have the documented historical GPU limitation.
Actual GPU GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27.
Same fixed model OpenGVLab/InternVL3-8B revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
BF16/Eager, Torch2.6.0+cu124, TF4.49.0, batch1 greedy normalEOS cap32.
Image-only single448tile256visual; same native Chat Template and scorer-aligned short instruction.
Prepared inputs contain no Gold or OCR text; all 131 cached input tensor bindings rechecked on CPU.

## Locations

RUN=/home/featurize/ate-runs/internvl3-svh-r-deep7-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-deep7-textvqa-3090
Old verified prepared tensors=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Isolated timm0.9.16=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps
Model=/home/featurize/ate-local-models/InternVL3-8B
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
All runtime weights/images/prepared tensors remain on local SSD, no Windows transfer.

## CPU-only audit

Set ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN,
PYTHONPATH=RUN/code:/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_deep7_textvqa.analyze then .report.
Place the uploaded references/textvqa_eval.py at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py.
The primary official scorer and independent ten-answer leave-one-out scoring match on every saved record.

## Reproduction source

The parameterized engine is inherited byte-for-byte from the parent. No algorithm repair or new attention path.
source_manifest.json binds all70 as-run Python files, frozen BEFORE GPU; four new post-processing files are separately bound.
CPU tiny official28layer Qwen2 float64 checks 1prefill+8decode.
GPU Smoke has Native/No-op strict equality plus3 Dense/Physical short/medium/long tests.
run.py loads one model and permits only128 NEW formal records. Fresh independent caches.
A new GPU run would require separate authorization and a NEW output directory/protocol; do not relaunch this completed run.

## Evidence

per_question.csv includes all944 saved answers, Gold, scores, Token IDs and raw record SHA256.
kv_layer_audit.csv includes944x28 real cache layouts and storage bytes.
physical_cutoff_audit.full.json.gz is lossless full shape/position/probability evidence; rawJSON stays server_only.
Whole-image-SHA paired bootstrap5000, ordinary95%CI, no multiplicity correction or independent-holdout claim.
No Deep7 E2E@32 or KV@32 was measured; normalEOS KV is explicitly labeled.

server_only/predictions.sqlite is a complete SQLite backup (944 rows), never a partial WAL.
Raw checkpoints, audit and execution logs remain on persistent work. No model/image/cache/key upload.
artifact_manifest.json indexes uploaded light and server-only evidence, except itself and SHA256SUMS.
SHA256SUMS includes the artifact manifest but excludes itself.
The post-push mutable receipt server_only/git_publish_receipt.json is excluded from the hashed self-index to avoid Commit recursion.
GitHub readback verifies every uploaded file using immutable Commit HTTPS bytes plus SSH branch head.
No main changes, force push, PR, extra benchmark or automatic server shutdown. GPU released after generation.
