# Reproduction and evidence map

This is an exploratory TextVQA Image-only historical same-question comparison, not an independent Holdout.
The original NUMERICAL_EQUIVALENCE_FAILED is preserved. Structure/scoring passed under the explicit numerical waiver.

## Source and environment

The parent Commit 1885b03de8a396e178f39c6128c22a47999d3ef1 contains all 57 inherited runtime files.
The new package vlm_research/internvl3_svh_r_deep_exit_textvqa adds only dynamic cutoff handling and independent task orchestration.
source_manifest.json records all 62 as-run files frozen BEFORE GPU; analysis_source_manifest.json separately binds post-run code.
Model OpenGVLab/InternVL3-8B revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9, BF16/Eager, Torch2.6.0+cu124, TF4.49.0.
timm0.9.16 is isolated in RUN/extra_deps. No replacement of the original base environment.
All 131 prepared BF16 image tensors and native tokenized inputs were SHA-exact against historical bindings.

## Actual server layout

RUN=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-deep-exit-textvqa-3090
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
Model=/home/featurize/ate-local-models/InternVL3-8B
Image bundle=/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr
Current GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, explicitly authorized hardware change.
The historical Native/R0/R6/R14 quality records came from GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e.
All new quality and ALL speed records used the current GPU. No old latency was reused.

## CPU audit only (does not run GPU)

Use PYTHONPATH=RUN/code:RUN/extra_deps, ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN,
PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.
Run Python -B -m vlm_research.internvl3_svh_r_deep_exit_textvqa.analyze
then Python -B -m vlm_research.internvl3_svh_r_deep_exit_textvqa.report.
The scoring reference must exist at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py.
A byte-identical copy is uploaded under references/textvqa_eval.py.
The official normalization is reused; independent leave-one-out ten-answer scoring is recomputed separately.

## GPU reproduction (requires separate authorization after this completed task)

prepare.py recreates frozen inputs and strictly verifies parent records. It requires the parent persistent evidence directory
specified in common.py, whose raw checkpoint stays on the server and is not claimed to be uploaded.
reused_records.json.gz provides the 512 full reused bound records for independent CPU auditing.
Copy inherited parent source files byte-for-byte according to source_manifest.json into RUN/code.
For a genuinely new run use a NEW RUN/output directory and freeze protocol/source manifests again BEFORE GPU.
Set CUDA_VISIBLE_DEVICES to the authorized current UUID, ATE_SEVEN_ROOT to the parent stop-policy source root,
HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, TOKENIZERS_PARALLELISM=false, USE_TF=0, USE_FLAX=0.
cpu_tests.py precedes run.py. run.py loads the model once, performs Native/No-op plus 3x4 Dense/Physical Smoke,
then only the frozen 352 new quality requests. The fixed quality gate alone determines speed eligibility.
Do not launch run.py again merely to reproduce a report; this completed task has released its GPU.

## Artifacts

per_question.csv contains all 864 quality +96 speed saved predictions, raw text, Gold, Token IDs and original record SHA256.
kv_layer_audit.csv audits all 960x28 layers using actual captured tensor shapes, bytes and lengths.
physical_cutoff_audit.full.json.gz is lossless complete Smoke evidence; the small same-name JSON points to it and the raw server copy.
operator_input_rows.csv and attention_probability_audit.csv expose actual Q/K/V/o_proj, MLP/Norm rows and permissions.
speed_summary.csv uses 8 images x3 repeats, 2 warmups/method, 32 outputs with 31 real Decode; not 128-token timing.
vision_projector_injection_ms is cumulative INCLUDING the vision encoder, not an additional independent duration.
All scores and CIs are exploratory, ordinary unadjusted image-cluster paired bootstrap.

server_only/predictions.sqlite is the full valid SQLite backup (960 records), not a partial WAL copy.
Raw audit, logs, model-download receipts and prepared reuse metadata stay under server_only on persistent work.
Model weights, images, prepared .pt inputs, isolated dependencies, system caches and keys are excluded from Git.
No files are downloaded to Windows. No main change, force push, PR, training, GQA or OCRBench execution.

artifact_manifest.json lists uploaded light files and server-only raw files with SHA256, excluding itself and SHA256SUMS from its self-index.
SHA256SUMS covers every staged artifact including artifact_manifest.json, but excludes itself.
The post-push receipt is server_only/git_publish_receipt.json, outside Git to avoid Commit/self-hash recursion.
