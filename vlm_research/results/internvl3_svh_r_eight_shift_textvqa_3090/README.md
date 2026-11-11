# Three eight-layer TextVQA follow-ups

Only user-requested windows L18-L25/cutoff26, L17-L24/cutoff25, L16-L23/cutoff24.
Same frozen128 historical TextVQA Image-only questions/order, no new sampling.
384 new formal quality answers,896 previous Main answers reused, total1280 records.
No new timing, extra control, other window, GQA, OCRBench, training or Windows transfer.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED remains.

## Sources and input

Parent Commit 1260af2e5e7a8a29d5b9cdbb06e06f106ce48d37, with74 inherited Python files retained byte-for-byte.
source_manifest.json freezes all78 as-run files BEFORE GPU; analysis_source_manifest.json binds post-processing separately.
Generic physical engine is unchanged; only the requested keep/cutoff parameters differ.
R->Image still blocked fromL20; answer image reads only KEEP; answer Attention/QKV skipped inL2-L5.
Original RoPE/logical positions unchanged. Cutoff removes all256 visual Hidden rows; no token-level selection.
Persistent visualKV only in8 KEEP layers, zeroSKIP cache, original-position zeroScratch transient/masked.
Official TextVQA ten-answer scoring independently recomputed for all1280 saved records.
Original Native/R0/R6/R14 quality is historical oldGPU; Deep6/7/8 and all new windows share the current GPU.
All comparisons are exploratory historical same-question comparisons, NOT independent Holdout or global layer optimum.
Whole-image-SHA paired bootstrap5000, ordinary95%CI, uncorrected multiple comparisons.

## Local runtime

RUN=/home/featurize/ate-runs/internvl3-svh-r-eight-shift-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-eight-shift-textvqa-3090
PREPARED=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
ExtraDeps=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps (isolated timm0.9.16)
Model=/home/featurize/ate-local-models/InternVL3-8B
Model revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, verified idle before load.
BF16/Eager, Torch2.6.0+cu124 TF4.49.0, batch1 greedy naturalEOS max32.
NativeChat, single448tile256visual, no OCR or Gold in model inputs.
All131 cached native tokenized/pixel/prompt/roles input bindings were rechecked on CPU.

## Reaudit without GPU

Set ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN, PYTHONPATH=RUN/code:ExtraDeps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_eight_shift_textvqa.analyze then .report.
Official reference is RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py,
and the same bytes are uploaded in references/textvqa_eval.py.
A genuinely new GPU run needs separate authorization and NEW output directory/source/protocol bindings.
cpu_tests.py precedes run.py. Do not rerun the completed GPU generation just to obtain a report.

## Evidence and storage

per_question.csv contains all1280 answers/Gold/TokenIDs/scores/recordSHAs.
per_question_changes.csv has all24 pairwise comparisons and harm cases.
kv_layer_audit.csv contains1280x28 actual tensor shape/dtype/length/live/storage layouts.
physical_cutoff_audit.full.json.gz is lossless fullSmoke evidence; rawJSON stays server_only.
operator_input_rows.csv and attention_probability_audit.csv expose actual execution and permission checks.
CPU float64 uses official28layerQwen2,1prefill+8decode;3short/medium/longGPU Smoke each condition.
All normalEOS KV numbers are labeled accordingly, NOT fixed32-token KV@32 or latency.
No speed has been measured for the new windows.

server_only/predictions.sqlite is a full1280-row SQLite backup, not partialWAL.
Complete raw audit/logs stay persistent; models/images/preparedtensors/dependencies/keys are not uploaded.
artifact_manifest.json inventories lightGitHub and server-only evidence except itself/SHA256SUMS/post-push receipt.
SHA256SUMS includes the artifactmanifest and excludes itself.
server_only/git_publish_receipt.json is post-push mutable to avoid Commit/self-hash recursion.
Every uploaded file is read back via fixedCommit GitHub HTTPS and matched by SHA256; SSH verifies branchhead.
No main change, forcepush, PR, server shutdown or automatic next experiment. GPU released.
