# TextVQA500: All-Writer plus dense C3 answer controls

N is the strictly audited native historical baseline. A prohibits P-1 and all
real Decode queries from reading image keys, while every R query remains open
at every one of 32 layers. B adds answer-query o_proj output zeroing only at
L2-L5. All compute and full native KV remain dense. Historical physical C3-V2
is separately labeled; its late writing restrictions and physical path differ.

The same frozen 500 questions, 309 image SHA clusters and official ten answers
are preserved. No OCR text, prompt edits, sample filtering or extra conditions
are introduced. There are no speed or FLOPs measurements in this runner.

Before the GPU run, CPU preparation validated original archive member hashes,
payloads, weight hashes and all prompt/token/FP16 pixel bindings. New inference
sources and installed Transformers source files are pinned in source_manifest.
Six true Smoke requests verify exact Native/no-op/HF logits, old native tokens
and first logits, unchanged pre-P1 hidden states across N/A/B at all32 layers,
zero forbidden probabilities, original RoPE, complete blocks and independent KV.

Environment: fixed LLaVA revision, Python ate-vlm-research, FP16/eager,
Transformers4.49.0, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms,
TF32 disabled, greedy max32 and logits_to_keep=1. The initial omitted cuBLAS
environment attempt is retained under diagnostics; no quality matrix started
before the corrected historical environment passed Smoke.

CPU reproduction:

```sh
python -m vlm_research.textvqa_allwriter_answer.tests
OPENBLAS_NUM_THREADS=1 python -m vlm_research.textvqa_allwriter_answer.analyze
python -m vlm_research.textvqa_allwriter_answer.verify
```

`run RANK smoke|accuracy` requires a valid source/model/data manifest, current
GPU authorization, a single exposed frozen UUID and the passed Smoke gate for
accuracy. Set CUBLAS_WORKSPACE_CONFIG before importing torch. Do not re-run
prepare over frozen results or launch a new GPU run without current resource
checks and applicable authorization. Resume requires exact request bindings.

Results are under `vlm_research/results/textvqa500_all_writer_c3_answer/`.
The original model and images remain on the server. Derived baseline JSON
retains quality fields and original archive/member SHA provenance; the inherited
raw archive is preserved and is never executed for timing in this experiment.
