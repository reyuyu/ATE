# Single-layer Decode Attention functional necessity

Frozen LLaVA-1.5-7B FP16/eager, Transformers4.49.0. This is an output-update
ablation, not an acceleration implementation. Native whole-Prompt Prefill and
the first generated token stay unchanged. Only one selected layer's current
Decode `o_proj` output is zeroed; MLP and normal KV updates remain active.

400 input-selected images (COCO200, TextVQA200) x33 conditions. Workers use
physical GPU1/2/3, each image paired on the same GPU. GPU0 is never claimed.
Each condition uses a deep copy of a verified native Prefill cache and generates
its own continuation. No VLM training, classifiers, source pruning or multilayer
experiments. All prior sources/results and main are preserved.

From the `vlm_research` directory using the pinned research Python:

```bash
python -m paper_rq1_decode_layerwise_quality.data
python -m paper_rq1_decode_layerwise_quality.run cpu
# Commit the exact collector sources before freeze.
python -m paper_rq1_decode_layerwise_quality.run freeze
python -m paper_rq1_decode_layerwise_quality.validation scorer
python -m paper_rq1_decode_layerwise_quality.launch smoke generation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m paper_rq1_decode_layerwise_quality.analyze all
python -m paper_rq1_decode_layerwise_quality.validation data
python -m paper_rq1_decode_layerwise_quality.report
python -m paper_rq1_decode_layerwise_quality.validation delivery
```

Do not overwrite frozen manifests/checkpoints. Rerunning `launch generation`
only resumes validated missing conditions; occupied GPUs fail closed. Smoke
must finish on all participating cards before any formal generation.

Ordinary paired95% and Bonferroni family96 intervals use200000 image-cluster
bootstrap draws. Frozen raw noninferiority margins: CIDEr.02, SPICE.01,
TextVQA.01 (1pp). No outcome-based layer/threshold/data tuning. TextVQA uses
all ten human answers and the pinned official soft scorer, not binary accuracy.
COCO uses the same200-image references and fixed reference-IDF for every
condition. Absence of observed harm does not certify rare-event safety.

Collector sources are hash-frozen in execution_manifest.json. Postprocessing
is separate and never changes executing workers. Its source hashes are included
in the final delivery. See results/paper_rq1_decode_layerwise_quality/report.md.
