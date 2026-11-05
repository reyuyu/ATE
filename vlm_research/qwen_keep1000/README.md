# Historical Qwen3-VL Keep versus D reproduction

This runner directly imports the unchanged historical Keep and All-Writer
controllers. It performs quality inference only. It contains no benchmark,
kernel tuning, physical token deletion, or new condition search.

The frozen execution source, protocol, sample and model manifests are under
`vlm_research/results/qwen3vl_keep_l0_21_gqa1000_reproduction/`.
`source_manifest.json` covers inference sources; `analysis_source_manifest.json`
covers the separate analysis and independent delivery verifier.

The original model and image files remain in the server resource directory.
The checkpoint index records repository-relative paths and actual file SHA256;
Keep records refer to the unchanged earlier D checkpoints in the same commit's
repository tree. No full-vocabulary logit arrays are claimed to be published:
Smoke files contain every effective logit vector's SHA256 plus exact-comparison
statistics, and every condition's full generated token sequence.

CPU checks in the frozen Qwen environment:

```sh
python -m vlm_research.qwen_keep1000.tests
python -m vlm_research.qwen_keep1000.verify
```

Reproduce statistics and figures in a NumPy/Matplotlib environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m vlm_research.qwen_keep1000.analyze
```

The stored run used one GPU UUID per worker. `run <rank> smoke` runs the
predeclared ten Smoke questions assigned to each rank. Only after all twenty
pass and the equivalence gate is written does `run <rank> accuracy` allow the
requested All-Writer run; Keep reuses D when strict equivalence passes. Worker
resume requires exact source, protocol, model, sample and input bindings.
Launching another GPU run requires a fresh resource check and applicable user
authorization. Existing historical results must not be overwritten.
