# Two fixed inclusive writer-layer cutoffs

Only Keep-L0-L24 (25 layers) and Keep-L0-L26 (27 layers) are added. This is a
quality-only follow-up on the same frozen 1,000 GQA questions. Both keep early
attention normal and prohibit P-1/Decode direct image access at all 36 layers.

The controller reuses the unchanged historical LayerWriterMask and native HF
generation implementation. The new code varies only the set of open writer
layers; there is no output bypass, physical exit, timing or further search.

Results: `vlm_research/results/qwen3vl_keep_l0_24_26_gqa1000/`.
The protocol, input/model/source manifests and before-run Git commit bind this
execution. Baselines reference unchanged checkpoints inherited from the parent
commit. Raw image and model assets remain under the server resource directory.
All generated token IDs and answer strings are published in JSON and CSV.

CPU checks in the frozen Qwen environment:

```sh
python -m vlm_research.qwen_keep_extended.tests
python -m vlm_research.qwen_keep_extended.verify
```

Statistics and PNG/PDF figures in a NumPy/Matplotlib environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m vlm_research.qwen_keep_extended.analyze
```

Worker execution is `python -m vlm_research.qwen_keep_extended.run RANK STAGE`,
with stage `smoke` or `accuracy`. It requires the exact frozen GPU UUID exposed
through CUDA_VISIBLE_DEVICES, matching source/model/input hashes and a passed
dual-worker Smoke gate before accuracy. Checkpoints resume only with matching
bindings. A new GPU launch requires checking current idle resources and user
authorization; do not overwrite completed historical outputs.
