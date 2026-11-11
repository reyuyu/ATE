# Qwen3-VL counterpart of ATE Figure7
Only true autoregressive decode image-key permission is varied. Prefill including P-1 remains native. This is a dense functional experiment, with no SVH skip, visual hidden exit or physical KV compression.

Model: Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; BF16; vision/text eager; Transformers4.57.6; Torch2.6.0+cu124; Python3.11.8; batch1; greedy, max32, naturalEOS, use_cache=True, logits_to_keep=1.

Original500 TextVQA Image-only questions/images/order/ten-answer Gold are copied exactly from Figure7 commit9b540dc6d5b7cce53c930fdfb322479f7478c534. Qwen uses its native chat template/processor/mRoPE/DeepStack. Nine 36-layer budgets are [0,4,9,13,18,22,27,31,36]. Continuous Middle start=(36+4-k)//2. One Random(seed20261108) permutation, nested prefixes. 30 distinct conditions share k0 and Native36 endpoints.

The approved OOM policy retains all500 input records, records original failures and BLOCKED conditions, and scores exactly the SAME complete intersection for all30 conditions. No zero filling, resampling, resolution/precision/backend fallback. The curve explicitly reports actual N/500, not full500 accuracy.

Reproduce on a server with the pinned model and exact original data:
1. Create an isolated local RUN folder with code/, resources/, references/, prepared/, checkpoints/, logs/, results/. Copy the .py modules here into RUN/code, resources/*.py to RUN/resources and frozen official references/textvqa_eval.py to RUN/references.
2. Copy state.example.json to RUN/state.json and update only machine paths/GPU after recording the new hardware; set ATE_QFIG_RUN=RUN. Keep the frozen original500 manifest from results/references/figure7/ and native question suffix.
3. Follow prepare.py to verify all original ID/image/question/Gold/order hashes and create prepared GPU-input bindings (Gold is never in model input). The original setup.py is retained on server; the input-validation receipt and manifest identify every exact source.
4. Run CPU tests. Freeze all source/protocol/input hashes BEFORE GPU. run.py refuses mixed source/protocol/GPU resume, runs native/no-op bitwise and fresh-vs-fork KV smoke, then all fixed conditions. CPU scoring uses the exact Figure7 M4C scorer plus an independent leave-one-out calculation.
5. resources/analyze.py produces a single four-curve PNG/PDF and image-SHA cluster-paired 20000-replicate bootstrap statistics.
No images, model weights, prepared tensors or keys are uploaded. Complete SQLite/raw logs remain in persistent server_only; manifests list SHA256. This experiment does not measure efficiency.
