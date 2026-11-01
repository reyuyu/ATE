# Qwen3-VL single-GPU feasibility

This module does not import or alter the LLaVA experiment environment. It performs
official local model generation, not attention ablation. Never use physical GPU1.
Full BF16, one visible RTX3090, no quantization or offload are enforced in code.

Execution order:

1. Inspect disk/GPU resources. Download and verify the pinned complete snapshot
   with `download.py`. HF direct access was unavailable on this server, so pinned
   HF Git/LFS metadata is transported via hf-mirror.com; the four weight shards
   come from official ModelScope only after exact LFS SHA256/size matching.
   Non-weight files remain bound to the pinned HF snapshot; all payloads are
   independently hashed locally. A `.partial` file is never a validated weight.
2. Only after download validation, create a separate Python3.11 venv on `/data`.
   Install `requirements.txt`, supplying the local CUDA124 torch/torchvision
   wheels via `--find-links /data/lk/ATE-resources/wheels`. Do not upgrade AFAC or
   ate-vlm-research. Record the actual resolved versions.
3. Run CPU unit tests, freeze the12-question manifest and source/protocol hashes
   using `run.py freeze`. The original six GQA pools are read-only inputs.
4. Recheck free GPU0/2/3. With exactly one `CUDA_VISIBLE_DEVICES`, run
   `run.py run --mode sdpa --gpu <physical-index>` followed by a separate
   `run.py run --mode eager --gpu <same-physical-index>` process. Both commands
   require `--model-dir` and `--out` paths. Model loading is strictly local.
5. After each process exits, run `finalize.py --out <results-dir>` to summarize
   actual evidence and verify tested processes are absent from the compute list.

The native model API performs cached greedy generation. A standard streamer
times the first token and later decode callbacks, without changing forward or
installing any attention hooks. Only per-generated-token score vectors are
requested; no full-layer attention matrices are requested or retained.

Image/query/P-1 mappings use the actual processor expansion, tokenizer offsets,
and official M-RoPE index utility. DeepStack vision extraction layer numbers are
not the language injection layer numbers. The static audit includes the pinned
installed implementation's source hash and relevant code snippets.

The twelve questions are technical smoke inputs, not an accuracy benchmark.
Passing this test does not establish feasibility for long contexts, retaining
all attention matrices, or any future hard-mask intervention.
