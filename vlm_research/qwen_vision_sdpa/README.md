# Qwen-only Vision SDPA continuation

Base: `213d4e3be766444e1ecd9a90ac479836f7756e9b`.
No preceding result or LLaVA execution path is modified.

CPU preparation uses the existing `ate-qwen3vl` Python with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES='' python -m vlm_research.qwen_vision_sdpa.prepare
CUDA_VISIBLE_DEVICES='' python -m vlm_research.qwen_vision_sdpa.tests
```

Prepare is intentionally one-shot. It verifies all 6,811 raw image files and
all model files, reuses the prior real-Processor input audit byte for byte,
freezes smoke cases and numerical gates before GPU prediction, and binds source
files by SHA. The GPU implementation imports the unchanged historical physical
engine; no repaired engine or alternate Text kernel is loaded.

After explicit current-round authorization, save `authorization.json` in the
new result directory with the authorization evidence and
`gpu_authorized_this_round: true`. Recheck actual idle UUIDs, GPU processes and
at least 20 GiB free disk before launch. Each process exposes one authorized
UUID through `CUDA_VISIBLE_DEVICES`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and
loads one model. Run `worker --rank 0/1 --stage smoke` first; both must pass
before `--stage run`. Worker source must match `source_manifest.json`.

Only official Vision SDPA is enabled. An official PyTorch backend allowlist
disables Math and cuDNN fallback. Read-only operator tracing confirms actual
Flash/Efficient CUDA SDPA in all 27 Vision blocks; all 36 Text layers remain
Eager. No timing, profiler, FLOPs measurement or kernel search is performed.

Every formal result has atomic payload/file SHA checks, per-question independent
Native/Physical checkpoints and an atomically updated SHA index. A partial,
unindexed write is not treated as completed. Corruption or changed input/source
stops the run. Both conditions for an image stay on its deterministic shard.

Any OOM, nonfinite tensor, memory margin below 512 MiB, unexpected backend,
changed mask/position/cache semantics or SHA mismatch stops the experiment.
There is no resolution, precision or algorithm fallback.

Analyze with the existing CPU research environment (NumPy 1.26.4). Full benchmark
scores require all paired questions; partial data never substitutes for the
complete population. Prior image-cluster bootstrap and official MME scoring
are reused. Raw smoke logits/features remain in the server resource directory,
with paths and SHA hashes in Git, not raw arrays in GitHub.
