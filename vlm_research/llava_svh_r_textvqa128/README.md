# LLaVA-SVH-R / Native same TextVQA128 image-only
Model: llava-hf/llava-1.5-7b-hf @ b234b804b114d9e37bb655e11cbbb5f5e971b7a9. FP16/Eager, TF4.49, Torch2.6+cu124.
The 26 frozen_sources files are byte-identical dependencies extracted from 5ee2a0dbe78df336229b55fed14a4a6d0d880bed. No frozen algorithm was edited.
Native / SVH-R each128 requests, same original sample IDs/order/images/Gold as the InternVL B128; model-specific prompt and 336 processor preserved. No OCR text, no speed experiment.
SVH-R: answer KEEP L11..L24, image Hidden exit entering L25, R->Image forbidden fromL20, answer Attention skip L2..L5 and real compact selective KV. Original P-1 prefill K/V for skip layers still computed; future Decode QKV all absent.
Use current results/protocol.json and source_manifest.json as authoritative as-run settings. initial_attempt preserves the original failed nine-step numerical assertion; v2 assesses numeric thresholds only before native EOS, with9 synthetic steps for finite/cache/position structure. Post-EOS numeric failure is retained; no claim of fixed-length numerical PASS.
Entry points (use separate empty local SSD RUN and verified local model/images, never overwrite results):
1. Copy this directory to RUN/code and references/textvqa_eval.py to RUN/references. Export ATE_LLAVA128_RUN=RUN, same CUDA_VISIBLE_DEVICES UUID, CPU threads1, CUBLAS_WORKSPACE_CONFIG=:4096:8, HF_HUB_OFFLINE=1.
2. prepare.py reproduces CPU inputs from the pinned previous sample_manifest and local original images; documented absolute paths must be bound to verified local resources. CPU preparation contains no GPU inference. Source sample manifest is distributed in the parent commit and its SHA is recorded.
3. python -B RUN/code/run.py runs native-EOS correctness smoke and exactly256 paired formal generations with checksum-bound resumable SQLite. Full checkpoint stays on server, not GitHub.
4. python -B RUN/code/analyze.py independently checks records, official and leave-one-out scoring, actual KV, paired20000 image-cluster bootstrap.
5. archive.py and publish.py only persist/verify this experiment's allowlist on its own branch. They never run inference.
Source_manifest freezes only inference/preparation files; analyze/diagnose/archive/publish post-code is separately SHA-bound by analysis_source_manifest.json. Frozen official scorer normalization is shared by two independently written score aggregations.
Models, images, prepared Tensor files, dependencies and all credentials are excluded from Git.
