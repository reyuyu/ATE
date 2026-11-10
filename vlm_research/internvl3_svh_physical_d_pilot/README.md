InternVL3-SVH Physical-D pilot (RTX 3090)
=======================================

This package implements the exact fixed L20 / L2-L5 intervention; it does not train,
search layers, replace the model, or physically delete Transformer blocks.

Conditions
----------
Native: official unmodified InternVL3-8B/Qwen2 eager language model.
Dense-D: historical mask + answer o_proj output zero; complete dense KV retained.
Physical-D: image hidden rows removed at entry L20; original P-1 omitted from all
Q/K/V/o_proj inputs at L2-L5, with a zero attention branch; subsequent decode does
not call attention or projections at those four layers. All norms, MLPs, and
residuals remain. Original logical RoPE positions remain. Early prefill may use
transient image KV for real R queries; persistent caches retain only text and no
KV at skipped layers. Compact storage is allocated, not a view of dense backing.

Numerical status and explicit user change
----------------------------------------
The first strict numerical gate FAILED. It was not loosened or marked passed.
CPU tiny official 28-layer Qwen2 float64 tests passed. GPU first divergence was
isolated at L2 AV under a changed, mathematically zero-masked reduction length.
Q/K/V, RoPE, QK, and softmax prefixes were bit-identical. Six complete smoke
generations matched, but logits drifted beyond the frozen threshold downstream.

After seeing the failure evidence, the user explicitly instructed:
"0.0004不大吧，在小测试集上得分差不多就行"
The effective protocol records this before any formal generation. The continuation
runs the frozen paired quality pilot and originally authorized efficiency pilot.
Strict numerical equality remains FAILED, irrespective of observed accuracy.

Frozen inputs / request counts
------------------------------
120 NEW official GQA testdev_balanced questions, six tasks x20; 120 unique image
SHA clusters; no QID/image SHA overlap with the supplied historical 3000 manifest.
Do not claim exclusion from unprovided exploratory cohorts.
Native/Dense-D/Physical-D each generate 120 normal-EOS answers (360 total).

Efficiency selects 30 different images beforehand, five per task at input-length
ranks 0,4,9,14,19. Each method gets 2 warmups, then five independently restarted
measurements per input (450 formal timings). Every request produces 32 output
tokens: one prefill and 31 real incremental decode calls, ignoring EOS.
Input and repeat-major execution rotates method order. No audit hooks are enabled
inside measured requests. Timing includes method installation/cleanup, the vision
encoder/projector/injection, prefill, real decode and final CUDA synchronization.
Input I/O, tokenization, preprocessing, H2D, model load and result serialization
are outside timing. The actual persistent cache stays live for byte/storage audit.

Runtime
-------
Use the pinned model revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9,
BF16/eager, Transformers 4.49.0, Torch 2.6.0+cu124, one RTX3090, batch1,
greedy32, logits_to_keep=1, single max_num=1 image (256 visual tokens).
Server constants and paths are in common.py. Inspect all constants before moving
to a new machine. No inference is started automatically by importing this package.

Reproduce on a fresh empty experiment directory
-----------------------------------------------
python -m vlm_research.internvl3_svh_physical_d_pilot.prepare
CUDA_VISIBLE_DEVICES='' python -m vlm_research.internvl3_svh_physical_d_pilot.cpu_tests
python -m vlm_research.internvl3_svh_physical_d_pilot.run

The original runner deliberately stops on a failed strict gate. The historical
quality-first continuation is a separate entry point, continue_pilot, and requires
the explicit user_override field and preserved initial failure evidence. Never
fabricate those files or silently waive the gate on another experiment.

The prepare entry point refuses to replace an already active run. Resume formal
requests only when protocol/source/input SHA bindings match. Every completed
quality sample and timed request has its own atomic SSD checkpoint.

After a completed, authorized continuation:
CUDA_VISIBLE_DEVICES='' python -m vlm_research.internvl3_svh_physical_d_pilot.analyze

statistics.py independently re-scores complete answers, checks exactly 360 quality
and 450 timing records, recomputes paired image-cluster bootstrap (20,000), and
writes all per-input and aggregate tables. analyze.py independently checks actual
layer KV lengths/live/capacity bytes and retains FAILED strict numerical status.
It never fills missing results with historical or estimated values.

Artifact policy
---------------
Persistent server results:
  vlm_research/results/internvl3_svh_physical_d_pilot_3090/
SSD checkpoints/logs:
  /home/featurize/ate-runs/internvl3-svh-physical-d-pilot-3090/

GitHub carries complete lightweight code, configuration, predictions/token IDs,
timings, per-layer KV layouts, numerical diagnostics and figures. Full original
checkpoints and detailed logs remain on the persistent server, mapped by SHA256
in artifact_manifest.json. Model weights, images, private keys and tokens are
excluded. No PR/main/force push, no 3000/seven-benchmark expansion.
