# Three-model first-token layer sensitivity

This module observes single-layer, original prompt P-1 attention-output bypass.
It never generates complete answers, scans decode steps, masks key sources, or
claims physical compute skipping. All 3000 original questions are preserved.

## Frozen execution

- Qwen formal execution: `aac6cdbcc2f1156d06eb667a2ddd8f131c2d90cf`.
- InternVL formal execution: `158e9c20e181d81e2a3503da9fc4c7275df64610`.
- Original shared protocol and inputs are unchanged across both runs.
- InternVL additionally freezes `internvl_native_addendum.json`: HF native
  generation uses `logits_to_keep=1`; the initial direct prefill adapter omitted
  this and failed bit-exact regression by 0.0625 BF16. The addendum restores the
  native head matrix shape, not a tolerance change. Initial failure logs remain.
- Existing model environments and LLaVA/Qwen/InternVL historical results are
  read-only. Full vocabulary logits are external, with paths and SHA256 in each
  committed checkpoint. No weights or activation batches are committed.

## Resume And Analysis

Run commands from this worktree root. `prepare` intentionally refuses to replace
frozen manifests. Never mix a new code version into an unfinished model run:
use its frozen execution commit and corresponding environment.

Qwen entry point: `python -m vlm_research.three_vlm_sensitivity.run qwen run`.
InternVL entry point: `python -m vlm_research.three_vlm_sensitivity.intern_native run`.
Both require `CUDA_VISIBLE_DEVICES=1`, a free physical GPU1, and passed CPU/GPU
gates. Other GPUs, offload, quantization and visual-budget changes are forbidden.
The workflow recognizes completed stages without loading or overwriting them.

After model subprocesses exit:

```bash
CUDA_VISIBLE_DEVICES='' python -m vlm_research.three_vlm_sensitivity.finish
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=1 python -m vlm_research.three_vlm_sensitivity.analyze
```

Use the existing `ate-vlm-research` environment for CPU analysis. No dependencies
are installed or upgraded. Analysis independently replays all saved metrics,
checks the common 3000 IDs and image SHA256s, and runs 20000 whole-image cluster
bootstrap resamples (seed 20261101, question-weighted). Signed and per-question
absolute changes are distinct; neither is gold NLL or generation accuracy.
Figure 1 has separate model-normalized colorbars with explicit raw nat maxima;
Figures 2-4 and their CSVs also preserve true absolute scales and relative depth.
