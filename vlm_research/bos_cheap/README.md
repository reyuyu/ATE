# BOS Cheap Attention Bypass Discovery

Independent from result221cdde5, reuses actual native-update resultb91f4ae2.
No original source/results changed. No model training or speedup claim.

The frozen official validation pool cannot supply100 unused LogicalObj images:
after all historical manifest exclusions,54 remain. Freeze554 unused-image
questions, explicitly exploratory-only, not the requested600-image confirmation.
No replacement, expansion, test-driven rule selection or synthetic annotations.

From vlm_research using the pinned ate-vlm-research Python:

```bash
CUDA_VISIBLE_DEVICES='' python -m bos_cheap.data
CUDA_VISIBLE_DEVICES='' python -m bos_cheap.tests
# Commit final tested sources and manifests before freeze.
CUDA_VISIBLE_DEVICES='' python -m bos_cheap.run freeze
CUDA_VISIBLE_DEVICES='' python -u -m bos_cheap.launch
```

The launch runs4-GPU Smoke, one Full feature pass per historical question,
CPU lightweight predictor fitting/calibration and freeze, then Full+32 original
single-layer ablations per new image. Per-question atomic checkpoints allow
resume. All old96000 ablations are reused, not repeated. Each image always
uses one physical GPU. Every stage checks idle GPUs without killing processes.

Primary rules use BOS-only inputs, no real mass or current Attention/MLP output.
The6-Key enhancement is fixed in advance and costed separately. Region/layer
thresholds, Layer-only, Length-only and non-cheap true-mass oracle are retained.
Only old training/calibration data chooses thresholds. New results never refit.
Logistic models operate on whole-layer P-1 labels, not head-level safety labels.

2000 whole-image bootstrap draws report descriptive intervals. Additional
bounded-cluster simultaneous Hoeffding ratio upper bounds avoid declaring zero
events automatically safe. The bound is conservative and the sample shortfall
prevents the requested independent confirmation claim. Optional joint gating
and full-generation validation are not executed; no joint safety inference.

Outputs: results/bos_cheap_discovery. Raw q/k/v, residuals and logits are external
under /data/lk/ATE-resources/bos_cheap_discovery, bound by SHA256. Full per-head
CSV remains local; identical lossless gzip partitions are committed to Git.
