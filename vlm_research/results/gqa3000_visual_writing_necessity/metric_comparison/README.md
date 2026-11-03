# Log Loss vs Accuracy

These are supplemental plots of the completed 3000-question experiment, not
new inference or a re-estimation of the frozen bootstrap intervals.

- `logloss_accuracy_aligned.png` / `.pdf`: shared layer axis, separate metric
  panels, pointwise and 32-comparison-corrected 95% intervals.
- `logloss_accuracy_overlay.png` / `.pdf`: two explicitly labeled units,
  symmetric axis limits, aligned zero lines, and original signed changes.
- `layer_metric_comparison.csv`: exact original estimates and intervals.
- `provenance.json`: frozen source CSV hash and baseline definition.

Both changes are relative to **All-Writer**, not native Full. Positive Delta
Gold NLL means higher exact-gold log loss; negative Delta Accuracy means worse
generated-answer accuracy. The two metrics do not share units or a common
direction of harm. The log-loss curve is Gold-answer NLL, not the historical
Full-top1-token LogP loss.

| Dropped layer | Delta Gold NLL (nat/token) | Delta Accuracy (pp) |
|---|---:|---:|
| L0 | -0.879658 | -0.2000 |
| L11 | -0.117873 | -2.6000 |
| L13 | +0.046826 | -0.1667 |
| L26 | +0.036309 | +0.0333 |
| L29 | +0.043720 | 0.0000 |

The largest accuracy harm is at L11, whereas the largest positive NLL changes
are at L13, L29, and L26. Lower exact-gold NLL does not necessarily imply higher
normalized generated-answer accuracy. Exact gold spelling/case/token form can
matter (for example, the official gold `white` versus generated `White`). This
observation does not establish the cause of the entire metric disagreement.

Reproduce on CPU from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python -m vlm_research.visual_writing_necessity.compare_metrics
```

All original experiment files, estimates, protocols, and completion hashes are
unchanged. No new causal or safety conclusion follows from plotting the curves
together.
