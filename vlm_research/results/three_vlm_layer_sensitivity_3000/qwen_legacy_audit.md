# Qwen legacy audit

All 120 original Full and 4320 OFF full-vocabulary vectors verified by SHA256.
Independent FP32 recomputation exactly matches every saved target ID, signed loss, per-question absolute change, flip and JS.
The previous task heatmap explicitly plots mean SIGNED loss, not mean ABSOLUTE change. This is a metric/figure-purpose mismatch if used as the new primary figure, not evidence that the raw logits are invalid.
The prior plotting code uses real task labels and layers 0-35; its relative-depth denominator is checked separately in the new plotter.
New primary heatmaps use mean(abs(Full-OFF)); signed heatmaps remain a separately named supplementary view.
Native GPU regressions at L0, L2, L17, L24, L35 are a separate mandatory gate before formal Qwen collection.

Final gate: PASS. Both frozen historical smoke questions reproduced Native Full,
No-op, complete visual-feature replay and all five specified OFF vocabulary
vectors with maximum logit difference 0. See `qwen_gpu_smoke.json` and
`qwen_legacy_validation.json` (`gpu_regression=passed`). No GPU tolerance changed.
All 3000 new Qwen questions and 108000 OFF rows subsequently passed independent
raw-vector SHA256 and exact metric replay; see `coverage_audit.json`.
