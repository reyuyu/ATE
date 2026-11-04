# Classification provenance sources

These are exact copies of the preparation and labeling sources used before outcome analysis. `blind_labels.tsv` contains the 300 original input-only observations. They are audit sources, not commands to rerun on a completed study: setup/freezing scripts create worktrees and freeze records. Do not run them over existing results.

Reproduce the numerical analysis with `../analysis.py`, then `../plots_report.py` and `../validate.py`, using the frozen labels and historical data. They do not load model weights. Original images and the complete official OCR JSON remain at their documented `/data` paths. Review chronology and its limitations in `../classification_code_provenance.json`.
