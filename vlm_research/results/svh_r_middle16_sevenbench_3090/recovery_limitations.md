# Data and Historical Control Recovery

This is a data-acceptance limitation, not a changed experimental condition.

- MME: all 2,374 rows are preserved in the frozen manifest, but the historical
  recovered `landmark/03d5e3bfc958be38.jpg` bytes are not present locally. The
  existing image SHA256 differs from the reference recovered SHA256. Neither
  affected question was silently deleted or replaced. MME remains BLOCKED.
- OCRBench: official annotations and scorer were recovered and hashed. The
  original 109,105,033-byte official image ZIP is absent locally. Both bounded
  direct requests to Google Drive endpoints failed connection establishment.
  The existing Hugging Face extraction is not accepted as equivalent image
  bytes. OCRBench remains BLOCKED.
- Previous complete predictions: reference Git commits contain summaries and
  Windows backup receipts, not the complete raw per-question predictions.
  Those receipts locate `D:\ATE-dataset-backups-3090\ATE-results\c3v2-reaccess-sevenbench`.
  A receipt is not a recovered prediction archive. Paired historical harm,
  rescue and difference intervals cannot be computed until the bound raw
  archive and mapped manifest are actually restored and verified.
- The five accepted datasets continue with newly generated Middle16 outputs.
  No old GQA output is reused as a new Middle16 prediction.

Restoring the exact files above requires a separately verified continuation;
the running frozen manifests and inference code are not modified in place.
