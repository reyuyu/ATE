# Frozen TextVQA500 Decode visual layer budget

This is a dense functional Section 3.3 mechanism experiment, independent of Physical C3. The budget is permission to read original image keys during **true autoregressive decode**. Entire native prefill, including P−1, stays native. No decoder/attention forward replacement or cache compression is implemented.

The run used `/home/featurize/ATE-section33-20261108`, an isolated worktree, and `/home/featurize/ate-section33-runtime/bin/python` (a system-site-packages environment overriding only Torch/torchvision and their required CUDA runtime dependencies). Model and dataset paths are defined in `common.py`; all are pre-existing local resources. No dataset/model download is part of this module.

The preparation evidence directory is `$ATE_BUDGET_AUDIT` (default `/home/featurize/ate-section33-audit-20261108`). Its `middle`, `l11`, `l10`, `l12` subdirectories contain the corresponding historical result files retrieved with `git show`, plus `source_commit.txt`. Evidence commits are recorded in `input_audit.json` and `protocol.json`. The original historical raw checkpoint tar files remain available in their existing repository result directories; they are read without modifying them.

1. `python -m vlm_research.textvqa500_budget.prepare` strictly checks the frozen cohort, tensor bindings, historical records and official scorer on CPU, then freezes the grid/statistical plan.
2. `python -m vlm_research.textvqa500_budget.run` refuses a busy GPU, freezes source/protocol hashes, loads one model, checks numerical/structural correctness and historical reuse on two questions, then runs only the 19 missing conditions. Each atomic checkpoint is authenticated by payload SHA and an execution-freeze identifier. Resume refuses changed source/protocol/bindings.
3. `python -m vlm_research.textvqa500_budget.analyze` requires a complete run and passing smoke. It rescores all predictions and uses one shared paired image-cluster bootstrap matrix. It creates only the main discrete accuracy figure plus required tabular evidence.

Set `PYTHONPATH` to this worktree and its `vlm_research`, `CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `USE_TF=0`, `USE_FLAX=0`. Large resumable checkpoints and logs stay on the server, outside Git. The compact final per-question file preserves all 11,500 predictions and traceability.

No existing benchmark algorithm, historical result or shared work directory is edited. Stop on mismatch/OOM/non-finite logits; do not silently change the backend, precision, prompt, images or cohort.
