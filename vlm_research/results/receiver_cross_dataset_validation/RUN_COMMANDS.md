# Reproduction and exact resume

All new experiment data and outputs are under `/data`; the historical `/data1` files are read-only references.

The frozen source is `vlm_research/receiver_cross_dataset/`. Model, annotation, selected-image, input, receiver-set, and execution-source SHA256 records are included in this delivery. Historical GQA accuracy is not used as new-dataset predictions.

On the original host:

```bash
cd /data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation/vlm_research
ATE_VALIDATION_PYTHON=/data/lk/深度学习/.envs/ate-vlm-research/bin/python

# CPU preparation of a NEW isolated reproduction, before any inference:
CUDA_VISIBLE_DEVICES='' "$ATE_VALIDATION_PYTHON" -m receiver_cross_dataset.validate_cpu
CUDA_VISIBLE_DEVICES='' "$ATE_VALIDATION_PYTHON" -m receiver_cross_dataset.prepare

# Frozen inference / exact resume. Verify no existing own worker first.
# UUID is fixed internally; the module checks idleness before loading one model.
"$ATE_VALIDATION_PYTHON" -m receiver_cross_dataset.run

# Statistics and scientific plots; no model or GPU is loaded:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 "$ATE_VALIDATION_PYTHON" -m receiver_cross_dataset.analyze
```

For exact resume, preserve the original protocol/source manifest and checkpoints. **Do not rerun preparation or change source files before resuming**: each checkpoint is bound to the frozen source manifest, image, input tensor hashes, and receiver positions. A STOP record after an error/OOM requires diagnosis; it must not be silently deleted to force continuation. Do not launch a second worker while the first remains active.

Data acquisition is reproducible from `receiver_cross_dataset/acquire_sources.py` and `data_acquisition.py`. These use official VQAv2 validation archives, official TextVQA 0.5.1 validation annotations, individually selected COCO images and bounded TextVQA ZIP range reads. They reuse existing image caches, preserve original Questions/answers, and record hashes. Do not rerun acquisition into completed frozen manifests; reproduce in a separate isolated directory.

The official VQAEval Python3 syntax port is checked in, with original-file and port hashes in `scorer_validation.json`. The TextVQA scorer is the original pinned MMF vendor implementation; its hash is enforced at runtime. The complete 36-test CPU-suite output is in `CPU_test_suite.json`. Synthetic statistical checks are in `cross_dataset/statistical_validation.py` and `statistical_validation.json`.

After the worker exits, run the independent delivery audit and interpretation appendices:

```bash
"$ATE_VALIDATION_PYTHON" results/receiver_cross_dataset_validation/cross_dataset/delivery_audit.py
"$ATE_VALIDATION_PYTHON" results/receiver_cross_dataset_validation/cross_dataset/finalize_interpretation.py
"$ATE_VALIDATION_PYTHON" results/receiver_cross_dataset_validation/cross_dataset/post_qc.py
```

The independent audit compares every CSV label with its atomic checkpoint and original manifest and evaluates the predictions with the full official VQA/MMF evaluator paths. It also verifies GPU process exit. Raw images/model weights remain in the hash-indexed server resource directories, not in GitHub. Complete generated answers, token IDs, checkpoint JSON, input/receiver manifests, scoring statistics, plotting CSV, PNG/PDF figures, and audit records are committed.

`post_qc.py` adds explicit anomaly accounting, official answer-type tables and report previews, and improves only the Figure B legend and Figure D coincident labels. It preserves the frozen inference/statistics source and every measured value. The full official VQAv2 evaluator comparison differs by at most floating point rounding (2.22e-16); Native/No-op model logits were compared using exact `torch.equal` throughout.
