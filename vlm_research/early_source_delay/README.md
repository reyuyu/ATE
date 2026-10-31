# Early Source-Specific Attention Necessity and Delayed Compensation

Base c67bc7d1; review f1e661fd and e495adf4. Only new source paths and output
directory are modified. Shared mask, cached generation, scoring and row capture
are reused, not replaced. A1-A6 and all B interventions are Prefill-only.
Question excludes the original answer instruction as well as chat template.

From vlm_research using /data/lk/深度学习/.envs/ate-vlm-research/bin/python:

1. CUDA_VISIBLE_DEVICES='' python -m early_source_delay.run prepare
2. CUDA_VISIBLE_DEVICES='' python -m early_source_delay.run cpu
3. Commit frozen code/config/tests, then python -m early_source_delay.run freeze.
4. Only after idle GPU2 preflight: python -u -m early_source_delay.run worker
5. After GPU exit: CUDA_VISIBLE_DEVICES='' python -u -m early_source_delay.analyze

Use offline model/data. For GPU only expose UUID
GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 before importing torch.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4.
The worker refuses an occupied GPU2 and never terminates any other process.

Results: results/early_source_delay_240/. Per-condition atomic checkpoints bind
code, data and prior outputs. The native answers are reused, but native Prefill
measurements are necessary. Historical P-1+Decode answers are not P-1-only.
Two predeclared positive conditional interactions form one family of two.
The six A contrasts are exploratory, separately corrected within six.
