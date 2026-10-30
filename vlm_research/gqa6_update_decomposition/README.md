# Native P-1 Attention / MLP Decomposition

Independent observational experiment from single-layer GQA6 result221cdde5.
No historical source/result is changed. No intervention, Decode or training.
The protocol is frozen in protocol.py before GPU collection.

Input-only hash selection takes100 of each previous500 questions. One native
Full forward/question captures five actual P-1 nodes at every layer. Preserve
FP16 vectors; all norms/dots/cosines use FP32. Exact FP16-rounded residual sums,
native-logit identity and historical Full comparisons gate collection.

Use the existing ate-vlm-research Python from vlm_research:

```bash
CUDA_VISIBLE_DEVICES='' python -m gqa6_update_decomposition.data
CUDA_VISIBLE_DEVICES='' python -m gqa6_update_decomposition.tests
# Commit tested sources/data, then freeze the exact model/code/environment.
CUDA_VISIBLE_DEVICES='' python -m gqa6_update_decomposition.run freeze
CUDA_VISIBLE_DEVICES='' python -u -m gqa6_update_decomposition.launch
```

The last command also resumes frozen per-question checkpoints. Four isolated
GPU UUIDs run Smoke first, then collect. Occupied GPUs are never terminated.
CPU aggregation requires all600 questions and replays all actual vectors.
Image-cluster uncertainty retains same-image sharing, including across tasks.
Old500 and matched100 sensitivity curves are distinctly labeled; correlations
are descriptive only. Null zero-vector cosine/cancellation is not zero-filled.

Results: vlm_research/results/gqa6_update_decomposition/.
Raw vectors: /data/lk/ATE-resources/gqa6_update_decomposition/raw/.
