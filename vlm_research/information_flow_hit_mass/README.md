# GPU1 First-Token Hits and Native Attention Mass

Append-only continuation of `codex/gpu1-information-flow-window9` at `4d800f0`.
No old sources/results, main, selected questions, prompts or window conditions change.
FIRST-TOKEN TEXT HIT is not complete-answer or VQA accuracy. No full Decode.

Use `/data/lk/深度学习/.envs/ate-vlm-research/bin/python` from `vlm_research`.
GPU stages require `CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417`.
Use HF offline mode, OMP/MKL threads=4, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Only GPU1 is exposed; preflight refuses to stop other processes.

1. CPU: `CUDA_VISIBLE_DEVICES='' python -m unittest information_flow_hit_mass.test_metrics information_flow_window9.test_masks -v`
2. Prepare: `CUDA_VISIBLE_DEVICES='' python -m information_flow_hit_mass.run prepare`
3. Commit source and configuration once, before GPU stages; do not edit executing source.
4. GPU1: `python -u -m information_flow_hit_mass.run sinks`
5. GPU1: `python -u -m information_flow_hit_mass.run smoke`
6. GPU1: `python -u -m information_flow_hit_mass.run run`
7. CPU: `CUDA_VISIBLE_DEVICES='' python -m information_flow_hit_mass.analyze`

All new outputs are in `results/gpu1_information_flow_hit_mass/`.
Sink definitions use the unchanged Phase 3.4 rule and three neutral prompts only.
All Sink candidates are used (not ROI-filtered); one definition per original image hash.
Native summaries stream existing eager weights without keeping full attention matrices.
Small per-head mass summaries and neutral-only maps are retained for reproducibility.
Each window's raw mass is averaged over the same clipped nine-layer span as masking.
Each repeated fixed-token LogP must reproduce the parent exactly, with no tolerance relaxation.

Associations are descriptive within-question window correlations, then question-level
medians. They are depth-confounded and not tests that Sink causes insensitivity.
