# GPU1 Sliding-Window Information Flow

Scoped experiment only: four frozen GQA task pools, seed 20261029, 24 questions each,
three paths, centered nine-layer window over L0-L31. No training, Decode or extra windows.
No shared model helper or existing experiment source is changed.

Run from `vlm_research` with `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
Set `CUDA_VISIBLE_DEVICES=GPU-b08489af-0a8a-eb48-55ba-8346733af417` before Python;
set offline HF mode, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, OMP/MKL threads=4.

1. CPU: `CUDA_VISIBLE_DEVICES='' python -m unittest information_flow_window9.test_masks -v`
2. Freeze selection: `python -m information_flow_window9.run prepare`
3. Commit code, config and manifest on the independent branch.
4. GPU1: `python -u -m information_flow_window9.run smoke`
5. With unchanged source and passing smoke, `python -u -m information_flow_window9.run run`
6. `python -m information_flow_window9.analyze`

Artifacts: `results/gpu1_information_flow_window9/`. Native and masked prefills use
four same-prompt replicas; each masked row has a different window center. Only one
question is the statistical unit. Success checkpoints are exclusive and binding-checked.
No input replacement or output-based selection. Native vocabulary logits must restore
bitwise exactly after all interventions for each question.

Supplementary "correct" means strictly Native first-token hit on the Gold first token,
not full-answer exact match. The latter cannot be claimed without complete decoding.
This distinction is explicit in both the figure and report. Main curves use every question.
