# GPU2 L0-L1 Attention and L2-L5 BOS mechanism

Base: e495adf4fd93337a3638981bbb66c2bbac81242d. Do not edit old sources/results.
Only new free-generation condition: L0/L1 o_proj output OFF at original P-1
and every Decode step. Existing generation, scoring, KV and row observer reused.
Full, L2-L5 OFF and L0-L5 OFF answers and native reference LogP are reused.

Run from vlm_research using /data/lk/深度学习/.envs/ate-vlm-research/bin/python.
Expose only CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6
for GPU stages. Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1,
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8.
CPU stages must use CUDA_VISIBLE_DEVICES=''.

1. python -m l0l1_bos.run prepare
2. python -m l0l1_bos.run cpu
3. Commit sources/protocol/CPU acceptance, then python -m l0l1_bos.run freeze
4. python -u -m l0l1_bos.run worker
5. After GPU2 exit: python -u -m l0l1_bos.analyze

Worker runs the necessary two real smoke samples before scoring. Resume checks
source/data bindings and atomic checkpoints; no duplicated question counts.
No main push, no other GPU/process termination, no training or new layer scans.
All new artifacts: results/gpu2_l0l1_bos_mechanism/.

Twelve cases are fixed before outcomes using seed20261031, two per task.
Mechanism capture is a single cached Prefill at P-1, never EOS prediction.
Only six attention ROWS per head, selected hidden/query vectors and actual BOS
cache K/V are stored. Prompt mass excludes BOS and image tokens. Figure A uses
visual-conditional patch probabilities with explicit raw image mass/log colors.
Full L0 attention is expected to be unchanged: intervention occurs after it.
Figure D is at most three post-hoc Full-correct/new-wrong examples, not a test set.

Accuracy interaction is F+E-D-Full (positive means extra joint harm), where
F=L0L1_OFF, E=L2L5_OFF, D=L0L5_OFF. Use per-question paired contrasts and whole
image SHA256 bootstrap clusters, not subtraction of independently computed CIs.
The ordinary paired95 CI and five-contrast family95 CI are both saved.
Taskwise intervals are descriptive. Heads are not independent samples.
Raw native-token LogP loss is signed, uses FP32 full-vocabulary log_softmax,
and is neither Gold LogP nor Accuracy. No speedup is inferred from output zeros.
