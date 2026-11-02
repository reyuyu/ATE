# Attention Bypass x Text Relay, fixed240

Independent exploratory 2x2 study at bd171312. Historical modules are reused
unchanged. A-Skip composes the existing P-1-only and Decode-only output hooks;
V-Off uses the existing cached Prefill and true incremental Image masks.
Combined mode of the old output hook is deliberately NOT used: it also removes
other Prefill Query updates and would answer a different question.

Stages (run from vlm_research with its pinned environment):

```bash
python -m attention_relay_joint.run prepare
CUDA_VISIBLE_DEVICES='' python -m attention_relay_joint.run cpu
# Commit source, then expose ONLY the frozen GPU0 UUID for smoke and worker.
python -m attention_relay_joint.run smoke
# Freeze's Git paths are repository-relative. Launch that stage from repo root:
# PYTHONPATH=/absolute/worktree/vlm_research python -m attention_relay_joint.run freeze
python -m attention_relay_joint.run worker
CUDA_VISIBLE_DEVICES='' python -m attention_relay_joint.run analyze
CUDA_VISIBLE_DEVICES='' python -m attention_relay_joint.run verify
```

The worker verifies the post-smoke frozen protocol before loading a single model.
There are exactly960 formal generations. An immutable per-condition checkpoint
is the completion marker; raw first-token logits stay on the server with explicit
path/SHA256 manifests. No other dataset, intervention, training or optimization
is scheduled. A blocked GPU preflight stops immediately without background waits.
