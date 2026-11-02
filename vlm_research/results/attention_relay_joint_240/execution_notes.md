# Execution and retained attempts

## Scope and isolation

Client date: 2026-11-02 (Asia/Shanghai). Independent worktree:
`/data/lk/ATE-worktrees/attention-relay-joint-240`.
Branch: `codex/attention-relay-joint-240`, baseline
`bd171312fd0489e8bc027744b7a976367da11668`.
The remote main observed before execution was
`e89c8acce6e674ecfbd2c8acad57faf71bf4d547`.
No main, historical module, other worktree, or third-party source was edited.

Only physical GPU0, UUID `GPU-a63c6362-dcfd-4a26-c44b-014b40127c89`,
may be exposed. GPU3 had an unrelated process during initial preflight; it was
not touched. Experiment1 has its own GPU1/worktree. Each launch checks GPU0 is
idle before loading exactly one model. Conditions run serially on that model.
Actual GPU tables, process checks, and logical/physical UUID binding are in the
successful smoke and completion JSON. No background waiting job is created.

## Cohort

Experiment1's manifest was available when selection was frozen, so this study
copies those exact240 samples, not a new draw. All six tasks contain40 questions;
all240 image SHA256 identities are distinct. The original3000-question pool,
question text, answer, image bytes, expanded Prompt IDs and P-1 positions are
verified. This is historical-source exploration, not an independent test set.

## Reuse

`paper_rq1_early_joint_attention.engine.JointAttention` supplies the validated
o_proj intervention. P-1-only and Decode-only instances have disjoint temporal
scopes. The old Combined mode is NOT appropriate: it zeros all Prefill queries.
`rq2_visual_sufficiency.engine.MultiImageKnockout` and
`rq2_decode_visual.engine.DecodeImageKnockout` supply the original cached
Prefill/true incremental masks, respectively. Existing Observer and
DecodeProbabilities inspect actual post-softmax probabilities. Mask and output
hooks are on different module sites and remain simultaneously installed in
Joint. Each condition starts with cache=None, never another condition's state.
All960 formal generations, including Full/Text-Only, are new.

## Retained preparation and smoke failures

1. Initial preparation used an incorrect guessed GQA evaluator directory.
   It stopped before GPU execution. The existing scorer's RESOURCES path was
   reused instead; official evaluator and converter still must match their
   unchanged published SHA256 constants. Traceback is retained under failures/.
2. The first GPU smoke stopped on an additional exact comparison of DEFAULT
   HF generation scores against historical manual-forward logits. Full/No-op
   logits, initial cache fingerprints and default HF greedy tokens had already
   passed. Transformers4.49's GenerationMixin sets logits_to_keep=1 by default,
   whereas the historical forward uses0 (all Prefill rows), changing the LM-head
   matrix shape. The failed process exited and released GPU0; no formal question
   had run. The failed console/source binding is retained in
   attempts/hf_lm_head_shape_reference/ and failures/.

   The new smoke keeps the DEFAULT HF token check AND adds a native HF call with
   logits_to_keep=0, matching the historical LM-head shape. Every effective logit
   of this aligned reference must be bitwise identical, with no new tolerance.
   Default HF score differences are explicitly recorded per step, not hidden.
   CPU tests were rerun against the revised source; Transformers generation,
   LLaVA and Llama files are SHA256-bound and unchanged. No model/intervention
   computation, historical output, or acceptance tolerance was changed to pass.

Earlier successful CPU validations for pre-final sources are archived rather
than overwritten. Successful smoke/CPU sources must exactly match the frozen
formal protocol. Any later failure is retained and stops that execution.

3. Freeze was initially invoked from vlm_research, but its Git path list is
   repository-relative. Its tracked-source assertion stopped before writing the
   protocol or loading a GPU. The unchanged code was invoked from repository
   root with PYTHONPATH pointing to vlm_research. The successful smoke/source
   hashes remain identical; the failed invocation is retained in failures/.

## Commands and artifacts

Pinned Python:
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`.
Stages run from the worktree's `vlm_research` directory using
`python -m attention_relay_joint.run STAGE`:
prepare, cpu, smoke, freeze, worker, analyze, verify.
The freeze stage alone runs from repository root with
PYTHONPATH=/data/lk/ATE-worktrees/attention-relay-joint-240/vlm_research so its Git
paths resolve to tracked files. Other stages run from vlm_research.
CPU stages expose `CUDA_VISIBLE_DEVICES=''`; GPU stages expose only the UUID
above with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUBLAS_WORKSPACE_CONFIG=:4096:8.
HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 prevent model downloads/changes.

Smoke, formal and analysis consoles are retained. Checkpoints are atomic and
bound to protocol/source/sample hashes; a completed checkpoint is never
overwritten with different data. Local raw first logits are not committed;
raw_logits_manifest.csv records every absolute path, byte size and SHA256.
No new research condition or dataset is run after this experiment.

## Output flags

EOS and truncation use the actual generated IDs and token32 limit. Empty means
the tokenizer's special-token-skipped, stripped text is empty. The repeated-output
flag is a limited lexical heuristic: at least6 whitespace words and fewer than
half as many distinct lowercase words. It is not a semantic repetition or
hallucination evaluator. All raw IDs/text remain available for review; no flag
causes samples to be dropped. First divergence is1-based and includes EOS/length
differences. The Full fixed-token loss is signed, whole-vocabulary FP32 log-softmax,
not Gold log probability.

## Final presentation

After960 formal generations completed, all model processes exited and GPU0
returned to27MiB/0% utilization. Frozen analysis emitted the original tables and
three PNG/PDF figures. supplement_transitions.py adds requested Joint-vs-single
paired transitions and an explicitly post-hoc Joint-only error list, solely from
the frozen CSVs. render_presentation.py moves Figure1 counts below the error bars
and makes Figure2's count axis integer-valued. Values, intervals, and model source
are unchanged. Chinese report interpretation was expanded from these real tables.
These result-side scripts are included in the final artifact hash audit.
