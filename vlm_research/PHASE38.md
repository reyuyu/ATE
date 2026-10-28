# Phase 3.8: GPU1-only sink-stage pilot

Independent branch `exp/phase3_8-sink-stage`, based on upstream main e89c8ac.
Only new phase38 files are added; shared algorithms and completed history
remain unchanged. No GPU0 worktree/source/results are edited.

The five frozen conditions are Native Full, L2-L4 P-1 Attention-output
bypass, L2-L4 P-1 MLP-output bypass, L2-L4 P-1 BOS-only pre-softmax causal
mask, and L5-L7 P-1 Attention-output bypass. All layers are zero-based.
No layer is skipped in its entirety. Actual o_proj/MLP outputs are edited
before their respective residual additions, and only at the real P-1.

Selection: seed3820261029, frozen Phase3.4 main48 only, 8 left and8 right.
Historical Full correct/wrong right strata are balanced if available.
No current model or intervention output is used to select data. All
images retain their original hashes, prompts, candidates and Sink/ROI sets.

Use the fixed Python environment and offline resources. Commit all source
before prepare; all subsequent runtime stages require that exact commit.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest test_phase38_sink -v
CUDA_VISIBLE_DEVICES='' python phase38_sink_stage.py prepare
CUDA_VISIBLE_DEVICES='' python phase38_sink_stage.py cpu
CUDA_VISIBLE_DEVICES=1 python phase38_sink_stage.py smoke
CUDA_VISIBLE_DEVICES=1 python phase38_sink_stage.py run
CUDA_VISIBLE_DEVICES='' python summarize_phase38_sink.py
```

GPU stages require the exact package pins, exactly one visible internal
cuda:0, enough free physical GPU1 memory, and prior CPU/smoke acceptance.
Temporary hooks are removed on normal/exception paths. Local zero/mask
audits, causal preceding-text hashes, pre-intervention L1, original input
hashes, candidate P-1 equality and Native Full restoration are checked.
Every GPU stage hashes all actual model state_dict tensors before/after,
as well as installed LLaMA source/function signatures. No global patch.

Checkpoint every complete image without overwriting it. Verify all input,
source and artifact bindings on resume; partial samples are preserved and
cause a stop for investigation. Do not mix incompatible runs. Only16 images
and C0-C4 may enter the one complete master summary.

Report raw BOS/Image/Question/other-text masses separately from the Sink
fraction within Image. Save both mean of per-head Sink fractions and ratio
of head-mean masses. Measure true projected Attention/MLP vectors against
their actual residuals, not Attention weights. Record hidden drift after
L1/L2/L3/L4/L7/L31 and after final RMSNorm. Raw vectors/logits/headwise native
rows, CSVs, tests, environment, source/model/data hashes and four plots are
saved in results/phase3_8_sink_stage. This is a small mechanism pilot, not
a population accuracy benchmark or proof that tokens can be deleted.

Review the actual outcomes to answer H1/H2; allow them to fail. Do not
automatically train, decode, run Physical Exit, widen windows/data or merge.
Push only the experiment branch and create a PR if account authorization
exists; otherwise provide the manual comparison link. Stop after completion.
