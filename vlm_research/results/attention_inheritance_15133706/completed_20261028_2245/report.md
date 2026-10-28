# Single-case visual attention inheritance: 15133706

Question: **Is the young girl to the left or to the right of the boy?** Correct candidate: **right**; unchanged order: left, right. This case was selected by the user because of its earlier Physical K24 flip; no population inference is possible.

## Fixed intervention

Only L24-L31 and query **P-1=602** to all **576** image keys (positions5..580). Full prompt length603; original matched teacher-forcing input length604. No cache, generation, physical deletion, training or new images. FP16/eager LLaVA-1.5-7B and Transformers4.49.0 unchanged.

For each head h, let a be the **current intervened forward's** pre-edit visual row, m=sum(a), and p_ref the frozen Full source layer's row divided by its own sum. Replace only those cells by `m * ((1-lambda) * a/m + lambda * p_ref)` after FP32 softmax has been cast to native FP16 and before multiplication by current-layer V. lambda0 bypasses assignment. Full L23 and L24 references are each reused across all eight target layers, head h to head h; no claim that different depths' heads have identical semantic roles. Values, weights, nonvisual cells and all other queries receive no direct edits. Later layers' pre-edit attentions may naturally change through the intervened hidden state; local non-target invariance does not imply equality with Full.

Reference rows freshly captured from this same Full pass match the Phase3.4 cache bitwise; each head is normalized independently in FP32. Zero/invalid reference masses cause failure rather than substitution. No global monkeypatch or third-party file modification: only selected model-instance forward functions have private globals, restored in finally.

## Actual results

Native Full margin **-0.03125000**, historical Physical K24 **+0.03906250** (historical only; Physical was not rerun). Score is complete single-token candidate log probability with double log-softmax over the full vocabulary. Both teacher-forcing candidates are run separately and their P-1 logits must be bitwise identical. Raw candidate logits and log probabilities are retained.

| Frozen source | lambda | Answer margin (nats) | Gain vs Full | Prediction | Two-candidate diagnostic seconds |
| --- | --- | --- | --- | --- | --- |
| L23 | 0 | -0.03125000 | +0.00000000 | left | 0.348 |
| L23 | 0.5 | +0.02343750 | +0.05468750 | right | 0.316 |
| L23 | 1 | +0.08593750 | +0.11718750 | right | 0.320 |
| L24 | 0 | -0.03125000 | +0.00000000 | left | 0.287 |
| L24 | 0.5 | -0.03125000 | +0.00000000 | left | 0.248 |
| L24 | 1 | -0.03906250 | -0.00781250 | left | 0.290 |

Positive margin means right beats left **among these two candidates**; it does not imply high unconditional right probability. The original Full and Physical margins are close to zero. FP16 backbone / output logits remain in use; double log-softmax does not undo FP16 quantization. No full-FP32 backbone validation or independent statistical replication was performed.

![Margin comparison](margin_comparison.png)

## Correctness / numerical checks

- CPU tests passed before GPU execution; see `cpu_tests.txt`.
- Exact historical prompt/candidate/question mappings; P-1 is disjoint from real Question queries. Native Full log probabilities match historical Full exactly. Fresh source rows match cached rows exactly.
- lambda0 for both sources: entire P-1 vocabulary logits and final Question hidden states exactly equal native Full, including both candidates.
- All targeted local audits passed; 8 layers x2 candidates x6 conditions. Other queries and nonvisual cells are bitwise unchanged by replacement, finite and nonnegative. Final Question hidden states remain bitwise Full for every condition.
- Predeclared row sum tolerance0.002; observed max error **0.000296473503**. Per-head visual mass tolerance atol2e-5+rtol0.001; observed max absolute error **0.000112831593**, max relative **0.000559832028**. Conditional mixing observed max L1 **0.00451080967**. Every patched value equals the formula cast to FP16 bitwise, and conditional L1 passes a dtype/mass-derived round-to-nearest bound (details in layer audits). **8** layer audits exceed the initial nominal0.002 threshold; this is retained explicitly. No re-normalization of the full row, because it would alter nonvisual weights.
- Instance overrides and hooks restored; post-experiment native Full exact; installed Transformers library SHA256 unchanged. No NaN/Inf or CUDA OOM.
- Peak allocated **13640.731 MiB**, peak reserved **13734.000 MiB** across recorded forwards. GPU is shared; times include validation overhead, are diagnostic and not speedup measurements. Model load recorded separately in environment.json.

## Initial validation stop and arithmetic-only correction

`../initial_attempt_fp16_validation/` preserves the first attempt. Both lambda0 source-L23 candidate passes were exact. Source-L23 lambda0.5 stopped in L25 before obtaining a candidate score: conditional L1=0.00429884 exceeded the initial fixed0.002 threshold in a head with visual mass about0.00210; visual mass/non-target invariants passed. This was an insufficient validator for FP16 subnormal rounding, not a changed intervention or favorable outcome selection.

Before rerunning, the formula and FP16 computation were left identical. Validation was strengthened to require per-cell bitwise equality with the exact FP32 formula cast to FP16, and a derived rounding bound: `u=eps/2; d=tiny*eps; E=u*sum(abs(x))+576*d/2; conditional_L1 <= 2*E/sum(round(x))+2e-6`. This follows the triangle inequality and includes subnormal rounding; it is calculated from dtype and row mass, not fitted to measured margin or error. The original nominal threshold, failures and hashes remain recorded. Row sum and visual mass tolerances were not changed. No failed partial run is counted as a completed condition.

## Observation versus interpretation

L23 inheritance flips this two-candidate decision at both nonzero strengths; L24 inheritance does not. At L23 lambda1 the margin exceeds the historical Physical K24 margin, but these are distinct interventions. Correct-candidate full-vocabulary probability **decreases** from 8.3651e-9 (Full) to 7.2265e-9 (L23 lambda0.5) and 6.7564e-9 (L23 lambda1), despite the improved relative margin. This is not an increase in absolute correct-answer likelihood or a free-generation success.

- L23: lambda=0.5: gain+0.05468750, margin+0.02343750 (right); lambda=1: gain+0.11718750, margin+0.08593750 (right).
- L24: lambda=0.5: gain+0.00000000, margin-0.03125000 (left); lambda=1: gain-0.00781250, margin-0.03906250 (left).

This is an intervention on attention shape, not just an attention correlation. Any observed margin change is the effect of this precisely defined local replacement under this one FP16 input. It does **not** establish that inherited attention is semantically better, that a sink is harmful, or that it explains Physical K24: physical deletion affects all text queries, hidden states, sequence computation and many subsequent connections, while this intervention changes only one row's visual shape and preserves its mass. A sign flip in this weak-margin selected case is insufficient evidence of a general capability improvement. No outcomes were used to choose new strengths, layers, sources or samples.

## Reproduction and provenance

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest test_attention_inheritance -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_inheritance_pilot.py --output-dir results/attention_inheritance_15133706/new_independent_run
```

The runner refuses to overwrite an existing configured output directory. For intentional reruns specify a new `--output-dir`. Resource checks run before model load; failures are saved, other GPU processes are never terminated. See `experiment_config.json`, `environment.json`, `gpu_resource.json`, `validation.json`, `per_condition_results.csv/json`, `reference_distributions.npz`, per-condition audit JSON and `prediction_logits.npz`. All old experiments remain untouched. Stop here; no expansion.

## Run provenance

Completed on 2026-10-28, GPU0 only, from repository commit 7970be981e3462560cb66d83e1014eaa1b1ebdae. All 101 CPU regression tests passed (`cpu_regression.txt`); the runner also repeated its 7 focused tests. Model loading took 8.815371 seconds. Fifteen synchronized diagnostic forwards totaled 2.743861 seconds, including Full observation, twelve condition/candidate passes, and cleanup validation; these do not include loading, all host-side processing, or plotting. Per-condition two-candidate times range 0.247927–0.348083 seconds; no warmup/repetition speed claim is made. Full final-vocabulary logits are stored for both candidates under every condition in `prediction_logits.npz`. Original partial results and failures in the parent directory remain historical records, not merged observations.

CPU-only artifact recheck: `verify_results.py` verifies six CSV/JSON rows against raw logits, all 96 local audits and seven frozen implementation hashes; `artifact_validation.json` records success. `historical_artifact_audit.json` independently rechecks all 1143 files in the original handoff fingerprint list; no old source/config/result bytes in that list changed. This repository-only audit does not revalidate all dataset images or model weights; this run separately verifies its one image SHA and cached Full rows.
