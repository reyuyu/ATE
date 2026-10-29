# Native Full attention alignment (GPU0)

Read-only supplement to `p1_sliding_window_gpu0` at `2360834`. No A-Skip
interventions are run. Historical source and results remain unchanged.

From `vlm_research`, using the pinned ate-vlm-research Python:

```bash
python -m unittest -v test_attention_roi_alignment
python -m attention_roi_alignment.run freeze
# Freeze a code commit before running. Expose only physical GPU0's UUID.
python -m attention_roi_alignment.run run
python -m attention_roi_alignment.analyze
```

The manifest binds source hashes, exact64 questions, Phase3.4 ROI geometry,
six existing Sink lists and the26 missing Sink definitions. Only the latter
require three neutral Native Full passes each. Original candidate N+1 inputs
are retained to match historical same-GPU Full logits exactly. TextVQA uses
its original prompt with no candidate or ROI fabrication.

Per-question JSON checkpoints support restart with matching bindings. GPU0
must be idle at launch; no other process is terminated. The observer inherits
Phase3.4 `AttentionRows` and only adds true BOS mass. It discards full attention
matrices per layer and retains only small visual rows during one sample.

The primary16 are selected solely by historical processor geometry, not by
accuracy. All32 ChooseRel is a separate reference. ROI enrichment matches
Phase3.4: Question-token average within head, visual-conditional ROI share
per head, equal-head mean, divided by binary ROI patch area. Window start s
always averages Full layers s through s+2. Correlations across30 overlapping
windows are descriptive, not independent-sample or causal evidence.
