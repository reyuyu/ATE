# Research Findings: Observational Only

All planned task/layer results are retained; no result-based retuning.

## Planned Region Contrasts

Positive early MLP-minus-Attention means the MLP update is larger using the SAME input-residual denominator. Negative early-minus-middle means the early update is smaller. Positive peak-minus-flanks means a local L14-L16 update-ratio increase relative to L12/L13/L17/L18. These are descriptive comparisons, not causal tests.

| Task | Contrast | Mean [ordinary paired95% CI] |
|---|---|---|
| ChooseAttr | early_attention_ratio_minus_middle | +0.272933 [+0.268586, +0.277261] |
| ChooseAttr | early_attention_norm_minus_middle | -9.073440 [-9.157570, -8.988383] |
| ChooseAttr | early_mlp_minus_attention_ratio | +0.074442 [+0.066489, +0.082314] |
| ChooseAttr | peak_attention_ratio_minus_flanks | +0.043494 [+0.036155, +0.051097] |
| ChooseAttr | deep_attention_ratio_minus_middle | -0.298348 [-0.301769, -0.295001] |
| ChooseAttr | deep_mlp_ratio_minus_middle | -0.157060 [-0.162674, -0.151289] |
| ChooseCat | early_attention_ratio_minus_middle | +0.284756 [+0.279680, +0.289773] |
| ChooseCat | early_attention_norm_minus_middle | -8.664055 [-8.767841, -8.564217] |
| ChooseCat | early_mlp_minus_attention_ratio | +0.075653 [+0.068102, +0.083217] |
| ChooseCat | peak_attention_ratio_minus_flanks | +0.086931 [+0.081479, +0.092421] |
| ChooseCat | deep_attention_ratio_minus_middle | -0.282354 [-0.287343, -0.277494] |
| ChooseCat | deep_mlp_ratio_minus_middle | -0.147859 [-0.152783, -0.143002] |
| ChooseRel | early_attention_ratio_minus_middle | +0.270156 [+0.266374, +0.273973] |
| ChooseRel | early_attention_norm_minus_middle | -8.895875 [-8.985948, -8.806185] |
| ChooseRel | early_mlp_minus_attention_ratio | +0.027140 [+0.021125, +0.033264] |
| ChooseRel | peak_attention_ratio_minus_flanks | +0.027710 [+0.022165, +0.033197] |
| ChooseRel | deep_attention_ratio_minus_middle | -0.318791 [-0.321595, -0.316020] |
| ChooseRel | deep_mlp_ratio_minus_middle | -0.143558 [-0.151236, -0.136034] |
| CompareAttr | early_attention_ratio_minus_middle | +0.304355 [+0.300544, +0.308078] |
| CompareAttr | early_attention_norm_minus_middle | -7.830958 [-7.899042, -7.762886] |
| CompareAttr | early_mlp_minus_attention_ratio | +0.050455 [+0.044167, +0.056724] |
| CompareAttr | peak_attention_ratio_minus_flanks | +0.047783 [+0.044450, +0.051159] |
| CompareAttr | deep_attention_ratio_minus_middle | -0.272941 [-0.275557, -0.270388] |
| CompareAttr | deep_mlp_ratio_minus_middle | -0.152226 [-0.154587, -0.149814] |
| LogicalObj | early_attention_ratio_minus_middle | +0.282085 [+0.277965, +0.286270] |
| LogicalObj | early_attention_norm_minus_middle | -7.849487 [-7.941434, -7.759545] |
| LogicalObj | early_mlp_minus_attention_ratio | +0.092505 [+0.084964, +0.099998] |
| LogicalObj | peak_attention_ratio_minus_flanks | +0.075276 [+0.069575, +0.081020] |
| LogicalObj | deep_attention_ratio_minus_middle | -0.283932 [-0.287324, -0.280535] |
| LogicalObj | deep_mlp_ratio_minus_middle | -0.166368 [-0.171056, -0.161713] |
| QueryAttr | early_attention_ratio_minus_middle | +0.278979 [+0.274558, +0.283342] |
| QueryAttr | early_attention_norm_minus_middle | -7.999231 [-8.113960, -7.887029] |
| QueryAttr | early_mlp_minus_attention_ratio | +0.068578 [+0.062423, +0.075070] |
| QueryAttr | peak_attention_ratio_minus_flanks | +0.121813 [+0.116537, +0.127000] |
| QueryAttr | deep_attention_ratio_minus_middle | -0.306121 [-0.310387, -0.301979] |
| QueryAttr | deep_mlp_ratio_minus_middle | -0.139598 [-0.145807, -0.133255] |
| ALL | early_attention_ratio_minus_middle | +0.282211 [+0.280239, +0.284255] |
| ALL | early_attention_norm_minus_middle | -8.385508 [-8.441663, -8.328590] |
| ALL | early_mlp_minus_attention_ratio | +0.064795 [+0.061389, +0.068243] |
| ALL | peak_attention_ratio_minus_flanks | +0.067168 [+0.063768, +0.070543] |
| ALL | deep_attention_ratio_minus_middle | -0.293748 [-0.295690, -0.291823] |
| ALL | deep_mlp_ratio_minus_middle | -0.151112 [-0.153476, -0.148704] |

## Questions and Boundaries

1. Early Attention magnitude: compare both early-minus-middle norm and ratio, plus layer curves. A nonnegative early ratio difference contradicts a simple small-update explanation even if ablation sensitivity is low.
2. Early MLP versus Attention: inspect the paired shared-denominator contrast. Larger norm alone is not evidence of causal dominance.
3. L14-L16 local enhancement: inspect the peak-minus-flanks contrast and A2; do not force the update peak to coincide with the historical sensitivity peak.
4. Sensitivity versus update direction: A3/A6 and the signed matched correlations describe alignment, not necessity. A large norm or a distinctive cosine does not prove why a perturbation is tolerated.
5. High BOS / low sensitivity: inspect A5 and the four matched correlation pairs, including per-layer and within-task/layer-centered results. Across-layer correlation alone can be depth-confounded.
6. Late versus middle structure: inspect deep-minus-middle Attention/MLP contrasts, net ratio, cancellation and cosines in the region summary.
Further research should test hypotheses independently rather than infer causal compensation or safe skipping from these observational vectors. No further GPU experiment was added in this run.

Raw FP16 captures: `/data/lk/ATE-resources/gqa6_update_decomposition/raw/`. The primary CSVs and manifest hashes provide the full review trail.
