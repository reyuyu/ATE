# Paired Comparison with Native Full

Same 16 frozen images and original complete-candidate protocol; B0 is current-machine Native Full, not historical Full. All masks affect L0/L1 only. Positive Margin is correct; ties are wrong.

| Condition | Correct / N | Mean Margin | Mean Gain vs Full | Mean absolute per-image Gain | Mean gold absolute LogP |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 8/16 | +0.662354 | +0.000000 | 0.000000 | -17.169756 |
| B1 | 8/16 | +0.518799 | -0.143555 | 0.545898 | -14.211541 |
| B2 | 8/16 | +0.761963 | +0.099609 | 0.307617 | -16.059063 |
| B3 | 8/16 | +0.670166 | +0.007813 | 0.649902 | -16.005950 |
| B4 | 8/16 | +0.792480 | +0.130127 | 0.312256 | -15.692263 |

No condition changes correctness on any image. Gold absolute LogP and the two-candidate Margin are different measurements; increasing both candidates can leave ranking unchanged. Positive average Gain does not establish a beneficial effect in the population.

## Every Image: Final Margin

| Question ID | Gold | Full | B1 | B2 | B3 | B4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 081003882 | left | -0.117188 | -0.375000 | -0.304688 | -0.703125 | -0.289062 |
| 11928567 | left | -0.679688 | -1.812500 | -0.820312 | -1.898437 | -0.718750 |
| 09240692 | left | -1.273438 | -1.695312 | -1.488281 | -2.136719 | -1.503906 |
| 12316303 | left | -1.343750 | -2.140625 | -1.570312 | -2.339844 | -1.511719 |
| 02599393 | left | +1.664062 | +1.351562 | +1.359375 | +1.531250 | +1.671875 |
| 04468301 | left | +3.507812 | +2.218750 | +3.578125 | +3.390625 | +3.468750 |
| 01851960 | left | +4.351562 | +4.179688 | +4.554688 | +4.867188 | +4.437500 |
| 00223311 | left | +6.464844 | +5.777344 | +6.988281 | +7.140625 | +7.023438 |
| 10541771 | right | -0.234375 | -0.320312 | -0.062500 | -0.128906 | -0.023438 |
| 05862856 | right | -0.765625 | -0.968750 | -0.867188 | -0.429688 | -0.921875 |
| 03539611 | right | -1.828125 | -1.164062 | -1.531250 | -1.226562 | -1.179688 |
| 16312698 | right | -5.351562 | -4.750000 | -5.839844 | -6.464844 | -6.003906 |
| 1351224 | right | +0.304688 | +0.148438 | +0.523438 | +0.195312 | +0.414062 |
| 1228576 | right | +1.199219 | +1.726563 | +1.656250 | +1.984375 | +1.726562 |
| 02159371 | right | +1.707031 | +1.867188 | +2.226562 | +2.328125 | +2.226562 |
| 11283646 | right | +2.992188 | +4.257813 | +3.789062 | +4.613281 | +3.863281 |

The exact wide CSV contains each condition's Margin/Gain, full-vocabulary gold/wrong LogP, signed gold LogP change and correctness/flip/tie flags. See full_comparison.csv and the existing margin_vs_full.png, whose diagonal is equality with Full.

This table only reformats completed checkpoint scores; it is not a new run, a new condition or a second scientific aggregation.
