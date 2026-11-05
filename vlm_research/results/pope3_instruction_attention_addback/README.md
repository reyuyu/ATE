# Three POPE Instruction Visual-Reading Cases

Exactly three frozen cases, 28 conditions each, 84 full generations. No training, new prompt, new data cohort or additional intervention search.

- [Chinese report](report_zh.md)
- [Combined token / attention / single-token add-back table](token_attention_addback_summary.csv)
- [All add-back windows and complete answers](addback_results.csv)
- [All layer/head/token attention metrics](per_layer_head_token_attention.csv)
- [Truck 1351 gallery](figures/1351/README.md)
- [Boat 2459 gallery](figures/2459/README.md)
- [Vase 301 gallery](figures/301/README.md)
- [Truck instruction ROI close-up](figures/1351/instruction_roi_focus.png)
- [Validation](validation.json), [completion audit](completion.json), [execution notes](execution_notes.md)

Important: the official boat target and the specified upper-right vases are outside the actual model center crop. Another official vase is visible at lower left. The partially visible truck is the only specified target with a nonempty model-input ROI. Do not interpret answer recovery as direct visual recovery of a cropped-out target.

CPU regression command: `CUDA_VISIBLE_DEVICES= python -m unittest vlm_research.reverse_writer.tests vlm_research.pope_instruction_probe.tests -v`

Analysis command from the repository root: `CUDA_VISIBLE_DEVICES= python -m vlm_research.pope_instruction_probe.analyze`

Independent final audit: `CUDA_VISIBLE_DEVICES= python vlm_research/results/pope3_instruction_attention_addback/audit_results.py`

Use the pinned local environment described in `environment.json`; the compressed actual attention rows remain in the shared `/data` resource directory, with paths and hashes in `attention/`. All requested PNG, CSV, official ROI subset and vector case figures are committed here.
