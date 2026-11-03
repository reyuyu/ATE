# Frozen receiver selector and research execution

The final selector is the original Semantic-k input rule plus the four native
ASSISTANT subwords immediately before P-1. It does not insert text. The P-1
colon and all decode queries remain blocked from original image keys.

`deploy.build_input` reconstructs the original whole-word groups and feasible
budgets from a native chat prompt, original tokenizer offsets, and expanded
prompt IDs. `deploy.load_selector` loads the frozen configuration and immutable
source snapshot referenced by `results/gqa5h_receiver_search/selection_rules.json`.
Neither API takes question IDs, Gold answers, image contents, model answers or
historical correctness. No POS parser is required by this Winner. An incompatible
template or tokenizer fails instead of silently selecting a different anchor.

```python
from gqa5h_receiver_search.deploy import build_input, load_selector

metadata = build_input(tokenizer, question, native_chat, native_prompt_ids,
                       image_positions, relay_positions)
selector = load_selector("vlm_research/results/gqa5h_receiver_search")
writers = selector(metadata)
```

For exact server-side CPU reconstruction of all frozen FINAL600 positions:

```bash
cd /data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=./vlm_research:/data/lk/ATE-worktrees/gqa360-structural-receiver-selection/vlm_research /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m gqa5h_receiver_search.reproduce --output /tmp/gqa5h-winner-reproduced.json
```

Expected SHA256:
`8bf47b46017f0074f2efe0c79720d8b3102089628116ab826326515d4bc33958`.

The complete repository supplies the original token-selection dependencies.
The command above uses the exact frozen server environment and input metadata;
it does not read Gold, model answers or image data, and uses no GPU.

`prepare`, `worker`, and `experiments` implement the initial frozen data split,
strict numerical smoke gates and immutable inference queue. `fit`, `fit_cv`,
`router_fit`, and `swaps` retain unsuccessful causal-risk and adaptive-budget
learning, including labels, image folds, parameters and measured durations.
`tune` freezes the complete shortlist before TUNE outcomes; `freeze_winner`
admits only one new FINAL method. `expansion` runs only that unchanged Winner
on the remaining historical pool after FINAL600. It adds a narrow process-local
authorization gate; it does not modify the original split or numerical engine.

`analyze`, `expanded_analysis`, `mechanisms`, and `figures` reproduce the retained
statistics and plots. `audit` rechecks checkpoints, compact resource SHA files,
all derived answers and scores, and model hashes. `report` and `finish` assemble
the delivery. Do not run collection modules against the completed immutable
experiment directory to perform a different experiment.

All confidence intervals are exploratory. The four predeclared FINAL paired
tests use Holm correction separately from their ordinary pointwise bootstrap
intervals. The 3000-question expansion includes development and selection data;
it is not a new external confirmation. Hard masks do not demonstrate physical
compute savings or latency speedup.
