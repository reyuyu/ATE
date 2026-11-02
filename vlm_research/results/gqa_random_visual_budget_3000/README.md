# GQA Random Visual Access Budget Scaling

Independent branch: `codex/gqa-random-visual-budget`. Main is not modified.

The exact historical 3,000 questions (six GQA tasks x500, 2,409 image SHA256
clusters) are used for every model and condition. No resampling by performance.
Physical GPU0 is Qwen3-VL, GPU2 is InternVL3, and GPU3 is LLaVA. GPU1 is excluded.

## Frozen Intervention

Only original Prefill P-1 and every real incremental Decode Query may lose
direct access to original Image Keys. Other Prefill queries are native.
This is a before-Softmax additive `-inf` source mask, not `o_proj` bypass,
relay removal, a skipped block, or physical KV pruning.

For each model, seeds 20261102/03/04 each define one complete random layer
permutation. Prefixes of that permutation define nested 20/40/60/80% budgets.
Rounding is `floor(p*N+0.5)`. L0 is sampled normally, not specially retained.
Shared 0% Only Text and 100% Full endpoints plus 12 random conditions give
14 unique conditions per model. All complete generations are greedy, max32,
with independent caches and actual generated prefixes.

## Results

- `report_zh.md`: Chinese interpretation, including non-monotonic and negative results.
- `budget_summary.csv`: seed-mean overall Accuracy, min/max seed scores, paired CIs,
  Full/Only Text differences, first-token signed/absolute LogP changes and flips.
- `task_budget_summary.csv`: the same six-task summaries, including CompareAttr.
- `per_seed_summary.csv`: each seed's score, integer corrected/harmed counts,
  EOS/truncation/empty outputs, sequence changes, and lengths. Shared endpoints
  repeat only for plotting; they are not independent observations.
- `per_question.csv`: 126,000 unique model/question/condition records, generated
  answers and token IDs, errors in both directions, hashes, and format flags.
- `termination_summary.csv`: termination and descriptive anomaly screening.
- `layer_sets.json`, `per_seed_layers/`: frozen complete permutations and subsets.
- `figure1_accuracy_budget`: image-bootstrap uncertainty and seed ranges shown
  separately, avoiding confusion between two different uncertainty sources.
- `figure2_paired_accuracy_differences`: differences from Full and Only Text.
- `figure3_six_task_budgets`: all six tasks.
- `figure4_first_token_sensitivity`: signed loss, absolute change, and Top1 flips.
- `figure5_budget_quality_summary`: cross-model budget/quality overview.
- Each figure has PDF and PNG versions. Summary CSVs are their raw plotting data.
- `endpoint_bindings/`, `model_files.json`, `protocol.json`, `sample_manifest.json`,
  CPU/GPU validation, worker checkpoints/logs, final integrity and resource audits.

Bootstrap uses 20,000 whole-image paired resamples, preserving question weights.
Average the three seeds within each question BEFORE image resampling. The CIs
are conditional on the three fixed permutations; seed min/max is descriptive,
not a confidence interval over possible layer sets. All comparisons are
exploratory, with ordinary unadjusted intervals, not independent noninferiority.

Accuracy is percent; changes are percentage points. LogP values use FP32
full-vocabulary log_softmax at Full's fixed original Top1, not gold-answer LogP.
`generated_tokens` includes EOS when present. The first sequence difference is
zero-based; `-1` means identical. Repeated trigrams, newlines, role markup and
answer prefixes are descriptive format flags, not replacement quality scores.

Historical endpoint generations require exact input/model/protocol/logit
bindings. Missing numeric LLaVA Only Text first-token metrics are recovered by
one Prefill replay, without repeating complete historical generation. LLaVA
retains its historical full-position LM-head shape (`logits_to_keep=0`), whereas
InternVL explicitly uses its native `logits_to_keep=1` contract.

`validation_attempt_1/2` retain pre-formal startup failures and corrections.
No formal conditions or tolerances were changed. No COCO extension, extra seeds,
training, best-layer search, or inference-acceleration claim is made.

LLaVA's original process ended unexpectedly after 2,929 complete questions and
eight random conditions on the next question. No Python error was recorded;
the termination cause is not established. The original logs/preflight/validation
are retained under `llava/execution_attempt_1`, with the checkpoint inventory in
`llava/resume_ledger.json`. The identical frozen code resumed on GPU3, passed
Smoke again, and generated only the remaining 844 conditions. Already complete
checkpoint hashes must remain unchanged. `execution_summary.json` reconciles
35,156 + 844 = 36,000 new generations. Worker `complete.json` counters describe
the last process only; cumulative generation timing sums each condition once.
