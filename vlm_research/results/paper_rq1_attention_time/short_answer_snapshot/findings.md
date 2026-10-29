# Short-Answer Findings and Reading Guide

Status: all four short-answer Confirmation cohorts are complete. COCO Caption
and TextCaps Confirmation remain running; the six-task study is still partial.
Do not interpret this delivery as completion of all 5,700 questions.

## Main Observations

- Full scores: GQA Balanced 70.30%, ScienceQA-IMG 66.625%, POPE Adversarial
  83.7778%, and TextVQA ten-answer soft score 48.87%.
- TextVQA shows clear early-Decode sensitivity. L0-L2 Decode-only decreases
  the score by 4.78 percentage points to 44.09%; paired marginal 95% CI for
  the change is [-6.28, -3.32] pp. The original 144-comparison familywise
  interval is [-7.73, -2.13] pp, beyond the predeclared -1 pp harm threshold.
  There are 63 per-question soft-score decreases and 10 increases. Combined
  L0-L2 also harms the score: -4.73 pp, with 91 decreases and 34 increases.
- GQA has larger observed losses from some Prefill interventions than from
  Decode-only. L14-L16 Combined gives 67.80% versus Full 70.30%, with 53
  correct-to-wrong and 28 wrong-to-correct cases. Its marginal CI excludes
  zero, but it does not pass the frozen familywise important-harm criterion;
  do not promote an unadjusted result to a multiple-comparison conclusion.
- ScienceQA L13-L15 Prefill-only and Combined each give 63.50% versus Full
  66.625%, with 53 correct-to-wrong and 28 wrong-to-correct cases. These are
  observed negative results, not proof of a familywise-significant effect.
- ScienceQA and POPE have exactly two native generated tokens in every Full
  answer: one content token followed by EOS (800/800 and 900/900 questions).
  Decode-only cannot change their first answer token by design. All eight
  Decode-only windows leave their primary scores unchanged; this is weak
  evidence about long-form content generation. ScienceQA can still change
  generated text in a few late-window cases without changing parsed accuracy.
- Native GQA length is also short: 750/1,000 answers use two tokens including
  EOS. TextVQA has longer outputs (mean 3.813 tokens including EOS), and
  Decode-only changes its text in 9.0%-25.8% of questions across the eight
  frozen windows. Task length and termination are important interpretation
  boundaries, not variables to tune after seeing results.

## Noninferiority Is Not Zero Observed Change

The frozen 144-primary-comparison family, 50,000 image-paired bootstrap
replicates and -1 pp short-score margin are unchanged for this partial upload.
There are 7/24, 0/24, 2/24 and 2/24 supported low-harm conditions for GQA,
ScienceQA, POPE and TextVQA, respectively. Inspect the complete 96-row table
in `report.md`; no window is newly selected using these confirmation scores.

For example, zero observed ScienceQA Decode-only score changes do not alone
establish noninferiority. The frozen zero-event resolution guard with 800
images and 144 comparisons has a rare-harm bound exceeding 1 pp. POPE has
only 150 independent image clusters, not 900 independent images. Lack of
statistical resolution is not evidence of damage either.

## Provenance and Verification

- Frozen GPU worker code: `906103b37b19d56e1752b5b3fc5bc04246ecd6a8`.
- Complete short cohorts: 3,700 questions, 2,950 images, 92,500 generations.
- All original sample bindings and checkpoint SHA256 values are preserved in
  `snapshot_manifest.json`. All short scores were recomputed and matched.
- `cpu_tests.txt` records 18 passing CPU tests. Original GPU smoke validation
  and environments are retained in the parent experiment directory.
- The adapter calls the existing frozen aggregation in a separate CPU-only
  process. It redirects output and the four completed dataset names locally;
  the numerical aggregation, scorer and comparison family are unchanged.
- No GPU inference is rerun for this snapshot, and no running worker source,
  model, prompt, data manifest or selected window is modified.
- Original generated token IDs and answers remain in the short-answer
  checkpoints. Per-question CSV files contain all 88,800 paired interventions.
- This study zeros an already computed output. It does not measure actual
  Attention compute skipping or inference acceleration.
