# Physical Visual Exit 32+32 Pilot

Read-only Phase 4.0 ScienceQA-IMG and TextVQA samples, seed20261029,
32 distinct actual image SHA256 per unit, selected before any model output.
Run only one prechecked empty physical GPU0 RTX3090, one local process.

Use the unchanged `physical_exit.py` implementation and `TOLERANCES`.
Full/K16/K24 all regenerate complete prefixes without any KV Cache,
Greedy, EOS or32tokens. No new cache algorithm, intervention or training.

Before scoring, CPU regression and first frozen sample per unit GPU smoke:
manual uncached Full versus native cached AND uncached generation tokens;
bitwise initial K32 versus native Full hidden/logits; actual deletion and
original position checks; existing Mask Exit numerical equivalence at every
K16/K24 smoke generation prefix. Any failed gate stops the run, preserves
failure records and prohibits interpreting scores. Never relax tolerances.

`python -m visual_exit_cross_dataset.run --freeze` freezes selection and code.
After the CPU gate and source commit, run
`CUDA_VISIBLE_DEVICES=<physical GPU0 UUID> python -m visual_exit_cross_dataset.run`.
Use the pinned environment and offline resources. Output is
`vlm_research/results/visual_exit_cross_dataset64/`.
Each complete three-condition sample is an atomic checkpoint. Resume requires
identical manifest, source hashes, code commit and model bytes. Old artifacts
are not overwritten. All per-step layouts/answers are in JSON; scores/answers
are also in `per_sample.csv`. This small pilot cannot establish equivalence.
