# Authorized Continuation

- User requested continuation after the previous delivery. Resume check: 2026-11-05 08:02:04 +08:00.
- The original start remains 04:31:15; candidate cutoff 08:31:15, GPU cutoff 10:01:15, delivery cutoff 10:31:15. No clock is reset.
- Previous delivered commit: `60d262f83a981e49a614ef8213e5311c3c7c6469`. Its worktree, results and main are untouched.
- New branch/worktree: `codex/ate-accuracy-autoresearch-6h-continued`, `/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h-continued`.

## Hypothesis Before New Generation

The prior DEV/CHECK results showed a small L24 All-R/Decode-only recovery versus L20. Four bounded extensions test whether L28 or L32 can recover more quality, crossed with no answer bypass and Decode-only bypass. P-1 text Attention is retained in all four. Image and all pre-P-1 text Attention updates remain normal. P-1 and every real Decode remain image-blind.

L32 is a writing-capacity diagnostic: it does not remove visual hidden rows inside the 32 decoder layers. It still uses genuine text-only persistent KV. It must not be described as an accelerated physical-exit winner merely because its quality is higher. Actual paired timings and the quality shortfall must both be reported.

These are four extension configurations, but L32/no-bypass was already a coarse diagnostic. Its old coarse records are reused and only its coverage is expanded. Thus there are three newly distinct candidates, fifteen in total, not sixteen independent candidates. No new token/length rules, prompts, models, training, kernels or old I+A bypass are introduced. GPU2/3 were both idle at authorization and startup; GPU0/1 remain external jobs.

## Selection and Statistical Boundary

The original DEV/CHECK image clusters are reused without changing membership. Only their outputs enter promotion and selection. The previous FINAL has already been inspected; a new evaluation on those images is named RECHECK, not a fresh independent FINAL. Any intervals are descriptive exploratory paired image-cluster intervals, not a new confirmatory noninferiority claim.

Top two DEV configurations enter CHECK with the fixed prior L24 All-R/Decode-only reference. Worst-dataset loss is primary; within 0.25pp, use measured same-complete-sequence minimum E2E speed. Native, C1, C3 and original selected Q+ASSISTANT remain references, not post-hoc additions to the winner pool.

Old records are read in place only after payload, sample, configuration and original source hashes match; original-prompt token bindings and scores are checked. Model revision/precision/dependencies and physical GPU UUIDs match. New real Smoke also reproduces the historical same-GPU Native first logits/prefix exactly. No old latency substitutes for new timing. Shared model, images and historical arrays are not copied.

## Early Observation

The coarse VQAv2 quick60 gave 79.0% for both deeper no-bypass settings and 80.667% for both deeper Decode-only settings, versus Native85.5%. This is a negative result, not proof of equivalence or absence of benefit. All four continue through the frozen full DEV before promotion.
