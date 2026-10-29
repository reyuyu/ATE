# L2-L4 Residual-only Pilot

Independent 64-question pilot; no training, layer search or speedup claim.
ChooseRel32 (left16/right16) is selected from frozen Phase3.2; ScienceQA16
and TextVQA16 from frozen Phase4, all by seed20261029, no prediction filtering.
ChooseRel retains its original raw-question chat prompt; other tasks retain
their Phase4 prompts and scorers. Full is rerun paired on the same GPU.

P1-Joint uses original Phase3.8 zero_query at both o_proj and MLP outputs of
zero-based L2/L3/L4, targeting only ORIGINAL prompt P-1 on every full-prefix
forward. Whole-Skip temporarily overrides only these three layer instances'
forward methods, returning the identical hidden-state input without invoking
attention, layer norms or MLP. All methods/hooks restore even on exceptions.
All32 decoder positions and text/RoPE positions remain present.

No KV cache. Recompute complete prefix for each greedy token, maximum32,
stop at EOS. Cache use is rejected. Smoke validates native Full generation,
bitwise no-op and independent computed-zero references, local edit scope,
causal prefix invariants and exact residual identity. CPU tests run first.
Freeze manifest/source and commit before GPU. Run only idle physical GPU0.

From vlm_research: python -m residual_only_pilot.run --stage freeze|run.
Run uses per-question JSON/NPZ checkpoints with binding and integrity checks.
Final summaries require all64 questions, unchanged model state, complete
192 condition rows and independently recomputed saved-logit LogP and scores.
Push source, JSON/CSV/logs/report; keep raw NPZ only on local disk.
