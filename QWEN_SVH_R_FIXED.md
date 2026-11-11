# qwen-SVH-R: frozen L19-L35

- Model: Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, BF16, vision/text Eager.
- P-1 and every actual Decode query use the same visual re-access set: 0-based L19-L35 inclusive (17 layers).
- R visual writing remains L0-L21; main-path visual Hidden exits before L22; answer Attention skip remains L1-L6. mRoPE, DeepStack, pixels and weights remain unchanged.
- Late-layer visual KV comes from the frozen causal-prefix sidecar, never another layer's KV.
- Protocol B: ATE unified HF, **not author-official reproduction**. ScienceQA/MMBench use frozen P2; other tasks keep their frozen Scheme-B input. POPE has 9000 questions. Total: 35,346.
- Layer selection used historical Discovery scores. Expanded evaluation is exploratory, not independent noninferiority confirmation.
- This delivery freezes the configuration and evaluates its **functional quality reference**. It does not establish physical speedup or selected-KV memory savings. The reference collects visual KV for all 30 active layers.
- Historical strict numerical status remains **FAIL_RETAINED**. No repeated Smoke or new full numerical PASS is claimed. All historical failures remain read-only.
- Eager OOM cases are retained explicitly, never silently removed from the full denominator. Common-subset scores are labeled separately; incomplete benchmarks are not COMPLETE full scores.
- No layer search, training, additional prompt search, image shrinking, runtime switching or performance benchmark is included.

Environment: `/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`.
Entry: `qwen_svh_r_full.py --freeze`, `qwen_svh_r_full.py --launch`, `summarize_qwen_svh_r.py`.
Only physical GPU1/GPU3 are used, identified by frozen UUID. Original main and historical worktrees remain unchanged.
