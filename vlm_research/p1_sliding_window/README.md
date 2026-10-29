# GPU0 P-1 Three-Layer Sliding Window

Reuse the exact64 residual-only pilot questions, no sampling or decoding.
Freeze 30 windows L0-L2 through L29-L31, stride1, with A/M/Joint clone-zero
updates only at the original Prompt P-1. Keep all other token computations.
No KV cache, training, physical skipping or speedup claims.

Closed candidates all have one token. Keep the legacy first-candidate N+1
teacher-forced layout; after exact causal smoke verification reuse its entire
vocabulary distribution for complete LogP of all one-token candidates. Reject
multi-token candidates rather than truncating. TextVQA fixes Full first argmax
and measures that token's LogP and whether the predicted first token changes,
not generated-answer accuracy. Same-GPU Full is recomputed for every question.

Hidden endpoints use layer residual outputs before final RMSNorm; separately
save normalized final hidden state. Compare each to Full at the same depth.
End-to-final drift decay is not evidence of recovered information by itself.

Only nine focused CPU tests and one real smoke question per dataset. Verify
legacy L2-L4 A/M/Joint and historical Full/Joint exactly, no tolerance changes.
Do not run heavyweight full-weight hashing or extra GPU mechanism audits.

From vlm_research: python -m p1_sliding_window.run --stage freeze|run|analyze.
Freeze/commit code first. GPU0 preflight, per-window atomic checkpoints, source
and parent-manifest binding, parameter-version/hook restoration checks.
Output results/p1_sliding_window_gpu0: raw per-question JSON/NPZ, flat CSV,
all-window and top/bottom summaries, three PNG curves and report.md.
Push only the independent branch, keep raw NPZ local, never modify main.
