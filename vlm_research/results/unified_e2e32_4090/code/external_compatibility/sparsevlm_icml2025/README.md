# Frozen HF 4.49 compatibility

ICML v1.5 nominal192 schedule [300,200,110], prune AFTER zero-based L2/L6/L15, official recycling adds 9/4/3 clustered tokens: real visual layouts 576 (L0..2),309 (L3..6),204 (L7..15),113 (L16..31), mean206.5. Official scaled_dot_product_attention is actually explicit eager QK, FP16 softmax, AV; this exact function and original ranking/pruning/merging AST are called unchanged. Bridge bypasses broken old sdpa-only class wiring and deprecated APIs without using fused SDPA/Flash or changing core math. Compact RoPE and per-layer real cache Decode.

Official source is unchanged. Independent adapter uses AST-loaded original core functions. See recipe.json and the unified result protocol. No trained weights or quality benchmark generated.
