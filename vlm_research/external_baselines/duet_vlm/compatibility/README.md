# Frozen HF 4.49 compatibility

Official unchanged AST CLIP Attention/EncoderLayer/local clustering/pdrop_rank_drop; dominant300 incl CLS + contextual7, cluster_width4; 307 -> 153 entering L16 -> 0 entering L24. Current source ranks ALL post-image text (C+all), not paper main C last-token ranking. Average191.75; no trained weights.

Official source is unchanged. Independent adapter uses AST-loaded original core functions. See recipe.json and the unified result protocol. No trained weights or quality benchmark generated.
