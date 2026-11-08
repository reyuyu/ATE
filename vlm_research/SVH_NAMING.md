# LLaVA SVH variants — canonical names

> Canonical naming effective 2026-11-08. **This is a naming-only change**, not a new algorithm, checkpoint, protocol, or benchmark result.

## Official labels

| Canonical name | Historical equivalent | Visual Hidden removed before | Persistent original-image KV in answer generation |
| --- | --- | --- | --- |
| **`llava-SVH`** | `C3-V2`, `Physical-C3` | L20 (0-based) | No layers (0 visual K/V) |
| **`llava-SVH-R`** | `C3-V2+L11-L24`, `Physical-L11-L24` | L25 (0-based) | L11–L24 only (14 of 32 layers) |

Model family: **`llava-hf/llava-1.5-7b-hf`**, pinned revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` when reproducing the frozen experiments. Both variants were evaluated with FP16 / eager attention / greedy decoding. **These names refer specifically to the LLaVA implementations**, not similarly named cross-model explorations.

### `llava-SVH` (formerly C3-V2)

- Visual Writing: non-answer text receiver queries may attend original image keys through L0–L19.
- Before L20: all image **hidden-state rows are physically removed**. L20–L31 run on text hidden states alone.
- During prefill, visual K/V may be computed transiently for the current attention operation, but **no original-image KV is persisted at any layer** for later answer generation.
- At P−1 and during autoregressive Decode, original image keys cannot be accessed in any layer.
- Answer-only attention is physically bypassed at L2–L5; norm, MLP, and residual paths remain. Original absolute RoPE positions are preserved.

### `llava-SVH-R` (formerly C3-V2 + L11–L24 re-access)

- Visual Writing for non-answer text receiver queries remains L0–L19.
- Visual hidden-state rows continue through L24 for evidence caching, then are physically removed before L25.
- The P−1 query and newly decoded answer queries may access original image keys **only at L11–L24**.
- Only L11–L24 retain original-image KV in persistent per-layer caches; remaining active layers use compact text-only KV (L2–L5 have no answer KV due to attention bypass).
- Decode processes one newly generated hidden token at a time. It **does not recompute 576 visual hidden rows**; re-access reads retained prefill image KV at the selected layers.
- L2–L5 answer-only attention bypass and absolute position semantics remain unchanged.

**Critical distinction:** physical image-Hidden exit and persistent image-KV residency are different lifetimes. Retaining L11–L24 image KV after L25 Hidden exit does not mean the image Hidden is recomputed in Decode.

## Compatibility and scientific provenance

- New manuscripts, plots, result-table row labels, and new experiment summaries: **use `llava-SVH` and `llava-SVH-R`**.
- Frozen source conditions, checkpoint keys, file names, existing report tables, results archives, manifest hashes, data bindings, and already published commit IDs: **do not rewrite**. They remain valid historical identifiers.
- Pure-Python alias conversion is in [`svh_names.py`](svh_names.py). Translate canonical labels to historical internal labels **at a new experiment's entry point** when using an unchanged frozen runner. Do not change internal condition strings or SHA-protected historical code.
- `C3+Middle9`, `C3+Middle10`, dense-mask controls, and `C3-V2-ZeroRef` are **not** synonyms for `llava-SVH-R`.
- For cross-paper accuracy comparisons, state the input protocol. The older GQA scores used **`GQA-legacy-HF-crop`**; later evaluations may use **`GQA-official-aligned-vicuna-pad`**. Never combine Native from one protocol with SVH/SVH-R from the other when calculating accuracy gaps.

## Historical evidence (not new measurements)

- [Physical C3-V2 five-benchmark results](https://github.com/reyuyu/ATE/commit/2c5d1820679b5eaad844e06db9cf46dcad23316f)
- [Physical C3-V2 + L11–L24 seven-benchmark results](https://github.com/reyuyu/ATE/commit/0414e553ba6fd1e21626ecf741489d225b44279c)
- [GQA protocol audit](https://github.com/reyuyu/ATE/commit/25989d8c0e00cf53a980c726be94d84feaabbc14)

The table above establishes a **canonical display and reporting name**, not evidence of newly run evaluations.
