# Development Diagnostics

Two pre-formal validation issues were corrected. Neither altered model weights,
the answer-only algorithm, sample selection, or historical numerical tolerances.

1. Initial CPU fixture: the tiny eight-layer model uses a physical cutoff of L6,
   unlike real LLaVA's L20. Observing its L6 input and indexing original image
   positions caused `IndexError: index 10 is out of bounds ... size 10`. The
   fixture now observes the L5 output, before that test-specific compaction.
   The real GPU audit still observes the real L6 input, before L20 compaction.
2. Initial GPU smoke: both P-minus-1 Attention updates were exactly zero, but
   the generic cosine metric returned distance 1 for two zero vectors. The
   audit now requires exact zero for these deliberately zeroed updates and
   does not define a cosine for them. All nonzero hidden-state comparisons
   retain the unchanged historical gates. Original traceback, log, and frozen
   protocol are retained in `failed_smoke_1/`.

The final frozen source passed nine CPU tests and all three GPU smoke inputs
before any formal quality or timing matrix started. Formal execution had no
exceptions, OOMs, NaNs, truncations, or empty answers.
