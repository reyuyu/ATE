# Source Review and Frozen Scope

- Base: c67bc7d1a9d0603927b1f18b52777d790f4d1d85.
- Phase3.7: f1e661fd228ec600e5a63b3016d68bd2fde64940,
  phase3_7/pathways.py PathwayMask/observed_forward and protocol/run mapping.
- Original complete-answer baseline: e495adf4fd93337a3638981bbb66c2bbac81242d.
- Shared directed_suffix.py, knockout_probe.py and attention_position_probe.py
  are byte-identical between Phase3.7 and this base; hashes are in protocol.json.
  No Phase3.7 pilot or old 3,000-question Full generation is rerun.

## Reuse and Required Adaptation

The old PathwayMask targets both L0 and L1 and its parent rejects KV caches.
Importing it unmodified would not implement the requested single-layer cached
Prefill experiment. SourceMask therefore subclasses the same existing
DirectedSuffixKnockout and uses the unchanged knockout_mask primitive. Only its
single-layer and Prefill-versus-Decode dispatch is adapted in a new directory.
The existing cached generate/JointAttention, gqa_prediction, EarlyRows and
whole-image paired bootstrap remain unchanged.

The original sample mapping calls the text slot question_and_instruction_positions.
It includes the answer instruction. We reuse the strict template/offset/expanded-ID
mapper on that exact historical text slot, then select only the original Question
character span. The selected IDs must decode exactly to the actual Question;
boundary-crossing or special tokens are rejected. Remaining text is Template.
This avoids treating task instructions as Question source keys.

## Temporal and Statistical Boundaries

All eleven new conditions intervene only in original Prefill. Decode is normal,
using each condition's own actual cache and generated prefix. Question-to-Image
targets all real Question rows, not merely P-1. Historical P-1-plus-Decode OFF
first logits validate EarlyOFF's first step but its answers are never substituted.

Native Full complete answers are reused. One native cached Prefill per selected
question measures source Mass/Hidden and checks exact historical logits. New
generation observations are captured during their first actual generation
Forward, not through an extra run. Six outcome-blind cases retain small selected
row/vector arrays; full32-layer attention matrices are never saved.

Two Accuracy interactions are a predeclared family: D_visual and D_question.
Positive means increased L2 source dependence after EarlyOFF. Paired ordinary95
and Bonferroni family95 intervals use whole-image SHA256 resampling. Six A
contrasts remain exploratory, with their separate family-six intervals.
There are no A-by-L2 cross conditions, indirect-path composition/mediation tests,
new layer scans, training, or expansion beyond the frozen240 questions.

## Retained CPU Failure

The first cleanup test replaced self_attn.forward with a mock. The existing
signature guard correctly rejected it before the intended injected exception.
The failed CPU record is retained in attempts/01_cpu_injection_signature_guard,
and the traceback is under failures/. The cleanup test now injects at q_proj;
a separate test asserts the original forward-signature guard remains enabled.
No acceptance tolerance or model/mask behavior was relaxed.
