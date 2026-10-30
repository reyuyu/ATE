from pathlib import Path
from benchmarks.protocols import MODEL_PATH, MODEL_REVISION
from paper_rq1.protocol import GPUS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper_rq1_token_role_decode'
OLD = ROOT / 'results/paper_rq1_attention_time'
RAW = Path('/data/lk/ATE-resources/token_role_decode/raw')
PYTHON = '/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
SEED = 20261030
LAYERS = [0, 3, 5, 9, 14, 15, 16, 26, 31]
TYPES = ['Function', 'Visual', 'Relation']
MODES = ['Whole', 'Image']
PROTOCOL = dict(
    version=1, seed=SEED, model_revision=MODEL_REVISION, model_path=MODEL_PATH,
    model=dict(dtype='float16', attention='eager', transformers='4.49.0', train=False),
    prompt='Describe the image in one sentence.', images=200,
    generation=dict(do_sample=False, use_cache=True, max_new_tokens=64, eos_token_id=2),
    layers=LAYERS, modes=MODES, all32=True, focus='t>=1 only, original Full next token',
    sampling='Select input-hash ordered original frozen COCO confirmation images with no prior Discovery checkpoint; verify official annotation, byte SHA256 and RGB. One image/SHA. Exactly50 per GPU by sorted selection index modulo4. Smoke image disjoint.',
    alignment='Fast tokenizer decoded-caption roundtrip IDs must exactly equal native generated non-special IDs. Offset overlap with ASCII lexical word spans; main only one original token exactly covering a complete unambiguous lexicon word after trimming whitespace. Multi-token/unknown/punctuation/special/EOS retained in audit, never tails as words.',
    labels='Frozen offline explicit lexicons v1. Relation/action takes priority over Function then Visual; on/in/at/with/over/under are Relation. Ambiguous lexical homographs excluded. Visual words are candidates, not verified grounding. No external generative annotator.',
    position_selection='Per image input-hash anchor in relative position[.1,.9]. Within each available class choose nearest generation position to same anchor, ties fixed hash. Never use confidence, entropy or interventions. At most one/class/image, no replacement. Freeze all positions before formal ablation.',
    intervention='Whole zeros only current query o_proj at one layer. Image subtracts bias-free Wo applied to original post-softmax image weights times original V, accumulated/projected FP32 and subtract/cast once to native FP16; never renormalize. All32 zeros only current query outputs in all32 layers. MLP/residual/weights untouched; no free continuation.',
    cache='Replay entire native Full track. t consumes generated[t-1] at original absolute position. Every condition deep clones identical parent history. Per-call metadata/version/no-alias checks; parent SHA256 before/after each target. All replay logits must equal saved native FP32 logits bit-exactly. Throw away intervened caches.',
    numerical='No-op/observer logits bit-exact; eager row sum tolerance inherited .002. Decomposition bound from FP16 rounding u=2^-11 and FP32 dot gamma_n=n*2^-24/(1-n*2^-24), including subnormal half-ULP2^-25. Head and projection checks both required, no data-fitted atol. TF32 off. FP32 finite logits/logsoftmax/JS; JS clamp only within existing1e-6 round bound.',
    analysis='Primary Visual-Function Whole mean across nine layers, same-image paired; also per-layer and Image. Image-cluster bootstrap10000, ordinary95% plus Bonferroni family22 (four global raw/adjusted contrasts plus18 mode-layer contrasts). Never heads/layers as independent images. Relation secondary descriptive.',
    controls=['full_token_logp', 'full_token_entropy', 'generation_step', 'relative_generation_position', 'token_length'],
    controlled_model='Within-image fixed-effect OLS on per-position nine-layer mean, role dummies Visual/Relation, centered/scaled controls; drop only constant columns (token_length=1 by design). Image-pairs cluster bootstrap weighted sufficient statistics. Report rank/condition, fixed rcond1e-10; no causal role claim.',
    severe='All32 logP loss>=1 nat (>=63.2% Full-token relative probability loss). Also report>.1 and>.5, flips and all tails.',
    near_unchanged='Observed All32 abs_delta_logp<=.01 AND no Top1Flip AND JS<=1e-4. Descriptive examples, not safety certification.',
    limits='Native-generated-token conditional mechanism study; no final answer accuracy, grounding guarantee, train, layer search, freely intervened generation, kernel skipping or acceleration claim.')
