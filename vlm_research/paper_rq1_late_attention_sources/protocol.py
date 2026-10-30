from pathlib import Path
from paper_rq1_token_role_decode.protocol import MODEL_PATH, MODEL_REVISION, PYTHON, GPUS, SEED, TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper_rq1_late_attention_sources'
OLD = ROOT / 'results/paper_rq1_token_role_decode'
POOL = ROOT / 'results/paper_rq1_attention_time'
RAW = Path('/data/lk/ATE-resources/late_attention_sources/raw')
LAYERS = [3, 14, 16, 25, 26, 27, 28, 29, 30, 31]
DEEP = list(range(25, 32))
SPECIAL = [26, 31]
GROUPS = ['BOS', 'IMAGE', 'PROMPT_TEXT', 'GENERATED_HISTORY']
SOURCE = dict(Image='IMAGE', Prompt='PROMPT_TEXT', History='GENERATED_HISTORY',
              BOS='BOS', Recent='RECENT', Older='OLDER_HISTORY')
MODES = ['Whole', 'Image', 'Prompt', 'History']
FAMILY = 41
PROTOCOL = dict(version=1, seed=SEED, base_commit='2ffdfdf5c003d10d2a2b39c6bd2c48f00b90f861',
    model_revision=MODEL_REVISION, model_path=MODEL_PATH,
    model=dict(dtype='float16',attention='eager',transformers='4.49.0',train=False),
    images=200,prompt='Describe the image in one sentence.',
    generation=dict(do_sample=False,use_cache=True,max_new_tokens=64,eos_token_id=2),
    layers=LAYERS, special_layers=SPECIAL, groups=GROUPS, modes=MODES,
    sampling='Input-hash late-source-images-v1 ordering of original unexecuted COCO confirmation. Exclude all prior Observation2 sample/smoke IDs/SHA/RGB and prior COCO Discovery IDs/SHA/RGB. Unique ID,bytes SHA,decoded RGB. First200 formal plus disjoint201st smoke, selection-index modulo4 gives50/card.',
    targets='Reuse Observation2 labels.align/select unchanged: frozen lexicon/priority, complete single-token unambiguous words at t>=1, one per class/image, same hash anchor. No confidence or outcome selection. Full200 finishes before target freeze and all formal ablations.',
    partition='Dynamically locate unique original BOS,576 image positions,remaining original valid prompt positions. GENERATED_HISTORY=all generated positions through current consumed y[t-1]. RECENT=last key, OLDER_HISTORY=other generated keys. Valid visible keys exactly partitioned; padding rejected, no unassigned key.',
    intervention='One current Decode Query at one specified layer only. Whole zeros o_proj including bias. Source subtracts bias-free Wo(original native post-softmax group PV) accumulated FP32 then subtraction/nativeFP16 cast once. No mask, no renormalization, no free continuation. Every condition owns cloned Full history.',
    observation='One native observer forward per selected position for all10 layers; logits bit-exact to bareFull. Capture actual layer-input residual and native eager weights/repeated V/o_proj input/output. Four primary contributions sum to native Attention within analytic rounding bounds. Extra Recent/Older only26/31.',
    numerical='Retain original Observation2 image-vs-other decompose audit unchanged. General partition adds ONLY analytic gamma_g summation error for four FP32 vectors and group projections to frozen FP16/FP32 bounds, never fitted atol. Verify History=Recent+Older similarly. Noop logits/replay exact; finite FP32 full-vocab metrics; TF32off.',
    cache='Independent deep clones with no-alias/metadata checks every call; parent actual SHA256 before/after every target and every smoke condition. Consume y[t-1] to predict y[t]; cache length promptN+t-1, visibleN+t. Discard every altered cache. Replay every native generation step bit-exact.',
    statistics=dict(bootstrap=10000,unit='image SHA256, paired whole-image clusters',family=FAMILY,
        A='10 contrasts: Visual losses History/Image/Prompt at26/31 (6), same-token History-Image and History-Prompt at26/31 (4).',
        B='12 contrasts: same-image Visual-Function for Image/Prompt/History at26/31, raw6 and adjusted6.',
        C='13 contrasts: Visual Whole loss at each25..31 vszero7; adjacent paired differences six. Positive continuous region requires allseven corrected lower bounds>0. CI coveringzero is uncertainty, not equivalence or insensitivity.',
        D='6 supplementary contrasts: Recent-Older loss paired at26/31 for3roles, included in same41 family.',
        controls=['full_token_logp','full_token_entropy','generation_step','relative_generation_position','token_length'],
        adjusted='Within-image fixed effects separately per source/layer on images with Function+Visual; availableRelation as third role. Same linear controls as Observation2; constant token_length1 explicitly dropped. Rank/condition/bootstrap audit. Coefficients conditional, not causal word-role effects.',
        secondary='Remaining sources/layers, norms/mass/cosines, Relation vs Function and other comparisons descriptive, no posthoc significance claims.'),
    limits='Native-generated-token conditional mechanism study. Source removal is a local causal intervention on projected Value contribution, not proof of transferred visual information, additive LogP attribution, final answer accuracy, layer skipping or speedup.')


def conditions(layer):
    return MODES + (['BOS', 'Recent', 'Older'] if layer in SPECIAL else [])
