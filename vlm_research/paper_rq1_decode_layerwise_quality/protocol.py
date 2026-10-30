from pathlib import Path
from benchmarks.protocols import MODEL_PATH, MODEL_REVISION
from paper_rq1_token_role_decode.protocol import PYTHON
from paper_rq1.protocol import GPUS as ALL_GPUS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/paper_rq1_decode_layerwise_quality'
RAW = Path('/data/lk/ATE-resources/decode_layerwise_quality')
GPUS = ALL_GPUS[1:]
SEED = 20261030
CONDITIONS = ['Full'] + [f'L{k}-off' for k in range(32)]
PROTOCOL = dict(version=1, seed=SEED, base_commit='a6bf615a6c515addd29fac508ea86ab8da12dc64',
    model_path=MODEL_PATH, model_revision=MODEL_REVISION, dtype='float16', attention='eager', transformers='4.49.0',
    gpus=list(GPUS), physical_gpus=[1,2,3], samples=dict(coco_caption=200,textvqa=200), conditions=CONDITIONS,
    intervention='Only the selected single layer current Decode query o_proj output is zero. MLP and all other layers normal. Prefill never edited; first generated token native. No speed claim.',
    generation=dict(do_sample=False, num_beams=1, use_cache=True, eos_token_id=2, coco_max_new_tokens=64, textvqa_max_new_tokens=32),
    prefill='Native prefill computed once/image; immutable parent cache deep-cloned separately for all33 conditions. Independent own-prefix Decode trajectories. Cached vs independent-prefill equivalence gated by Smoke.',
    stage_detection='Real cache_position >= full prompt length, nonempty past cache, one current query; never query length alone.',
    coco_prompt='Describe the image in one sentence.', textvqa_prompt='Existing official image-only TextVQA question + frozen short-answer instruction; no added OCR.',
    sampling='Input-SHA seeded ordering; one question/image; prior-used COCO IDs/byte/RGB excluded. TextVQA prefer locally available never-used identities, then explicit historical-overlap fallback only if insufficient. No model-output filtering.',
    scoring='COCO pycocoevalcap1.2 PTBTokenizer/CIDEr/SPICE, same200 reference images/IDF across33 conditions. TextVQA pinned official ten-answer soft score.',
    statistics=dict(bootstrap_repetitions=200000, seed=SEED, unit='Image-cluster paired bootstrap, fixed reference-IDF; resample per-image score deltas, not words/heads/conditions.',
        ordinary_ci=.95, familywise_ci=.95, comparison_family=96,
        correction='Bonferroni two-sided percentile intervals across32 layers x3 quality endpoints=96; alpha/192 in each tail;200000 draws gives~52 expected tail draws. Report Monte Carlo limitation; no claim of exact finite-sample coverage.',
        margins=dict(CIDEr=.02,SPICE=.01,TextVQA=.01), criterion='Lower confidence bound > -margin, ordinary and corrected separately. Cross-task candidate requires all3 corrected lower bounds; not safety certification.'),
    boundaries='Only32 individual Decode-layer interventions. No VLM/classifier training, source sparsity, multi-layer intervention, acceleration claims or confirmation expansion.')


def condition_layer(condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown frozen condition')
    return None if condition == 'Full' else CONDITIONS.index(condition)-1


def limit(sample):
    return 64 if sample['dataset']=='coco_caption' else 32
