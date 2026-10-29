"""Predeclared RQ1 protocol; no confirmation outcomes enter selection."""
from benchmarks.protocols import MODEL_PATH, MODEL_REVISION

SEED = 20261029
COUNTS = {'gqa_balanced': (300, 1000), 'scienceqa_img': (300, 800),
          'pope_adversarial': (300, 900), 'textvqa': (300, 1000),
          'coco_caption': (400, 1000), 'textcaps': (400, 1000)}
TIMES = (0, 1, 4, 8, 16)
GPUS = ('GPU-a63c6362-dcfd-4a26-c44b-014b40127c89', 'GPU-b08489af-0a8a-eb48-55ba-8346733af417',
        'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185')
PROTOCOL = dict(
    version=1, seed=SEED, model=MODEL_PATH, model_revision=MODEL_REVISION,
    dtype='float16', attention='eager', transformers='4.49.0', train=False,
    target_counts={k: dict(discovery=a, confirmation=b) for k, (a, b) in COUNTS.items()},
    generation=dict(do_sample=False, num_beams=1, short_max_new_tokens=32, caption_max_new_tokens=64,
                    eos_token_id=2, use_cache=True),
    intervention='Zero only current Query o_proj output in zero-based three-layer window, MLP unchanged. '
                 't0=current original prompt P-1. t>0=single actual Decode Query consuming Full token t-1. '
                 'Discovery each intervention starts with a deep clone of identical unmodified Full history; '
                 'do not free-generate after that intervention. Confirmation maintains its own naturally '
                 'intervened cache and generated history; Prefill-only modifies only original P-1, '
                 'Decode-only only subsequent single-token forwards, Combined both.',
    windows=list(range(30)), decode_times=list(TIMES),
    window_selection='After complete Discovery, average JS within dataset x reached-time cells, then '
                     'equal-weight available cells. Fixed references starts0,9,20,29; add lowest2 remaining '
                     'starts, then highest2 remaining. Ties smaller start. Freeze eight starts before '
                     'reading or generating Confirmation. Missing cells reported, no imputation. '
                     'Do not select using confirmation or generation accuracy.',
    splitting='Global source-aware image identity (COCO/OpenImages/GQA identities), byte SHA256 and '
              'decoded RGB SHA256. One canonical image assigned to exactly one dataset and partition. '
              'Fixed hash order seed20261029, labels and model outputs not used. Multiple questions '
              'within one partition allowed; same image always same worker. No repeated question.',
    short_prompt='Reuse frozen Phase4.0 short-answer and ScienceQA multiple-choice format; '
                 'GQA official balanced questions, never correct_question subsets.',
    caption_prompt='Describe the image in one sentence.',
    metrics={'gqa_balanced': 'Official GQA generated answer accuracy (verified official evaluator)',
             'scienceqa_img': 'Pinned original option-letter parser/index accuracy',
             'pope_adversarial': 'Pinned official lexical accuracy, F1/precision/recall/yes-ratio secondary',
             'textvqa': 'Pinned10-human-answer VQA soft accuracy, image-only prompt',
             'coco_caption': 'pycocoevalcap1.2 PTBTokenizer, CIDEr primary, SPICE secondary, official refs',
             'textcaps': 'pycocoevalcap1.2 PTBTokenizer, CIDEr primary, SPICE secondary, official refs'},
    inference='Image-cluster paired bootstrap50000 replicates, fixed seed; marginal95%CI for all deltas. '
              'Familywise two-sided95% bounds use Bonferroni across144 primary comparisons '
              '(6 tasks x8 windows x3 bypass modes);48 caption SPICE comparisons separate secondary family. '
              'Low-harm claim requires familywise lower bound > negative predeclared margin. '
              'CI crossing0 is not equivalence. No heads/layers/timepoints treated as independent images.',
    noninferiority_margins={'short_accuracy_or_soft_0_to_1': .01, 'CIDEr_raw_0_to_10': .02, 'SPICE_0_to_1': .01},
    zero_effect_guard='When every paired score delta is exactly zero, percentile bootstrap is degenerate. '
                      'Also require score_range * (1-alpha**(1/n_images)) < margin, where alpha=.05/(2*family). '
                      'This exact zero-negative-event upper bound prevents low-harm claims with inadequate '
                      'image-cluster rare-harm resolution. Score range10 for CIDEr,1 otherwise. '
                      'Report empirical zero changes and underpowered inference separately.',
    caption_bootstrap='Use paired per-image official scorer outputs with fixed complete-cohort CIDEr '
                      'document-frequency calibration. This estimates image-average score uncertainty '
                      'conditional on fixed benchmark IDF, not a refitted-IDF estimand; disclose explicitly.',
    mechanism='Discovery Native Full only:32-layer current-query BOS/Image/original Question/generated-text '
              'mass and o_proj-output / layer-input residual norm. History includes current consumed '
              'generated token at Decode. Group windows by mean of three layers. Within task x time x '
              'window centered, image-cluster descriptive regression adjusting Full confidence and layer '
              'fixed effects; no claim that across-layer correlation explains causality. No Sink/ROI.',
    completion_gate='All sources, official scorers and image-disjoint manifests verified before formal GPU work. '
                    'CPU tests + real cache/bypass smoke must pass. Partial availability or failed smoke '
                    'is explicitly partial; never manufacture scores/figures or silently substitute sources.')


def select_windows(cell_mean_js):
    """Input must contain30 finite, equally weighted Discovery cell means."""
    import math
    if set(cell_mean_js) != set(range(30)) or not all(math.isfinite(x) and x >= 0 for x in cell_mean_js.values()):
        raise ValueError('Require complete30-window Discovery')
    chosen = [0, 9, 20, 29]
    low = sorted((s for s in range(30) if s not in chosen), key=lambda s: (cell_mean_js[s], s))[:2]
    chosen.extend(low)
    high = sorted((s for s in range(30) if s not in chosen), key=lambda s: (-cell_mean_js[s], s))[:2]
    return sorted(chosen+high)
