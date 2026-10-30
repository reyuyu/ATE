from pathlib import Path
from paper_rq1_token_role_decode.protocol import MODEL_PATH, MODEL_REVISION, PYTHON, GPUS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper_rq1_function_attention_sparsify'
RAW = Path('/data/lk/ATE-resources/function_attention_sparsify')
OLD = [ROOT/'results/paper_rq1_token_role_decode', ROOT/'results/paper_rq1_late_attention_sources']
LAYERS = [14, 26, 31]
MODES = ['No-Image', 'BOS+History', 'History-only']
CONDITIONS = [(a,m) for a in ['A1','A2'] for m in MODES] + [('zero','Whole-off')]
SEED = 20261030
PROTOCOL = dict(version=1, seed=SEED, base_commit='6964407e6b4c1291cb0ca7c41128947d3025520d',
    model=MODEL_PATH, revision=MODEL_REVISION, dtype='float16', attention='eager', transformers='4.49.0',
    layers=LAYERS, conditions=CONDITIONS, gpus=[1,2,3], train_images=300, calibration_images=100,
    holdout_images=200, max_new_tokens=64, eos=2, prompt='Describe the image in one sentence.',
    source_partition='Original BOS, image576, all remaining prompt text (including separators), consumed generated history. Exhaustive/disjoint.',
    sampling='Reuse all400 images and frozen selected positions from both earlier studies. Input-hash image split300/100; independent new200 holdout excludes both studies/smokes and earlier COCO Discovery, by ID/bytes/RGB.',
    a1='Native full softmax; subtract bias-free Wo of jointly removed PV, FP32 accumulation/subtraction then one FP16 cast. No speed claim.',
    a2='Slice unrepeated cached K/V before QK and AV. Native FP16 QK, FP32 softmax cast FP16, native FP16 AV/o_proj. Keep positions/RoPE/cache unchanged.',
    risk_gate=dict(function_l31_bos_history_max_mean_loss=.10, max_p95_loss=.50, max_flip_rate=.10,
                   meaning='Conservative exploratory development gate, NOT evidence of safety. Failing any stops dynamic C3/C4 development.'),
    classifier=dict(features='Prefix last3 token hashed categorical IDs, position/64 and log1p(position), first/last-token indicators; hidden raw h31_in/its norm plus norm,position. No final generation length, next ID/logits/output or true attention mass.',
        algorithms=['A-prefix logistic','B-hidden linear logistic','C-prefix+hidden logistic'],
        training='CPU FP32 full-batch Adam300 steps lr.03 weight_decay.01, train-only normalization; no model weight training.',
        threshold_grid=[.80,.85,.90,.925,.95,.975,.99,.995,.999,1.000001],
        calibration_gate='Among thresholds with image-bootstrap95% precision lower>=.95, >=50 predicted tokens spanning>=20 images, choose max coverage. Select simplest A then B then C unless next improves recall by>=.05 absolute. If none, abstain and stop dynamic strategies.',
        labels='All t>=1 native positions. Function only complete single-token frozen Function lexicon; EOS/Special, ambiguous, unknown, unaligned, multi-token, punctuation remain explicit negative categories.'),
    stage_c=['S0-Full','S1-Static-No-Image','S2-Static-BOS+History','S3-Predicted-BOS+History','S4-Predicted-No-Image'],
    quality=dict(CIDEr_raw_noninferiority_margin=.02, SPICE_noninferiority_margin=.01,
        ci='10000 paired whole-image bootstrap, ordinary95%. CIDEr fixed full-cohort reference IDF. CI containing0 is not equivalence;200 images cannot establish rare-harm safety.'),
    latency='Same GPU, warmup3, repeated5 runs on fixed first10 hash-ordered holdout images; CUDA events with synchronization, paired random strategy order seed. Separate sparse vs A1 mechanism only. Full K/V storage and cache update retained; gather and reduced QK/AV reads reported. No theoretical FLOPs as measured speedup.',
    limitations='StageA/B development exploratory; only locked new200 is held out. Attention sparsity is not whole-layer skipping. Negative results retained.')
