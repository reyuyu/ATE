from pathlib import Path
from paper_rq1_single_layer_gqa6.protocol import MODEL_PATH,MODEL_REVISION,GPUS,TASKS,SEED,ANNOTATION,IMAGES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/bos_cheap_discovery'
RAW=Path('/data/lk/ATE-resources/bos_cheap_discovery')
OLD=ROOT/'results/paper_rq1_single_layer_gqa6'
UPDATES=ROOT/'results/gqa6_update_decomposition'
EPS=(.01,.03,.05)
REGIONS=((0,1),(2,5),(6,13),(14,16),(17,24),(25,31))
HEAD_FEATURES=('bos_cos','bos_score','q_norm','k_norm','v_norm','v_mean','v_std','v_min','v_max','projected_v_norm','sample_margin','sample_score_mean')
INPUT_FEATURES=('residual_norm','residual_mean','residual_std','residual_absmax','length','log_length')
PROTOCOL=dict(seed=SEED,model=MODEL_REVISION,base='b91f4ae259ecf4ce1307a1abb2836d035e612077',dtype='float16',attention='eager',transformers='4.49.0',
 historical='Reuse all 96000 historical single-layer labels; exactly one new observed Full per old question.',
 target='Native Full top1 token delta LogP; dangerous=top1_flip OR -delta_logp>epsilon; not accuracy harm.',eps=EPS,
 splits='image SHA256 hash: <70% train, remainder calibration; cross-task same bytes stay together. All old data exploratory.',
 new='Up to100/task, unused image IDs AND SHA256; one question per new image globally. Shortfall is reported, never confirmation if600 unavailable.',
 collection='Four independent GPUs, original image-rank binding, no DDP. New-image features plus32 original single-layer ablations only after rule freeze.',
 cheap='Actual RoPE query and BOS K/V; independently reproduce using only selected normalized rows and projection weights, no full QK or Softmax.',
 sampled_keys='Exactly up to6 extra: image patches0,191,383,575 and first/last question+instruction positions; deduplicate; input-only.',
 cheap_tolerances=dict(atol=.02,rtol=.005),native_logits='same-run bit-exact',historical_tolerances=dict(atol=.02,rtol=.002),
 predictors=['Layer-only','Length-only','BOS-cos','BOS-score','BOS-QK+Layer','BOS-QK+Value','BOS-QK+Residual','Sampled6Keys','Global-noLayer','Early-separate','Region-grouped','Early-never-skip','Region-threshold','Layer-threshold','BOS-Mass-Oracle'],
 training='CPU FP64 standardized logistic regression, LBFGS max80,l2=1e-3, no class weighting; no LLaVA training. Platt calibration on old calibration only.',
 thresholds='Probability grid0,.001,.002,.005,.01,.02,.03,.05,.1,.2,.4,.6,.8,1; fixed BOS oracle .5,.6,.7,.75,.8,.85,.9,.95; scalar score/cos train quantiles .00:.05:1 in both directions.',
 selection='For each epsilon and predictor choose maximum calibration coverage with observed dangerous<=1%, then lower risk, then fixed threshold order; no selected conditions =>never skip. Calibration is not a safety guarantee.',
 uncertainty='2000 image-cluster percentile bootstrap, question/layer conditions not independent. New-image risk upper bound additionally uses simultaneous bounded-cluster Hoeffding numerator/denominator bounds; no <1% claim unless upper<.01.',
 joint='Not executed: optional joint-gating and generation deferred; single-layer labels are not multi-layer safety evidence.',
 timing='Separate CUDA-event GPU microbench and CPU timings; no end-to-end speedup claim; P-1 is one prefill row.',regions=REGIONS)
