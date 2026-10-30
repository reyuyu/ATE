from pathlib import Path

from paper_rq1_single_layer_gqa6.protocol import MODEL_PATH, GPUS, SEED, TASKS

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/paper_rq1_single_layer_gqa6'
OUTPUT = ROOT/'results/gqa6_update_decomposition'
RESOURCES = Path('/data/lk/ATE-resources/gqa6_update_decomposition')
BASE = '221cdde56dde53908aaa953fe26672fd33334f84'
NAMES = ('h_in', 'attention', 'h_mid', 'mlp', 'h_out')
REGIONS = dict(early=list(range(5)), middle=list(range(12,19)),
               peak=list(range(14,17)), deep=list(range(25,32)))
METRICS = ('norm_residual','norm_attention','norm_mlp','norm_output','norm_mid',
    'attention_ratio','mlp_ratio','net_update_ratio','cancellation_ratio',
    'cos_attention_residual','cos_mlp_residual','cos_attention_mlp','cos_input_output',
    'actual_net_update_ratio')
PROTOCOL = dict(base=BASE, seed=SEED, samples_per_task=100, tasks=list(TASKS),
    selection='Sort each original frozen500 by SHA256([seed,native-updates-A-v1,task,question_id,image_sha256,prompt_ids]); first100. No output-dependent selection.',
    sharding='Reuse original image-bound physical GPU rank; same image always one GPU.',
    intervention='NONE. Native prompt only, no Decode, cache or ablation. One forward/question in formal collection.',
    capture=list(NAMES), query='Actual last valid Prompt query P-1; preserve original input mapping.',
    arithmetic='Capture actual FP16 vectors at five hooks; all norms, dot products, sums for statistics and cosines in FP32 on CPU.',
    denominator_floor=1e-12, cosine_roundoff_bound=4*2**-23,
    residual_check='Actual h_mid must equal FP16-round(h_in+A) bit-exactly, and h_out=FP16-round(h_mid+M). Also record FP32 sum errors with fixed FP16 eps=2^-10 and min-subnormal=2^-24 bound.',
    fp16_sum_rtol=2**-10, fp16_sum_atol=2**-24,
    native_hook_logits='bit-exact same-run Smoke before/observed/after',
    historical_logits_atol=.02, historical_logits_rtol=.002,
    historical='Strict task/question/layer join of old BOS and Delta LogP CSVs; retain old500 curve separately from matched100.',
    bootstrap=dict(repetitions=10000,seed=SEED,cluster='image_id',
        estimator='Question-weighted resampled whole-image sums/counts. Overview pools600 questions (equal100/task at observed cohort), retaining cross-task shared images.',
        interval='Ordinary paired95% CI, descriptive; no multiplicity-adjusted significance, equivalence or causal claims.'),
    regions=REGIONS, planned_contrasts=['early Attention ratio minus middle',
        'early Attention norm minus middle','early MLP ratio minus Attention ratio',
        'peak14-16 Attention ratio minus flank12-13/17-18',
        'deep Attention ratio minus middle','deep MLP ratio minus middle'],
    zero_policy='Reject near-zero h_in denominator; undefined zero-vector cosine/cancellation are null, not imputed0. Record valid counts.',
    raw='Actual five-vector FP16 arrays and Full FP32 vocabulary logits saved externally with SHA256; replay all measurements.')
