from pathlib import Path

from benchmarks.protocols import MODEL_PATH, MODEL_REVISION
from paper_rq1.protocol import GPUS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/paper_rq1_single_layer_gqa6'
RESOURCES = Path('/data/lk/ATE-resources/paper_rq1_single_layer_gqa6')
ANNOTATION = Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_val_balanced.parquet')
IMAGES = Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')
SEED = 20261030
TASKS = {
    'ChooseAttr': ('chooseAttr',),
    'ChooseCat': ('categoryThatChoose', 'categoryThisChoose', 'objThisChoose'),
    'ChooseRel': ('relChooser',),
    'CompareAttr': ('twoCommon',),
    'LogicalObj': ('existThatOr',),
    'QueryAttr': ('positionQuery',),
}
PROTOCOL = dict(seed=SEED, samples_per_task=500, generation_subset_per_task=80,
    tasks={k:list(v) for k,v in TASKS.items()}, model_path=MODEL_PATH, model_revision=MODEL_REVISION,
    dtype='float16', attention='eager', transformers='4.49.0', layers=list(range(32)),
    selection='Unfiltered official GQA Balanced validation, exact types.detailed; hash(seed,task,qid). '
              'Select500 before any model output, then independently hash(seed,accuracy,task,qid) for80. '
              'Never substitute missing/corrupt selected images. Record shared image IDs across tasks.',
    sharding='All questions for the same image ID use hash(seed,worker,image_id) modulo4. '
             'Report actual per-task shard sizes; do not move questions based on results.',
    prompt='Unchanged pinned HF chat template; question plus existing single-word-or-phrase instruction.',
    ablation='One single layer0..31, original prompt last valid Query P-1: zero only self_attn.o_proj '
             'output. Native MLP/LayerNorm/residual, other queries/layers, inputs and weights unchanged. '
             'Independent identical prompt forward for every condition. Not source-target edge knockout.',
    first_token='FP32 CPU log_softmax over full vocabulary. Always track the native Full argmax y_full. '
                'delta_logp is primary; relative_delta_p=100*expm1(delta_logp); JS is natural-log nats. '
                'Store all33 FP32 vocabulary logit vectors externally for complete numerical replay.',
    correctness_groups='Gold-first-token matching is not full-answer accuracy. Actual Full-correct/wrong '
                       'groups are available only for the predefined80-question generation subset.',
    bos='Discover unique BOS by tokenizer.bos_token_id in actual valid input IDs, not index0. '
        'Native post-softmax P-1 row, mean over heads; store every head, row-sum error and valid keys. '
        'Release full attention matrices immediately after reading current row.',
    generation=dict(do_sample=False, num_beams=1, max_new_tokens=32, eos_token_id=2, use_cache=True,
        ablation='Prefill original P-1 only, once. All later Decode native, own naturally intervened cache.'),
    scorer='Official GQA primary equality after pinned LLaVA submission converter (rstrip period/lowercase); '
           'verified official source expressions and source hashes. Not a claim of full GQA auxiliary suite.',
    bootstrap=dict(repetitions=10000, seed=SEED, cluster='image_id',
        estimator='Paired question-weighted ratio of resampled whole-image cluster sums/counts; unequal cluster sizes retained.',
        interval='Ordinary percentile95% CI, descriptive only.192 primary task-layer comparisons; '
                 'no multiple-comparison significance or noninferiority claim.80 subset is exploratory, not independent.'),
    tolerances=dict(row_sum_atol=.002, noop_logits='bit-exact',
        cross_gpu_logits_atol=.02, cross_gpu_logits_rtol=.002,
        cross_gpu_bos_atol=.0005, cross_gpu_bos_rtol=.005,
        raw_metric_replay_atol=2e-6, raw_relative_replay_atol=.002,
        raw_metric_replay_rtol=2e-5),
    interpretation='Prediction sensitivity, Top-1 flips, JS and full-generation accuracy are different endpoints. '
                   'BOS correlations descriptive, not causal; no computation skipping or speedup claim.')
