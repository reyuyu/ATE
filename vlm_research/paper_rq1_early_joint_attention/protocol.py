from pathlib import Path
from benchmarks.protocols import MODEL_PATH, MODEL_REVISION
from paper_rq1.protocol import GPUS as ALL_GPUS
from paper_rq1_token_role_decode.protocol import PYTHON
from paper_rq1_single_layer_gqa6.protocol import TASKS, ANNOTATION, IMAGES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/paper_rq1_early_joint_attention'
RAW = Path('/data/lk/ATE-resources/early_joint_attention')
GPUS = ALL_GPUS[1:]
SEED = 20261031
CONDITIONS = {
    'S0': dict(name='Full', layers=[], mode='Full'),
    'S1': dict(name='L0–L4 P−1-only', layers=list(range(5)), mode='P-1-only'),
    'S2': dict(name='L0–L4 Prefill-all-only', layers=list(range(5)), mode='Prefill-all-only'),
    'S3': dict(name='L0–L4 Decode-only', layers=list(range(5)), mode='Decode-only'),
    'S4': dict(name='L0–L4 Combined', layers=list(range(5)), mode='Combined'),
    'S5': dict(name='L2–L5 P−1-only', layers=list(range(2,6)), mode='P-1-only'),
    'S6': dict(name='L2–L5 Prefill-all-only', layers=list(range(2,6)), mode='Prefill-all-only'),
    'S7': dict(name='L2–L5 Decode-only', layers=list(range(2,6)), mode='Decode-only'),
    'S8': dict(name='L2–L5 Combined', layers=list(range(2,6)), mode='Combined'),
    'S9': dict(name='L2–L4 Combined', layers=list(range(2,5)), mode='Combined'),
    'S10': dict(name='L0–L1 Combined', layers=[0,1], mode='Combined'),
}
PROTOCOL = dict(version=1, seed=SEED, base_commit='f469eedbc960a0a1cd73661723e70c1ff46a6b1f',
    model_path=MODEL_PATH, model_revision=MODEL_REVISION, dtype='float16', attention='eager', transformers='4.49.0',
    tasks={k:list(v) for k,v in TASKS.items()}, target_per_task=500, conditions=CONDITIONS,
    gpus=list(GPUS), physical_gpus=[1,2,3],
    sampling='Official GQA Balanced validation only. Exclude all historical manifest/config image IDs and SHA256; '
      'input-hash ranking within task, up to500 questions per task without replacement. Repeated images within/across '
      'current tasks allowed and clustered; no model-output selection. No fallback to exposed images. Separate smoke images.',
    generation=dict(do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2),
    intervention='Simultaneous self_attn.o_proj output zeros in frozen layer sets. Four explicit temporal scopes. '
      'MLP/LayerNorm/residual native. Every condition independently prefills and generates using its own cache/prefix. '
      'No cached Full state reused, no teacher forcing, no actual compute omission or acceleration claim.',
    phase='Use explicit cache_position, original prompt length, per-layer past length and current query length, not qlen alone.',
    scoring='Pinned official GQA primary exact answer accuracy after decoded text.strip() and LLaVA rstrip period/lower conversion.',
    statistics=dict(repetitions=200000,seed=SEED,unit='Paired image-cluster bootstrap, question-weighted sum/count ratio',
      main_family=8, auxiliary_family=2, taskwise_family=60, planned_contrast_family=6,
      correction='Two-sided Bonferroni percentile intervals separately for8 pooled primary S1–S8, '
        '2 pooled auxiliary S9/S10,60 taskwise endpoints (10x6, including unavailable tasks in planned family), '
        'and6 pooled prespecified direct contrasts. Ordinary95% also shown. No cross-family simultaneous claim.',
      pooled_ni_margin_pp=1.,taskwise_risk_margin_pp=3.,
      criterion='Corrected lower bound strictly greater than -1pp pooled. Subtask -3pp is a heterogeneity risk screen, not safety.',
      caveat='Finite Monte Carlo percentile approximation, not exact simultaneous finite-sample coverage. '
        'Zero discordances give degenerate bootstrap; do not interpret as zero population risk. '
        'Achieved unequal task counts yield question-weighted observed mixture, not target balanced3000 conclusion.'),
    optional_midlayer_reference=False,
    supplementary='Not run in this core delivery; optional cross-task risk check remains explicitly unexecuted.',
    boundaries='Only11 conditions on actually frozen eligible GQA questions. No training, layer search, source pruning, Decode time scan or caption task.')
