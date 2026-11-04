from pathlib import Path
import subprocess

from ..physical_visual_handoff.common import (ROOT, HISTORY, TOLERANCES, MODELS, key, read, atomic, digest, sha, write_csv)

OLD = ROOT / 'vlm_research/results/physical_visual_handoff_pilot'
OUT = ROOT / 'vlm_research/results/physical_visual_handoff_numerical_debug'
BASE = 'e4272ec80fbf54579d4dc10d1d057a5302aee0b4'
BRANCH = 'codex/physical-visual-handoff-numerical-debug'


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare():
    previous=read(OUT/'protocol.json') if (OUT/'protocol.json').exists() else None
    if previous is not None:
        atomic(OUT/('protocol_attempt_'+digest(previous)[:12]+'.json'),previous)
    pool = read(HISTORY/'llava/sample_manifest.json')['samples']
    old = read(OLD/'llava/sample_manifest.json')
    old_images = {s['image_sha256'] for s in old['samples']+old['smoke']}
    new = []
    used = set(old_images)
    quotas = dict(ChooseAttr=2,ChooseCat=2,ChooseRel=2,CompareAttr=2,LogicalObj=1,QueryAttr=1)
    for task,count in quotas.items():
        ranked = sorted((s for s in pool if s['task']==task),key=lambda s:digest([20261104,'numerical-debug-regression',task,s['question_id']]))
        for s in ranked:
            if s['image_sha256'] not in used:
                new.append(s);used.add(s['image_sha256']);count-=1
            if count==0:break
        assert count==0
    fail = next(s for s in pool if s['question_id']=='03567478')
    historical43 = [next(s for s in pool if key(s)==p.name) for p in sorted((OLD/'llava/checkpoints').iterdir()) if p.is_dir()]
    assert len(historical43)==43 and len(new)==10
    for kind in MODELS:
        smoke = list({key(s):s for s in ([fail] if kind=='llava' else [])+old['smoke']}.values())
        samples = list({key(s):s for s in smoke+new+(historical43 if kind=='llava' else [])}.values())
        specs = read(HISTORY/kind/'input_bindings.json')
        atomic(OUT/kind/'manifest.json',dict(smoke=smoke,new10=new,historical43=historical43 if kind=='llava' else [],
               selection='Input-independent stable hash; new10 unique images disjoint from entire previous pilot and smoke set',
               all_samples=samples))
        atomic(OUT/kind/'input_bindings.json',{key(s):specs[key(s)] for s in samples})
    atomic(OUT/'protocol.json',dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           sources=sources(),tolerances=TOLERANCES,old_partial_unchanged=True,cutoffs=dict(llava=20,qwen=22),
           repeated_failure_runs=3,regression_new_questions=10,no_60_or_240_matrix=True,
           stages=['original deterministic repeat','same-input operator trace','local FP32 arithmetic control',
                   'fixed numerical repair candidates','same-kernel paired regression','KV audit','diagnostic benchmark'],
           kernel_candidate_order=['native','fp32_linears','fp32_linears_attention','canonical_tiles'],
           previous_protocol_sha256=digest(previous) if previous else None,
           reuse_diagnosis_if_source_exact=previous['sources'].get('vlm_research/physical_handoff_debug/diagnose.py') if previous else None,
           selection_rule='Choose first candidate passing unchanged numerical and greedy gates on frozen smoke; report all failures',
           no_P4_or_question_only=True,comparison='Only common prefixes are numerical pairs; later branch differences descriptive only',
           benchmark=dict(warmups=2,repeats=5,fixed_generated_lengths=[16,64],batch=1,
                          classification_if_failed='UNVALIDATED_PERFORMANCE',compare_matched_kernels=True),
           output_root=str(OUT)))


def verify():
    p=read(OUT/'protocol.json')
    assert p['sources']==sources(), 'Diagnostic source changed after freeze'
    return p
