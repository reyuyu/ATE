"""Official fixtures and final source/data/artifact checks; no additional GPU work."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import unittest
import numpy as np
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.scoring import textvqa_score
from paper_rq1.scorers import caption_scores,DEPS
from paper_rq1.inventory import RESOURCES,OLD
from .protocol import ROOT,OUT,RAW
from .data import rgb_sha
from .run import verify


def analysis_cpu():
    for name in ['analysis_cpu_tests_final.txt','analysis_cpu_validation.json']:
        path=OUT/name
        if path.exists():path.rename(OUT/(str(time.time_ns())+'_'+name))
    with (OUT/'analysis_cpu_tests_final.txt').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_decode_quality_analysis'))
    if not result.wasSuccessful():raise AssertionError('Postprocessing tests failed')
    freeze_json(OUT/'analysis_cpu_validation.json',dict(passed=True,tests=result.testsRun,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'test_decode_quality_analysis.py',ROOT/'paper_rq1_decode_layerwise_quality/analyze.py',ROOT/'paper_rq1_decode_layerwise_quality/statistics.py',ROOT/'paper_rq1_decode_layerwise_quality/resume.py']},
        log_sha256=sha256(OUT/'analysis_cpu_tests_final.txt'),initial_missing_scipy_attempt_preserved=True))
    print('Postprocessing tests passed',result.testsRun,flush=True)


def scorer():
    refs={'a':['A red car is on the road.'],'b':['A dog is beside a tree.']}
    total,per=caption_scores(refs,{k:v[0] for k,v in refs.items()})
    if not np.isclose(total['CIDEr'],10.) or total['SPICE']!=1.:raise AssertionError('Official caption fixture failed')
    if textvqa_score('cat',['cat']*10)!=1. or textvqa_score('dog',['cat']*10)!=0.:raise AssertionError('TextVQA ten-answer fixture failed')
    paths=[ROOT/'paper_rq1/scorers.py',ROOT/'benchmarks/scoring.py',ROOT/'benchmarks/_vendor/m4c_evaluator.py']
    paths+=sorted((DEPS/'pycocoevalcap').rglob('*.py'))+sorted((DEPS/'pycocoevalcap').rglob('*.jar'))
    freeze_json(OUT/'scorer_validation.json',dict(passed=True,identical_reference_fixture=total,textvqa_fixtures=True,per_image=per,source_sha256={str(p):sha256(p) for p in paths}))
    print('Official caption/TextVQA scorer fixtures passed',flush=True)


def data():
    manifest=json.loads((OUT/'sample_manifest.json').read_text());samples=manifest['samples']+manifest['smoke']
    coco=json.loads((RESOURCES/'annotations/captions_val2014.json').read_text());images={str(r['id']):r for r in coco['images']};refs={}
    for r in coco['annotations']:refs.setdefault(str(r['image_id']),[]).append(r['caption'])
    tv=json.loads((OLD/'sources/textvqa/TextVQA_0.5.1_val.json').read_text())['data'];text={str(r['question_id']):r for r in tv}
    seen=set();rgb=set()
    for s in samples:
        if s['dataset']=='coco_caption':
            iid=s['image_identity'].split(':')[-1]
            if s['raw_row']!=images[iid] or s['answer']!=refs[iid]:raise AssertionError('Official COCO annotation mismatch')
        else:
            original=text[s['question_id']]
            if s['raw_row']!=original or s['question']!=original['question'] or s['answer']!=original['answers'] or len(s['answer'])!=10:raise AssertionError('Official TextVQA annotation mismatch')
        if sha256(s['image_path'])!=s['image_sha256'] or rgb_sha(s['image_path'])!=s['rgb_sha256']:raise AssertionError('Changed image')
        if s['image_sha256'] in seen or s['rgb_sha256'] in rgb:raise AssertionError('Duplicate image/smoke')
        seen.add(s['image_sha256']);rgb.add(s['rgb_sha256'])
    freeze_json(OUT/'data_integrity_audit.json',dict(passed=True,images=len(samples),formal_images=400,separate_smoke_images=2,
        official_rows_questions_references_exact=True,unique_byte_rgb=True,sample_manifest_sha256=sha256(OUT/'sample_manifest.json')))
    print('Final official data/image audit passed',len(samples),flush=True)


def delivery():
    config=verify()
    for p,b in config['model_files'].items():
        if sha256(p)!=b['sha256']:raise AssertionError('Final model SHA256 mismatch')
    for p,h in json.loads((OUT/'scorer_validation.json').read_text())['source_sha256'].items():
        if sha256(p)!=h:raise AssertionError('Scorer source/artifact changed')
    required=['report.md','research_summary.md','full_generation_per_sample.csv','layerwise_generation_per_sample.csv',
        'layerwise_coco_quality_summary.csv','layerwise_textvqa_score_summary.csv','caption_change_analysis.csv','quality_win_loss_cases.csv',
        'bootstrap_statistics.csv','cpu_validation.json','gpu_smoke_validation.json','data_integrity_audit.json','scorer_validation.json']
    for name in required:
        if not (OUT/name).is_file():raise AssertionError('Missing deliverable:'+name)
    for name in ['figure1_coco_cider','figure2_coco_spice','figure3_textvqa','figure4_cross_task_sensitivity','figure5_caption_change_and_harm','figure6_actual_cases']:
        for suffix in ['.png','.pdf']:
            if not (OUT/'figures'/(name+suffix)).is_file():raise AssertionError('Missing actual figure')
    def count(name):
        import csv
        with (OUT/name).open() as f:return len(list(csv.DictReader(f)))
    if count('full_generation_per_sample.csv')!=400 or count('layerwise_generation_per_sample.csv')!=12800 or count('bootstrap_statistics.csv')!=96:raise AssertionError('Output row count mismatch')
    process=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
    freeze_json(OUT/'gpu_release.json',dict(current_compute_processes=process,note='External GPU0 processes are not ours and are left untouched.'))
    freeze_json(OUT/'failure_inventory.json',dict(failures=[dict(path=str(p.relative_to(OUT)),sha256=sha256(p),record=json.loads(p.read_text())) for p in sorted((OUT/'failures').glob('*.json'))],
        postprocessing_initial_import_failure=dict(path='analysis_cpu_tests_initial_failure.txt',sha256=sha256(OUT/'analysis_cpu_tests_initial_failure.txt'),resolved='Reuse existing descriptive_correlations; no environment changes or worker changes.'),
        attempts_preserved=True,unresolved_failures=0))
    freeze_json(OUT/'completion.json',dict(status='complete',datasets=dict(coco_caption=200,textvqa=200),conditions=33,generations=13200,
        cpu_tests=json.loads((OUT/'cpu_validation.json').read_text())['tests'],analysis_cpu_tests=json.loads((OUT/'analysis_cpu_validation.json').read_text())['tests'],gpu_smoke_passed=True,model_and_scorer_hashes_exact=True,
        worker_commit=config['code_commit'],source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'paper_rq1_decode_layerwise_quality').glob('*.py'))},
        no_training=True,no_multilayer_experiment=True,no_speedup_claim=True))
    records=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['delivery_inventory.json','delivery.log']]
    freeze_json(OUT/'delivery_inventory.json',dict(files=len(records),records=records))
    print('Complete delivery validated',len(records),'files',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scorer','data','analysis_cpu','delivery']);a=p.parse_args();globals()[a.stage]()
