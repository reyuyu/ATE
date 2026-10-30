"""Official fixtures and final source/data/artifact checks; no additional GPU work."""
import argparse
from collections import defaultdict
import csv
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


def tables():
    def read(name):
        with (OUT/name).open() as f:return list(csv.DictReader(f))
    manifest=json.loads((OUT/'sample_manifest.json').read_text());samples={(s['dataset'],s['question_id']):s for s in manifest['samples']}
    full=read('full_generation_per_sample.csv');rows=read('layerwise_generation_per_sample.csv')
    base={(r['dataset'],r['question_id']):r for r in full};stats=read('bootstrap_statistics.csv');groups=defaultdict(list)
    official={c:json.loads((OUT/'official_scores'/(c+'.json')).read_text()) for c in ['Full']+[f'L{k}-off' for k in range(32)]}
    max_difference=0.;validated=0
    if len(base)!=400 or len(samples)!=400:raise AssertionError('Baseline sample keys not unique')
    seen=set()
    for row in full+rows:
        key=(row['dataset'],row['question_id']);condition=row['condition'];s=samples[key]
        identifier=(key,condition)
        if identifier in seen:raise AssertionError('Duplicate CSV condition')
        seen.add(identifier)
        path=OUT/f"generation/shard{s['rank']}/checkpoints"/(s['dataset']+'__'+s['question_id'])/(condition+'.json')
        actual=json.loads(path.read_text())['generation'];tokens=json.loads(row['token_ids'])
        if tokens!=actual['token_ids'] or row['text']!=actual['text'] or row['image_sha256']!=s['image_sha256']:raise AssertionError('CSV not equal actual generation')
        if tokens[0]!=json.loads(base[key]['token_ids'])[0]:raise AssertionError('First token changed')
        if actual['edited_steps']!=(0 if condition=='Full' else len(tokens)-1):raise AssertionError('Wrong number of Decode edits')
        metrics=['CIDEr','SPICE'] if key[0]=='coco_caption' else ['soft_score']
        for m in metrics:
            expected=official[condition]['per_image'][key[1]][m] if key[0]=='coco_caption' else textvqa_score(row['text'],s['answer'])
            delta=abs(float(row[m])-expected);max_difference=max(max_difference,delta)
            if delta>1e-12 or not np.isfinite(float(row[m])):raise AssertionError('Official per-image score mismatch')
            if abs(float(row['delta_'+m])-(float(row[m])-float(base[key][m])))>1e-12:raise AssertionError('Paired score delta mismatch')
            if condition!='Full':groups[(key[0],int(row['layer']),m)].append(row)
        validated+=1
    for stat in stats:
        m='soft_score' if stat['metric']=='TextVQA' else stat['metric'];selected=groups[(stat['dataset'],int(stat['layer']),m)]
        if len(selected)!=200 or len({r['question_id'] for r in selected})!=200:raise AssertionError('Wrong per-layer coverage')
        changes=np.array([float(r['delta_'+m]) for r in selected]);mean=float(changes.mean());score=np.mean([float(r[m]) for r in selected])
        if abs(mean-float(stat['delta']))>1e-12 or abs(score-float(stat['ablated_score']))>1e-12:raise AssertionError('Aggregated mean mismatch')
        if stat['dataset']=='coco_caption':
            corpus=official['L'+stat['layer']+'-off']['corpus'][m]
            if abs(score-corpus)>1e-12:raise AssertionError('Corpus metric vs per-image mean mismatch')
        if int(stat['improved'])!=sum(changes>1e-12) or int(stat['harmed'])!=sum(changes< -1e-12):raise AssertionError('Harm/correction count mismatch')
        for prefix in ['ci95','family95']:
            if float(stat[prefix+'_low'])>float(stat[prefix+'_high']):raise AssertionError('Invalid interval')
        if float(stat['family95_low'])>float(stat['ci95_low']) or float(stat['family95_high'])<float(stat['ci95_high']):raise AssertionError('Family interval narrower than nominal')
        for key,prefix in [('ordinary_noninferior','ci95'),('family_noninferior','family95')]:
            if (stat[key]=='True')!=(float(stat[prefix+'_low'])> -float(stat['ni_margin'])):raise AssertionError('Incorrect noninferiority label')
    findings=json.loads((OUT/'main_findings.json').read_text())
    if findings['report_sha256']!=sha256(OUT/'report.md'):raise AssertionError('Report changed after numeric summary')
    freeze_json(OUT/'table_consistency_audit.json',dict(passed=True,actual_generations_checked=validated,endpoints_checked=len(stats),
        raw_text_and_ids_exact=True,official_scores_exact=True,max_score_difference=max_difference,paired_mean_flip_and_ci_labels_checked=True,report_sha256=sha256(OUT/'report.md')))
    print('Independent table/actual-generation consistency passed',validated,'generations',len(stats),'endpoints',flush=True)


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


def text_length():
    from transformers import AutoTokenizer
    from .protocol import MODEL_PATH
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=True)
    special=set(tokenizer.all_special_ids);records=[]
    with (OUT/'full_generation_per_sample.csv').open() as f:
        for row in csv.DictReader(f):
            if row['dataset']!='textvqa':continue
            tokens=json.loads(row['token_ids']);steps=[]
            for i,token in enumerate(tokens):
                piece=tokenizer.decode([token],clean_up_tokenization_spaces=False)
                if i>=1 and token not in special and any(c.isalnum() for c in piece):steps.append(dict(t=i,id=token,piece=piece))
            records.append(dict(question_id=row['question_id'],image_identity=row['image_identity'],full_text=row['text'],content_tokens=int(row['content_tokens']),
                has_substantive_decode=bool(steps),substantive_decode_steps=json.dumps(steps,ensure_ascii=False)))
    with (OUT/'textvqa_full_decode_content_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    freeze_json(OUT/'textvqa_full_decode_content_audit.json',dict(questions=len(records),with_substantive_decode=sum(r['has_substantive_decode'] for r in records),
        only_eos_or_whitespace_punctuation_decode=sum(not r['has_substantive_decode'] for r in records),
        definition='A native generated position t>=1 with non-special token whose individual decoded piece contains an alphanumeric character. Includes subwords; not an independent-word or human semantic annotation.'))
    print('TextVQA substantive subsequent-token audit',sum(r['has_substantive_decode'] for r in records),'of',len(records),flush=True)


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
    tables()
    timings=[]
    for stage in ['smoke','generation']:
        for rank in range(3):
            completed=json.loads((OUT/f'{stage}/shard{rank}/completion.json').read_text())
            timings.append(dict(stage=stage,physical_gpu=rank+1,seconds=completed['seconds'],
                scope='Worker stage after model loading; three workers parallel, do not sum as elapsed wall time.'))
    with (OUT/'execution_times.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
    raw=[]
    for smoke in json.loads((OUT/'gpu_smoke_validation.json').read_text())['checks']:
        path=Path(smoke['raw_path'])
        if sha256(path)!=smoke['raw_sha256']:raise AssertionError('Saved Smoke raw logits changed')
        raw.append(dict(path=str(path),sha256=smoke['raw_sha256'],bytes=path.stat().st_size))
    with (OUT/'raw_artifacts_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);writer.writeheader();writer.writerows(raw)
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
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scorer','data','analysis_cpu','text_length','delivery']);a=p.parse_args();globals()[a.stage]()
