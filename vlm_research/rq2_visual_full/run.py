"""Reuse all historical generators/masks; add only cohort binding and scheduling."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared, atomic_json, tensor_hashes
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from figure3_decode_windows.run import cohort
from rq2_visual_sufficiency import run as previous
from rq2_visual_sufficiency.engine import CONDITIONS, LABELS, generate
from vlm_common import load_model, gpu_memory

OUT = ROOT/'results/rq2_visual_sufficiency_full3000'
BASE = 'bdcf9058e3ae676689c01f91653d371fd1e42102'
TASKS, GPUS, SEED = previous.old.TASKS, previous.GPUS, 20261031
PRIMARY = 'validation_image_disjoint'
SECONDARY = [('C','A'),('C','B'),('D','Full'),('A','Full')]


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names=set(previous.sources())|{'rq2_visual_full/__init__.py','rq2_visual_full/run.py',
                                  'test_rq2_visual_full.py','first_answer_attention/run.py'}
    return {n:sha256(ROOT/n) for n in sorted(names)}


def split(ss, discovery):
    ids={s['question_id'] for s in discovery}
    images={s['image_sha256'] for s in discovery}
    assert len(ids)==240 and ids <= {s['question_id'] for s in ss}
    return {s['question_id']:dict(exploration=s['question_id'] in ids,
        validation=s['question_id'] not in ids,
        image_disjoint=s['question_id'] not in ids and s['image_sha256'] not in images)
        for s in ss}


def populations(ss, flags):
    return dict(exploration=[s for s in ss if flags[s['question_id']]['exploration']],
        validation=[s for s in ss if flags[s['question_id']]['validation']],
        validation_image_disjoint=[s for s in ss if flags[s['question_id']]['image_disjoint']],
        all3000=list(ss))


@lru_cache(maxsize=1)
def history():
    return read(OUT/'history_bindings.json')['records']


def native_logits(s, verify_file=False):
    rec=history()[s['question_id']]
    if verify_file:
        assert sha256(rec['raw_logits_path'])==rec['raw_logits_sha256']
    with np.load(rec['raw_logits_path']) as z:
        first=torch.from_numpy(z['full_logits'].copy())
    assert tensor_hash(first)==rec['first_logits_sha256']
    return first


def historical_new_path(s,key):
    rank=read(previous.OUT/'assignments.json')[s['question_id']]
    return previous.OUT/f"shard{rank}/checkpoints/{s['task']}__{s['question_id']}/{key}.json"


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    ss=sorted(cohort(),key=lambda s:(s['task'],s['question_id']))
    discovery=previous.samples()
    flags=split(ss,discovery);groups=populations(ss,flags)
    assert len(ss)==3000 and len(groups['validation'])==2760
    assert Counter(s['task'] for s in ss)==Counter({t:500 for t in TASKS})
    assert all(s in ss for s in discovery)
    # Verify repository artifacts against the exact user-specified historical commits.
    artifacts={
        '221cdde56dde53908aaa953fe26672fd33334f84':'results/paper_rq1_single_layer_gqa6/sample_manifest.json',
        'e495adf4fd93337a3638981bbb66c2bbac81242d':'results/first_answer_early_attention/baseline_bindings.json',
        '3a7c498c722dab7344419490372c2af5dbb67a4d':'results/rq2_single_layer_source_240/sample_manifest.json',
        BASE:'results/rq2_two_layer_visual_sufficiency_240/artifact_hashes.json'}
    blobs={}
    for commit,name in artifacts.items():
        blob=subprocess.check_output(['git','rev-parse',f'{commit}:vlm_research/{name}'],cwd=ROOT,text=True).strip()
        assert blob==subprocess.check_output(['git','hash-object',ROOT/name],text=True).strip()
        blobs[commit]=dict(path=name,blob=blob,sha256=sha256(ROOT/name))
    bound={s['question_id']:s for s in read(ROOT/'results/first_answer_early_attention/baseline_bindings.json')}
    prior_manifest=read(previous.OUT/'artifact_hashes.json')
    for name,h in prior_manifest.items():
        assert sha256(previous.OUT/name)==h
    figure=read(previous.FIGURE/'execution_manifest.json')
    records={}
    for s in ss:
        assert sha256(s['image_path'])==s['image_sha256']
        full_path=previous.full_path(s);full=read(full_path)
        assert full['binding']==digest(figure) and full['sample_sha256']==digest(s)
        assert full['result_sha256']==digest(full['generation'])
        assert bound[s['question_id']]['full_checkpoint_sha256']==sha256(full_path)
        assert bound[s['question_id']]['sample_sha256']==digest(s)
        g=full['generation']
        assert g['score']==int(gqa_prediction(g['text'])==s['answer']) and g['edits']==0
        old,path=previous.old.old_record(s)
        assert sha256(old['logits_path'])==old['logits_sha256']
        with np.load(old['logits_path']) as z:
            first=torch.from_numpy(z['full_logits'].copy())
        assert torch.isfinite(first).all() and int(first.argmax())==g['token_ids'][0]
        records[s['question_id']]=dict(sample_sha256=digest(s),prompt_sha256=digest(s['input_mapping']),
            full_path=str(full_path.relative_to(ROOT)),full_sha256=sha256(full_path),
            first_token_checkpoint=str(path.relative_to(ROOT)),first_checkpoint_sha256=sha256(path),
            raw_logits_path=old['logits_path'],raw_logits_sha256=old['logits_sha256'],
            first_logits_sha256=tensor_hash(first),reused={})
        if flags[s['question_id']]['exploration']:
            for key in CONDITIONS:
                p=historical_new_path(s,key);r=read(p)
                assert r['sample_sha256']==digest(s) and digest(r['generation'])==r['result_sha256']
                assert digest(r['first_token'])==r['metric_sha256']
                assert r['generation']['masked_layers']==CONDITIONS[key]
                assert sha256(previous.OUT/r['logits_file'])==r['logits_sha256']
                records[s['question_id']]['reused'][key]=dict(path=str(p.relative_to(ROOT)),sha256=sha256(p))
    assignments=previous.assign(groups['validation'])
    integrity=dict(original_questions=3000,original_images=len({s['image_sha256'] for s in ss}),
        populations={k:dict(questions=len(v),images=len({s['image_sha256'] for s in v}),
                            tasks=dict(Counter(s['task'] for s in v))) for k,v in groups.items()},
        validation_shared_with_exploration_questions=len(groups['validation'])-len(groups[PRIMARY]),
        shared_image_sha256=sorted({s['image_sha256'] for s in groups['validation']}&{s['image_sha256'] for s in discovery}),
        question_id_disjoint=True,image_disjoint_primary=True,primary_population=PRIMARY,
        statistical_role='Image-disjoint validation is the sole primary NI test. All2760 is supporting; exploration/all3000 are not independent confirmation.',
        selection='Original3000 unchanged. Remove original240 question IDs for validation; exclude every discovery image SHA256 for primary, without model-output filtering.',
        historical_git_blobs=blobs,shards=dict(Counter(assignments.values())),
        historical_full_correct=sum(read(previous.full_path(s))['generation']['score'] for s in ss))
    assert integrity['historical_full_correct']==2627
    assert not ({s['image_sha256'] for s in groups[PRIMARY]}&{s['image_sha256'] for s in discovery})
    freeze_json(OUT/'sample_manifest.json',dict(samples=ss,splits=flags,assignments=assignments,
        original_manifest_sha256=sha256(previous.old.ORIGINAL/'sample_manifest.json')))
    freeze_json(OUT/'history_bindings.json',dict(records=records,historical_git_blobs=blobs,
        figure_protocol_sha256=sha256(previous.FIGURE/'execution_manifest.json'),
        discovery_protocol_sha256=sha256(previous.OUT/'protocol.json')))
    freeze_json(OUT/'split_integrity.json',integrity)
    print(json.dumps(integrity,ensure_ascii=False,indent=2),flush=True)


def cpu():
    stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_rq2_source_knockout','test_rq2_visual_sufficiency','test_rq2_visual_full'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()
    freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source=sources(),output=stream.getvalue()))


def code_check():
    paths=['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert read(OUT/'cpu_validation.json')['source']==sources()


def smoke(rank):
    code_check();pre=previous.preflight(rank);started=time.perf_counter()
    model,processor,environment=load_model(MODEL_PATH)
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    manifest=read(OUT/'sample_manifest.json');ss=manifest['samples'];checks=[]
    selected=[next(s for s in ss if s['task']=='ChooseRel' and manifest['splits'][s['question_id']]['exploration']),
              next(s for s in ss if s['task']=='QueryAttr' and manifest['splits'][s['question_id']]['image_disjoint'])]
    for s in selected:
        prompt=prepared(model,processor,s);before=tensor_hashes(prompt);bos=processor.tokenizer.bos_token_id
        full,first,trace,debug=generate(model,prompt,bos,[],audit=True,collect=True)
        previous_full=read(previous.full_path(s))['generation']
        assert full['token_ids']==previous_full['token_ids'] and torch.equal(first,native_logits(s,True))
        with torch.inference_mode():
            native=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,prompt['input_ids'].shape[1]:].tolist()
        assert native==full['token_ids']
        noop,no_first,no_trace,no_debug=generate(model,prompt,bos,CONDITIONS['A'],noop=True,audit=True,collect=True)
        assert torch.equal(debug['prefill_logits'],no_debug['prefill_logits']) and noop['token_ids']==full['token_ids']
        assert len(no_trace)==len(trace) and all(torch.equal(a,b) for a,b in zip(trace,no_trace))
        interventions=[]
        for key,blocked in CONDITIONS.items():
            g,logits,_,d=generate(model,prompt,bos,blocked,audit=True)
            assert torch.equal(d['prefill_logits'][0,:-1],debug['prefill_logits'][0,:-1])
            assert d['observer'].states==debug['observer'].states
            assert all(a['other_query_attention_sha256']==b['other_query_attention_sha256'] for a,b in zip(d['observer'].rows,debug['observer'].rows))
            assert all(r['Image_max']==0 if r['layer'] in blocked else r['Image']>0 for r in d['observer'].rows)
            metric=metric_rows(first,logits[None])[0];metric['logp_loss']=-metric['delta_logp']
            if manifest['splits'][s['question_id']]['exploration']:
                old=read(historical_new_path(s,key))
                assert metric==old['first_token'] and all(g[k]==old['generation'][k] for k in g)
                with np.load(previous.OUT/old['logits_file']) as z:
                    assert np.array_equal(z['first_logits'],logits.numpy())
            interventions.append(dict(condition=key,first_logits_sha256=tensor_hash(logits),token_ids=g['token_ids'],
                masked_probabilities_zero=True,kept_layers_native_mask=True,other_queries_exact=True,
                decode_native=True,own_cache=True,historical_reuse_exact=manifest['splits'][s['question_id']]['exploration']))
        assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
        assert versions=={n:p._version for n,p in model.named_parameters()}
        checks.append(dict(task=s['task'],question_id=s['question_id'],p_minus_one=prompt['input_ids'].shape[1]-1,
            visual_keys=576,noop_all_logits_and_generation_exact=True,native_history_exact=True,conditions=interventions))
    atomic_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,source=sources(),environment=environment,
        preflight=pre,checks=checks,seconds=time.perf_counter()-started))
    print('Smoke passed GPU',rank+1,flush=True)


def freeze():
    code_check();smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(3)]
    envkeys=['python','torch','transformers','cuda_runtime','cudnn','gpu','model_id','model_revision','parameter_dtypes']
    assert all(s['passed'] and s['source']==sources() and s['checks']==smokes[0]['checks'] for s in smokes)
    assert all({k:s['environment'][k] for k in envkeys}=={k:smokes[0]['environment'][k] for k in envkeys} for s in smokes)
    old=read(previous.OUT/'protocol.json')
    for p,b in old['model_files'].items():assert sha256(p)==b['sha256']
    names=['sample_manifest.json','history_bindings.json','split_integrity.json','cpu_validation.json',
           *[f'shard{k}/smoke.json' for k in range(3)]]
    freeze_json(OUT/'protocol.json',dict(status='frozen',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_revision=old['model_revision'],model_files=old['model_files'],
        transformers_source=old['transformers_source'],transformers='4.49.0',dtype='FP16',attention='eager',
        gpu_uuids=GPUS,conditions=CONDITIONS,labels=LABELS,total=3000,reused_full=3000,reused_interventions=960,
        new_questions=2760,new_generations=11040,generation=old['generation'],
        intervention=old['intervention'],primary_population=PRIMARY,
        statistics=dict(seed=SEED,repetitions=200000,primary=['C','Full'],noninferiority_pp=-1.,
            criterion='Sole primary image-disjoint validation lower two-sided95% percentile image-cluster paired bootstrap bound strictly > -1pp (one-sided97.5%).',
            secondary_family=SECONDARY,secondary_control='Bonferroni4 two-sided familywise95% intervals on primary population; exclude0 for a directional difference claim.',
            unit='Resample whole image SHA256 clusters, retain question weighting; pair all conditions within image.',
            other_populations='All2760 supporting. Exploration240 and all3000 descriptive, not independent confirmation. Show ordinary intervals and Bonferroni4 reference intervals without new confirmatory claims.',
            task_results='Descriptive only; no additional hypothesis selection.',
            first_metrics='Existing signed FP32 Full-top1 LogP loss and whole-vocabulary JS; not gold accuracy.'),
        boundaries='Only frozen five conditions; no new layers, training, model/data expansion or physical K/V acceleration. Analysis/figure-only sources get separate hashes before aggregation.'))
    print('Protocol frozen; main population:',PRIMARY,flush=True)


def verify():
    c=read(OUT/'protocol.json')
    assert c['source']==sources() and c['conditions']==CONDITIONS and c['gpu_uuids']==GPUS
    for p,h in c['bindings'].items():assert sha256(OUT/p)==h
    for p,h in c['transformers_source'].items():assert sha256(p)==h
    for p,b in c['model_files'].items():
        st=Path(p).stat();assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    return c


def checkpoint_valid(path,binding,s,key):
    r=read(path)
    assert r['binding']==binding and r['sample_sha256']==digest(s) and r['condition']==key
    assert digest(r['generation'])==r['result_sha256'] and digest(r['first_token'])==r['metric_sha256']
    assert sha256(OUT/r['logits_file'])==r['logits_sha256']
    return r


def worker(rank):
    c=verify();binding=digest(c);folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding
        print('Already complete; no GPU loaded',flush=True);return
    pre=previous.preflight(rank);started_epoch=time.time();started=time.perf_counter()
    model,processor,environment=load_model(MODEL_PATH)
    assert environment['model_revision']==c['model_revision'] and environment['transformers']=='4.49.0'
    assert environment['parameter_dtypes']==['torch.float16'] and not model.training
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    manifest=read(OUT/'sample_manifest.json')
    ss=[s for s in manifest['samples'] if manifest['assignments'].get(s['question_id'])==rank]
    resumed=0
    for i,s in enumerate(ss):
        prompt=None
        for key,blocked in CONDITIONS.items():
            path=folder/f"checkpoints/{s['task']}__{s['question_id']}/{key}.json"
            if path.exists():
                checkpoint_valid(path,binding,s,key);resumed+=1;continue
            if prompt is None:
                prompt=prepared(model,processor,s);before=tensor_hashes(prompt);full=native_logits(s)
            tick=time.perf_counter()
            g,first,_,_=generate(model,prompt,processor.tokenizer.bos_token_id,blocked)
            text=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            g.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==s['answer']))
            metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp']
            raw=path.with_suffix('.npz');raw.parent.mkdir(parents=True,exist_ok=True);temp=raw.with_suffix('.npz.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,first_logits=first.numpy())
            temp.replace(raw)
            assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint
            assert versions=={n:p._version for n,p in model.named_parameters()}
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),task=s['task'],question_id=s['question_id'],
                image_id=s['image_id'],image_sha256=s['image_sha256'],condition=key,generation=g,
                result_sha256=digest(g),first_token=metric,metric_sha256=digest(metric),
                logits_file=str(raw.relative_to(OUT)),logits_sha256=sha256(raw),seconds=time.perf_counter()-tick,
                hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True,physical_gpu=rank+1))
        if (i+1)%25==0 or i+1==len(ss):
            print(json.dumps(dict(gpu=rank+1,questions=i+1,total=len(ss),generations=(i+1)*4,
                seconds=time.perf_counter()-started)),flush=True)
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,questions=len(ss),generations=len(ss)*4,
        resumed=resumed,started_epoch=started_epoch,finished_epoch=time.time(),seconds=time.perf_counter()-started,
        environment=environment,preflight=pre,memory=gpu_memory(),hooks_clean=hook_footprint(model)==footprint,
        parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('Complete GPU',rank+1,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cpu','smoke','freeze','worker'])
    parser.add_argument('--rank',type=int,choices=[0,1,2],default=0);args=parser.parse_args()
    try:
        if args.stage in ('smoke','worker'):globals()[args.stage](args.rank)
        else:globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
