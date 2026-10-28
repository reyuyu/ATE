"""Frozen GPU1-only 16-image sink-stage pilot: prepare, CPU, smoke, run."""
import argparse
from contextlib import nullcontext
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import unittest
import uuid
from vlm_common import ROOT, load_model
import numpy as np
import torch
from fixed_k_exit import atomic_json
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from knockout_probe import model_layers, resource_preflight, write_csv
from roi_sink_probe import inputs_and_history, score
from roi_sink_common import forward
from phase38_sink_hooks import CONDITIONS, SinkStageTrace, thash

OUTPUT = ROOT/'results/phase3_8_sink_stage'
SOURCE = ROOT/'results/phase3_4_roi_sink_diagnostic'
HISTORY = ROOT/'results/phase3_2_fixed_k_exit'
SEED = 3820261029
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
IMAGES = Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')
PINS = {'torch':'2.6.0+cu124','torchvision':'0.21.0+cu124','transformers':'4.49.0',
        'accelerate':'1.4.0','huggingface_hub':'0.29.3','safetensors':'0.5.3',
        'pillow':'11.1.0','numpy':'1.26.4','matplotlib':'3.10.1','sentencepiece':'0.2.0'}
HIDDEN = [1,2,3,4,7,31,'final_norm']


def read(p):return json.loads(Path(p).read_text())


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def sources():return {p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))}


def choose_samples():
    m = read(SOURCE/'manifest.json')
    with (HISTORY/'per_sample_results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    margins = {r['question_id']:float(r['full_margin']) for r in rows if r['cutoff']=='24'}
    pool = sorted([copy.deepcopy(s) for s in m['samples'] if s['question_id'] in m['main_question_ids']],key=lambda s:s['question_id'])
    if len(pool)!=48 or any(s['question_id']=='15133706' for s in pool):raise ValueError('Frozen 48 main images required')
    rng = random.Random(SEED)
    left = [s for s in pool if s['answer']=='left']
    right_good = [s for s in pool if s['answer']=='right' and margins[s['question_id']]>0]
    right_bad = [s for s in pool if s['answer']=='right' and margins[s['question_id']]<=0]
    for group in [left,right_good,right_bad]:rng.shuffle(group)
    right = right_good[:4]+right_bad[:4]
    rest = sorted([s for s in right_good+right_bad if s not in right],key=lambda s:s['question_id'])
    rng.shuffle(rest);right += rest[:8-len(right)]
    chosen = sorted(left[:8]+right,key=lambda s:s['question_id'])
    if len(chosen)!=16 or len({s['img_id'] for s in chosen})!=16:raise ValueError('Pilot selection incomplete')
    for s in chosen:s['selection_historical_full_margin']=margins[s['question_id']]
    rule = dict(seed=SEED,source='Phase3.4 frozen 48 main images only; no five historical cases',
        algorithm='Sort string qid; Random(seed) shuffles left, historical-Full-correct right, historical-Full-wrong right in that order. Take8 left, up to4 each right stratum; fill missing right slots by shuffling sorted remaining right. Return qid sorted.',
        labels='8 left / 8 right; right selection uses historical Native Full only, never current interventions',
        excluded_discovery='15133706', historical_right_correct_available=len(right_good),historical_right_wrong_available=len(right_bad))
    return chosen,rule


def prepare():
    if OUTPUT.exists():raise FileExistsError('Never replace an existing frozen experiment directory')
    if git('branch','--show-current')!='exp/phase3_8-sink-stage':raise ValueError('Wrong isolated branch')
    if git('diff','HEAD','--','*.py'):raise ValueError('Commit implementation before protocol freeze')
    samples,rule = choose_samples()
    sink = {s['question_id']:s for s in read(SOURCE/'sink_manifest.json')['samples']}
    files = [SOURCE/'manifest.json',SOURCE/'sink_manifest.json',SOURCE/'preregistered_protocol.json',
             HISTORY/'manifest.json',HISTORY/'per_sample_results.csv',ROOT/'configs/model.json',ROOT/'requirements.txt']
    for s in samples:
        p = IMAGES/f"image_{s['img_id']}.jpg"
        if sha256(p)!=s['image_sha256']:raise ValueError('Frozen image hash mismatch')
        files += [HISTORY/'checkpoints/run'/f"{s['question_id']}.json"]
        entry = sink[s['question_id']]
        if not entry['main'] or entry['roi_indices']!=s['roi_indices']:raise ValueError('Frozen ROI mismatch')
        if sorted(set(entry['roi_indices'])&set(entry['sink_indices']))!=sorted(entry['roi_sink_overlap_indices']):raise ValueError('Frozen ROI/sink intersection mismatch')
    model_files = {}
    for item in read(ROOT/'configs/model.json')['files']:
        p = MODEL/item['name'];h = sha256(p)
        if p.stat().st_size!=item['bytes'] or (item.get('lfs_sha256') and h!=item['lfs_sha256']):raise ValueError('Fixed model file mismatch')
        model_files[item['name']] = h
    config = dict(phase='3.8',base_commit=git('rev-parse','HEAD^') if git('rev-list','--count','HEAD')!='1' else None,
        code_commit=git('rev-parse','HEAD'),upstream_main_checked='e89c8acce6e674ecfbd2c8acad57faf71bf4d547',
        sources=sources(),inputs={str(p.relative_to(ROOT)):sha256(p) for p in files},
        model_path=str(MODEL),image_root=str(IMAGES),model_files_sha256=model_files,packages=PINS,
        conditions=CONDITIONS,selection=rule,samples=samples,
        sink_entries={s['question_id']:sink[s['question_id']] for s in samples},hidden_locations=HIDDEN,
        gpu_policy='Only physical GPU1; internal cuda:0 and exactly one visible device; no other jobs killed',
        interpretation='16-image mechanism pilot, not overall accuracy evaluation. No additional conditions, training, physical exit, decoding, new data or causal token deletion claim.',
        denominators='Original raw masses: BOS/Image/Question/other text partition all keys. Sink raw is inside Image; conditional Sink fraction is mean of headwise ratios, separately also ratio of mean masses. Norms never summed across sources.',
        residuals='Attention o_proj output / layer input; MLP output / pre-MLP residual, plus / layer input. Cosines use corresponding actual residual. Hidden layer labels are post-layer, zero-based; final_norm additionally after final RMSNorm.')
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT/'manifest.json',dict(binding=digest(config),config=config))
    print('FROZEN',config['code_commit'],digest(config),[s['question_id'] for s in samples],flush=True)


def protocol():
    m = read(OUTPUT/'manifest.json');c = m['config']
    if digest(c)!=m['binding'] or c['conditions']!=CONDITIONS or sources()!=c['sources'] or git('rev-parse','HEAD')!=c['code_commit']:
        raise ValueError('Frozen source/condition/commit mismatch')
    for p,h in c['inputs'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Historical input modified: '+p)
    return c,m['binding']


def cpu():
    c,b = protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU acceptance must hide all GPUs')
    import vlm_common
    vlm_common.default_model=lambda:str(MODEL)
    folder=OUTPUT/'tests';folder.mkdir()
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test*.py')
    expected=suite.countTestCases()
    with (folder/'cpu_tests.txt').open('x') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    pins={k:importlib.metadata.version(k) for k in PINS}
    pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (folder/'pip_check.txt').write_text(pip.stdout+pip.stderr)
    (folder/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    passed=result.wasSuccessful() and not result.skipped and result.testsRun==expected and pins==PINS and pip.returncode==0
    atomic_json(folder/'validation.json',dict(passed=passed,binding=b,code_commit=c['code_commit'],tests=result.testsRun,
        expected=expected,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        packages=pins,python=platform.python_version(),interpreter=sys.executable))
    print('CPU',passed,result.testsRun,expected,flush=True)
    if not passed:raise RuntimeError('CPU gate failed; see logs, do not relax')


def library_hash():
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaAttention,LlamaMLP
    classes=[LlamaDecoderLayer,LlamaAttention,LlamaMLP]
    return dict(files={inspect.getfile(cls):sha256(inspect.getfile(cls)) for cls in classes},
                functions={cls.__name__:hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest() for cls in classes})


def weights_hash(model):
    values={name:dict(shape=list(p.shape),dtype=str(p.dtype),sha256=thash(p)) for name,p in model.state_dict().items()}
    return dict(sha256=digest(values),tensors=values)


def hooks_state(model):return [(name,len(m._forward_pre_hooks),len(m._forward_hooks)) for name,m in model.named_modules()]


def input_hash(inputs):return digest({k:thash(v) for k,v in inputs.items()})


def evaluate(model, inputs, qpos, apos, trace=None):
    before=input_hash(inputs);hooks=hooks_state(model);captures=[]
    positions=torch.where((inputs['input_ids'][0]!=model.config.image_token_index)&inputs['attention_mask'][0].bool())[0]
    handle=model.register_forward_hook(lambda m,a,o:captures.append(o.logits[0,positions].detach().cpu()))
    try:
        with trace if trace else nullcontext():
            value=forward(model,inputs,qpos,apos)
            if trace:trace.assert_complete()
        if len(captures)!=1 or not torch.isfinite(captures[0]).all() or input_hash(inputs)!=before:raise AssertionError('Input/logits contamination')
        value['effective_text_sha256']=thash(captures[0]);value['input_sha256']=before
        return value
    finally:
        handle.remove()
        if hooks_state(model)!=hooks:raise AssertionError('Hook cleanup failed')


def drift(current, full):
    x,y=np.asarray(current,dtype=np.float64),np.asarray(full,dtype=np.float64)
    delta=float(np.linalg.norm(x-y));den=float(np.linalg.norm(y))
    return dict(l2=delta,relative_l2=delta/den,cosine_distance=0. if np.array_equal(x,y) else max(0.,float(1-np.dot(x,y)/(np.linalg.norm(x)*den))))


def one_sample(model,processor,frozen,sink,folder):
    sample=copy.deepcopy(frozen);sample['image_path']=str(IMAGES/f"image_{sample['img_id']}.jpg")
    if sha256(sample['image_path'])!=sample['image_sha256']:raise ValueError('Input image changed')
    prompt,candidates,qpos,apos,historical,qrecords,counts=inputs_and_history(model,processor,sample)
    ids=prompt['input_ids'][0]
    visual=torch.where(ids==model.config.image_token_index)[0].cpu().tolist()
    bos=torch.where(ids==processor.tokenizer.bos_token_id)[0].cpu().tolist()
    if len(visual)!=576 or len(bos)!=1 or bos[0]!=int(torch.where(prompt['attention_mask'][0].bool())[0][0]) or apos!=len(ids)-1:
        raise AssertionError('Dynamic BOS/image/P-1 mapping invalid')
    bos=bos[0];prefix=[p for p in range(apos) if p not in set(visual)]
    def trace(condition):return SinkStageTrace(model_layers(model),model.language_model.model.norm,apos,bos,visual,qpos,prefix,sink['sink_indices'],condition)
    clean=evaluate(model,candidates[0][0],qpos,apos)
    baseline=[];baseline_traces=[];rows=[];mechanics=[];hidden_rows=[];audit=[];arrays={}
    gold=sample['candidates'].index(sample['answer']);wrong=1-gold
    for name in CONDITIONS:
        values=[];traces=[]
        for i,(inputs,records) in enumerate(candidates):
            t=trace(name);value=evaluate(model,inputs,qpos,apos,t)
            if name=='C0':
                if i==0 and (value['effective_text_sha256']!=clean['effective_text_sha256'] or not torch.equal(value['question_hidden'],clean['question_hidden'])):
                    raise AssertionError('Native observation perturbed output')
                baseline.append(value);baseline_traces.append(t)
            else:
                bt=baseline_traces[i]
                if t.prefix_hashes!=bt.prefix_hashes or not torch.equal(value['question_hidden'],baseline[i]['question_hidden']):
                    raise AssertionError('Causally earlier text/Question changed')
                if not np.array_equal(t.hidden[1],bt.hidden[1]):raise AssertionError('Pre-intervention L1 changed')
            values.append(value);traces.append(t)
            audit.append(dict(condition=name,candidate=sample['candidates'][i],edits=t.edits,
                prefix_text_layer_hashes=t.prefix_hashes,question_hidden_sha256=thash(value['question_hidden']),
                input_sha256=value['input_sha256'],effective_text_logits_sha256=value['effective_text_sha256'],
                hooks_cleaned=True,layer_count=len(t.metrics),pre_attention_residual_equations_exact=True,
                post_mlp_residual_equations_exact=True,all_attention_and_mlp_modules_executed=True,
                pre_intervention_L1_sha256=hashlib.sha256(t.hidden[1].tobytes()).hexdigest(),
                measurement=value['measurement']))
        if not torch.equal(values[0]['prediction_logits'],values[1]['prediction_logits']):raise AssertionError('Answer leaked into P-1')
        if any(not np.array_equal(traces[0].hidden[k],traces[1].hidden[k]) for k in HIDDEN):raise AssertionError('Candidate leaked into P-1 hidden')
        logs=[score(v,records) for v,(_,records) in zip(values,candidates)]
        full_logs=[score(v,records) for v,(_,records) in zip(baseline,candidates)]
        margin=logs[gold]-logs[wrong];full_margin=full_logs[gold]-full_logs[wrong]
        if not np.isfinite(logs).all():raise AssertionError('NaN/Inf candidate scores')
        rows.append(dict(question_id=sample['question_id'],img_id=sample['img_id'],answer=sample['answer'],condition=name,
            margin=margin,full_margin=full_margin,gain=margin-full_margin,gold_logp=logs[gold],wrong_logp=logs[wrong],
            left_logp=logs[sample['candidates'].index('left')],right_logp=logs[sample['candidates'].index('right')],
            full_gold_logp=full_logs[gold],gold_logp_gain=logs[gold]-full_logs[gold],correct=margin>0,full_correct=full_margin>0,
            correct_to_wrong=full_margin>0 and margin<=0,wrong_to_correct=full_margin<=0 and margin>0,
            weak_full_margin=abs(full_margin)<=.15,prediction_sha256=thash(values[0]['prediction_logits']),
            forward_seconds=sum(v['measurement']['forward_seconds'] for v in values),
            peak_allocated_mib=max(v['measurement']['gpu_memory']['peak_allocated_mib'] for v in values)))
        arrays[f'{name}_prediction_logits']=np.stack([v['prediction_logits'].numpy() for v in values])
        arrays[f'{name}_hidden']=np.stack([traces[0].hidden[k] for k in HIDDEN])
        for k in HIDDEN:hidden_rows.append(dict(question_id=sample['question_id'],answer=sample['answer'],condition=name,location=str(k),
                                            **drift(traces[0].hidden[k],baseline_traces[0].hidden[k])))
        if name=='C0':
            arrays['native_p1_attention_L0_L10']=np.stack([traces[0].rows[i] for i in range(11)])
            for label,values in [('attention_update',traces[0].attention_updates),('mlp_update',traces[0].mlp_updates),
                                 ('layer_input',traces[0].inputs),('pre_mlp_residual',traces[0].mid)]:
                arrays['native_L2_L4_'+label]=np.stack([values[i].cpu().numpy() for i in (2,3,4)])
            for i,m in traces[0].metrics.items():
                mechanics.append(dict(question_id=sample['question_id'],answer=sample['answer'],layer=i,
                    **{k:v for k,v in m.items() if k not in ['layer','attention','mlp']},
                    **{'attention_'+k:v for k,v in m['attention'].items()},**{'mlp_'+k:v for k,v in m['mlp'].items()}))
    restored=evaluate(model,candidates[0][0],qpos,apos)
    if restored['effective_text_sha256']!=clean['effective_text_sha256'] or not torch.equal(restored['question_hidden'],clean['question_hidden']):
        raise AssertionError('Post-cleanup Native Full not exact')
    np.savez_compressed(folder/'raw_vectors_logits_attention.npz',**arrays)
    old_logs=[c['logp'] for c in historical['full_candidates']]
    data=dict(question_id=sample['question_id'],scores=rows,mechanics=mechanics,hidden_drift=hidden_rows,
        audit=audit,input_mapping=dict(prompt_ids=ids.cpu().tolist(),answer_query=apos,bos=bos,visual_indices=visual,
            question_mapping=qrecords,candidate_mapping=historical['candidate_mapping'],counts=counts),
        frozen_sink_indices=sink['sink_indices'],roi_sink_overlap_indices=sink['roi_sink_overlap_indices'],
        historical_candidate_logp_difference=[n-o for n,o in zip(full_logs,old_logs)],
        historical_full_margin=sample['selection_historical_full_margin'],same_machine_full_margin=rows[0]['margin'],
        post_cleanup_native_exact=True,causal_text_all_layers_exact=True,input_unchanged=True,passed=True,
        raw_artifact_sha256=sha256(folder/'raw_vectors_logits_attention.npz'))
    atomic_json(folder/'sample.json',data)
    return data


def execute(stage):
    c,b=protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only physical GPU1 is authorized')
    if {k:importlib.metadata.version(k) for k in PINS}!=PINS:raise ValueError('Fixed package mismatch')
    cpu_gate=read(OUTPUT/'tests/validation.json')
    if not cpu_gate['passed'] or cpu_gate['binding']!=b:raise ValueError('CPU gate incomplete')
    if stage=='run':
        gate=read(OUTPUT/'smoke/validation.json')
        if not gate['passed'] or gate['count']!=2 or gate['binding']!=b:raise ValueError('Smoke incomplete')
    selected=c['samples']
    if stage=='smoke':selected=[next(s for s in selected if s['answer']==a) for a in ('left','right')]
    out=OUTPUT/stage;out.mkdir(exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        complete={}
        if set(p.name for p in out.glob('sample_*'))-{f"sample_{s['question_id']}" for s in selected}:raise ValueError('Foreign checkpoint')
        for s in selected:
            p=out/f"sample_{s['question_id']}"
            if (p/'checkpoint.json').exists():
                e=read(p/'checkpoint.json')
                if e['binding']!=b or e['sample_sha256']!=digest(s) or sha256(p/'sample.json')!=e['sample_json_sha256']:raise ValueError('Checkpoint modified')
                d=read(p/'sample.json')
                if not d['passed'] or sha256(p/'raw_vectors_logits_attention.npz')!=d['raw_artifact_sha256']:raise ValueError('Raw checkpoint artifact changed')
                complete[s['question_id']]=d
            elif p.exists():raise ValueError('Partial sample preserved; investigate before compatible restart')
        if len(complete)==len(selected):print('Already complete',stage,len(complete));return
        resource=resource_preflight()
        if not resource['ready'] or torch.cuda.device_count()!=1:raise RuntimeError('GPU1 resource gate failed; no jobs stopped')
        model,processor,environment=load_model(model_path=str(MODEL))
        library=library_hash();before=weights_hash(model)
        versions={k:p._version for k,p in model.named_parameters()}
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        atomic_json(out/f'environment_{stamp}.json',dict(environment=environment,resource=resource,binding=b,code_commit=c['code_commit'],
            packages={k:importlib.metadata.version(k) for k in PINS},python=platform.python_version(),library_sha256=library,weights_before=before))
        started=time.perf_counter()
        try:
            for s in selected:
                if s['question_id'] in complete:continue
                if sources()!=c['sources']:raise ValueError('Source changed during execution')
                folder=out/f"sample_{s['question_id']}";folder.mkdir()
                data=one_sample(model,processor,s,c['sink_entries'][s['question_id']],folder)
                if versions!={k:p._version for k,p in model.named_parameters()} or library_hash()!=library:raise AssertionError('Weights/version/library changed')
                atomic_json(folder/'checkpoint.json',dict(binding=b,sample_sha256=digest(s),sample_json_sha256=sha256(folder/'sample.json')))
                complete[s['question_id']]=data
                print(stage,len(complete),'/',len(selected),s['question_id'],flush=True)
            after=weights_hash(model)
            if after!=before or library_hash()!=library:raise AssertionError('Full model tensor hash or third-party source changed')
            atomic_json(out/'validation.json',dict(passed=True,binding=b,code_commit=c['code_commit'],count=len(complete),
                question_ids=[s['question_id'] for s in selected],physical_gpu=1,visible_cuda_devices=1,
                wall_seconds=time.perf_counter()-started,weights_before_sha256=before['sha256'],weights_after_sha256=after['sha256'],
                full_state_dict_exact=True,library_unchanged=True,
                sample_sha256={q:sha256(out/f'sample_{q}/sample.json') for q in complete}))
            print('COMPLETE',stage,len(complete),flush=True)
        except BaseException as error:
            atomic_json(out/f'failure_{stamp}.json',dict(status='partial',error=repr(error),traceback=traceback.format_exc(),completed=len(complete)))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','cpu','smoke','run'])
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='cpu':cpu()
    else:execute(args.stage)
