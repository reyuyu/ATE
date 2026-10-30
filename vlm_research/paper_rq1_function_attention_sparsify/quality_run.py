import argparse
import inspect
import json
from pathlib import Path
import subprocess
import time
import traceback
import unittest
import numpy as np
import torch
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from paper_rq1.run import prepared,preflight
from paper_rq1.engine import step_inputs,cache_fingerprint
from paper_rq1_token_role_decode.engine import generate
from paper_rq1_token_role_decode.run import raw_save
from paper_rq1_token_role_decode.labels import align
from paper_rq1_single_layer_gqa6.run import atomic_json,tensor_hashes
from vlm_common import load_model,gpu_memory
from .protocol import ROOT,OUT,RAW,MODEL_PATH,GPUS,PROTOCOL
from .engine import step,partition,metrics
from .predictor import OnlinePredictor,category
from .generation import generate_policy,Policy
from .run import verify as verify_collector


def sources():
    return {str(ROOT/n):sha256(ROOT/n) for n in ['paper_rq1_function_attention_sparsify/quality_run.py',
        'paper_rq1_function_attention_sparsify/generation.py','paper_rq1_function_attention_sparsify/predictor.py',
        'test_function_generation.py']}


def freeze():
    verify_collector();classifier=json.loads((OUT/'predictor_freeze.json').read_text());gate=json.loads((OUT/'stage_a_gate.json').read_text())
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_function_generation','test_function_predictor'])
    with (OUT/'quality_cpu_tests.txt').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('Generation CPU validation failed')
    active=['S0','S1','S2'] if gate['dynamic_source_gate_passed'] else ['S0','S1']
    if classifier['dynamic_generation_allowed']:active+=['S3','S4']
    freeze_json(OUT/'quality_execution_manifest.json',dict(active_strategies=active,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        sources=sources(),classifier_sha256=sha256(OUT/'predictor_freeze.json'),stage_a_gate_sha256=sha256(OUT/'stage_a_gate.json'),
        collector_sha256=sha256(OUT/'execution_manifest.json'),model=MODEL_PATH,quality=PROTOCOL['quality'],
        classifier_gate=classifier['classifier_gate_passed'],source_gate=gate['dynamic_source_gate_passed'],
        holdout_not_read=True,stop_dynamic_if_gate_fails=True,cpu_tests=result.testsRun,latency=PROTOCOL['latency']))


def verify():
    verify_collector();r=json.loads((OUT/'quality_execution_manifest.json').read_text())
    if r['sources']!=sources() or sha256(OUT/'predictor_freeze.json')!=r['classifier_sha256'] or sha256(OUT/'execution_manifest.json')!=r['collector_sha256']:raise ValueError('Frozen policy sources/config changed')
    freeze_=json.loads((OUT/'predictor_freeze.json').read_text())
    if sha256(freeze_['parameters']['raw_path'])!=freeze_['parameters']['raw_sha256']:raise ValueError('Predictor artifact changed')
    return r


def predictor(device='cuda:0'):
    freeze_=json.loads((OUT/'predictor_freeze.json').read_text())
    if freeze_['selected'] is None:return None
    with np.load(freeze_['parameters']['raw_path']) as f:parameters={k:f[k] for k in f.files}
    selected=freeze_['selected'];return OnlinePredictor(parameters,selected['predictor'],selected['threshold'],device)


def smoke(model,processor,sample,chosen):
    prompt=prepared(model,processor,sample);hashes=tensor_hashes(prompt);footprint=hook_footprint(model);original=llama.eager_attention_forward
    native,raw=generate(model,prompt);full,observed=generate_policy(model,prompt,'S0',collect=True)
    if native['token_ids']!=full['token_ids']:raise AssertionError('Free Full vs validated native generation mismatch')
    np.testing.assert_array_equal(raw,observed['full_logits'])
    history=None;prefix=[];checks=[]
    for t,token in enumerate(native['token_ids']):
        inputs=step_inputs(prompt,prefix);base,future,_=step(model,inputs,history)
        np.testing.assert_array_equal(base.numpy(),raw[t])
        if t in [1,4]:
            groups=partition(prompt,t);fingerprint=cache_fingerprint(history)
            for strategy,mode in [('S1','No-Image'),('S2','BOS+History')]:
                expected,_,_=step(model,inputs,history,groups=groups,algorithm='A2',mode=mode,layer=31)
                from paper_rq1_token_role_decode.engine import owned_clone
                with torch.inference_mode(),Policy(model,prompt,strategy) as policy:
                    policy.next(prefix);output=model(**inputs,past_key_values=owned_clone(history),use_cache=True,return_dict=True)
                np.testing.assert_array_equal(expected.numpy(),output.logits[0,-1].detach().float().cpu().numpy())
                if cache_fingerprint(history)!=fingerprint:raise AssertionError('Policy smoke parent contaminated')
                checks.append(dict(t=t,strategy=strategy,single_step_matches_frozen_A2=True))
        prefix.append(token);history=future
    for strategy in ['S1','S2']+(['S3','S4'] if chosen is not None else []):
        result,_=generate_policy(model,prompt,strategy,chosen)
        ids=result['token_ids']
        for r in result['steps']:
            if r['t'] and r['consumed_token_id']!=ids[r['t']-1]:raise AssertionError('Not own-prefix generation')
        checks.append(dict(strategy=strategy,own_prefix_all_steps=True,length=result['length']))
    class Broken:
        def probability(self,*args):raise RuntimeError('intentional predictor failure')
    try:generate_policy(model,prompt,'S3',Broken())
    except RuntimeError as error:
        if str(error)!='intentional predictor failure':raise
    else:raise AssertionError('Missing exception cleanup exercise')
    if hook_footprint(model)!=footprint or llama.eager_attention_forward is not original or tensor_hashes(prompt)!=hashes:raise AssertionError('Leaked hooks/mutated input')
    return dict(passed=True,full_matches_native=True,checks=checks,exception_cleanup=True)


def risk(model,processor,sample,full,raw,chosen):
    prompt=prepared(model,processor,sample);tokens=full['token_ids'];ann=align(processor.tokenizer,tokens)
    history=None;prefix=[];rows=[];predictions=[];arrays=[];parameters=np.load(OUT/'predictor_parameters.npz')
    freezes=json.loads((OUT/'predictor_freeze.json').read_text());classifiers={k:OnlinePredictor(parameters,k,freezes['thresholds'][k],'cpu') for k in ['A','B','C']}
    for t,token in enumerate(tokens):
        inputs=step_inputs(prompt,prefix);base,future,_=step(model,inputs,history)
        np.testing.assert_array_equal(base.numpy(),raw['full_logits'][t])
        if t:
            label=category(ann,t);h=torch.from_numpy(raw['h31_in'][t]);ps={k:float(v.probability(prefix,h)) for k,v in classifiers.items()}
            predicted=chosen is not None and ps[chosen.kind]>=chosen.threshold
            shared=dict(image_id=sample['image_identity'],image_sha256=sample['image_sha256'],t=t,token_id=token,category=label,
                oracle_function=label=='Function',predicted_function=predicted,**{k+'_probability':v for k,v in ps.items()})
            predictions.append(shared)
            if label=='Function' or predicted:
                groups=partition(prompt,t);before=cache_fingerprint(history)
                for mode in ['No-Image','BOS+History']:
                    changed,_,_=step(model,inputs,history,groups=groups,algorithm='A2',mode=mode,layer=31)
                    rows.append(dict(shared,mode=mode,**metrics(base,changed)));arrays.append(changed.numpy())
                if cache_fingerprint(history)!=before:raise AssertionError('Risk replay cache contaminated')
        prefix.append(token);history=future
    return dict(rows=rows,predictions=predictions,oracle_is_offline_only=True),np.stack(arrays) if arrays else np.zeros((0,32064),dtype=np.float32)


def worker(stage,rank):
    config=verify();binding=digest(config);physical=rank+1
    if stage in ['quality','risk','latency']:
        for i in range(3):
            r=json.loads((OUT/f'quality_smoke/shard{i}/completion.json').read_text())
            if r['binding']!=binding or r['status']!='complete':raise ValueError('All policy smokes required')
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'resource_check.json',preflight(physical));model,processor,env=load_model(MODEL_PATH)
    atomic_json(folder/'environment.json',env);hooks=hook_footprint(model);original=llama.eager_attention_forward;versions={n:p._version for n,p in model.named_parameters()}
    manifest=json.loads((OUT/'sample_manifest.json').read_text());chosen=predictor()
    samples=[manifest['smoke']] if stage=='quality_smoke' else [s for s in manifest['holdout'] if s['rank']==rank]
    if stage=='latency':
        if rank!=0:raise ValueError('Latency is on one fixed physical GPU1 only')
        samples=manifest['holdout'][:10]
    started=time.perf_counter()
    for i,s in enumerate(samples):
        name=s['image_identity'].replace(':','_');path=folder/f'checkpoints/{name}.json'
        if path.exists():
            record=json.loads(path.read_text())
            if record['binding']!=binding or record['sample_sha256']!=digest(s):raise ValueError('Invalid policy resume')
            if record.get('raw_path') and sha256(record['raw_path'])!=record['raw_sha256']:raise ValueError('Changed raw')
            continue
        try:
            if stage=='quality_smoke':result=smoke(model,processor,s,chosen)
            elif stage=='quality':
                prompt=prepared(model,processor,s);runs={};raw={}
                for strategy in config['active_strategies']:
                    result,values=generate_policy(model,prompt,strategy,chosen,collect=strategy=='S0')
                    result['caption']=processor.tokenizer.decode(result['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    runs[strategy]=result
                    if strategy=='S0':raw=values
                result=dict(runs=runs,**raw_save(RAW/f'quality/shard{rank}/{name}.npz',**raw))
            elif stage=='risk':
                full=json.loads((OUT/f'quality/shard{rank}/checkpoints/{name}.json').read_text())
                if full['binding']!=binding or sha256(full['raw_path'])!=full['raw_sha256']:raise ValueError('Native holdout changed')
                with np.load(full['raw_path']) as f:raw={k:f[k] for k in f.files}
                result,changed=risk(model,processor,s,full['runs']['S0'],raw,chosen)
                result.update(raw_save(RAW/f'risk/shard{rank}/{name}.npz',changed_logits=changed))
            else:
                policies=[(k,'A2') for k in config['active_strategies']]+[('S2','A1')]
                prompt=prepared(model,processor,s)
                for strategy,implementation in policies:
                    for _ in range(3):generate_policy(model,prompt,strategy,chosen,implementation=implementation)
                rows=[];rng=np.random.default_rng(20261030+i)
                for repeat in range(5):
                    for j in rng.permutation(len(policies)):
                        strategy,implementation=policies[j]
                        torch.cuda.reset_peak_memory_stats()
                        r,_=generate_policy(model,prompt,strategy,chosen,implementation=implementation)
                        rows.append(dict(image_id=s['image_identity'],repeat=repeat,strategy=strategy,implementation=implementation,
                            wall_seconds=r['wall_seconds'],cuda_timeline_ms=r['cuda_timeline_ms'],prefill_cuda_ms=r['prefill_cuda_ms'],decode_cuda_ms=r['decode_cuda_ms'],
                            length=r['length'],tokens_per_second=r['length']/r['wall_seconds'],peak_mib=torch.cuda.max_memory_allocated()/2**20,
                            pruned_steps=sum(x['pruned'] for x in r['steps']),retained_keys=sum(x['retained_keys'] for x in r['steps'][1:]),full_keys=sum(x['full_keys'] for x in r['steps'][1:])))
                profiles=[]
                for strategy,implementation in policies:
                    r,_=generate_policy(model,prompt,strategy,chosen,implementation=implementation,profile=True)
                    profiles.extend(dict(strategy=strategy,implementation=implementation,**x) for x in r['profile'])
                # Separate instrumented microbenchmark; never use profiler timings
                # as the uninstrumented end-to-end speed measurement.
                if i==0:
                    from torch.profiler import profile,ProfilerActivity,record_function
                    import paper_rq1_function_attention_sparsify.generation as generation_module
                    original_sparse=generation_module.sparse_attention
                    def marked(*args,**kwargs):
                        with record_function('L31_TRUE_SPARSE_GATHER_QK_AV'):return original_sparse(*args,**kwargs)
                    operator_rows=[]
                    try:
                        generation_module.sparse_attention=marked
                        for strategy in ['S0','S1','S2']:
                            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
                                generate_policy(model,prompt,strategy,chosen)
                            for event in prof.key_averages():
                                if event.key in ['aten::index_select','aten::matmul','aten::bmm','aten::softmax','aten::_softmax','L31_TRUE_SPARSE_GATHER_QK_AV']:
                                    operator_rows.append(dict(strategy=strategy,operator=event.key,calls=event.count,self_cpu_us=event.self_cpu_time_total,
                                        self_cuda_us=event.self_device_time_total,total_cuda_us=event.device_time_total,
                                        scope='Entire Caption operator totals; index_select includes sparse gather and any native index_select. Instrumented, not speedup evidence.'))
                    finally:generation_module.sparse_attention=original_sparse
                    atomic_json(folder/'operator_profile.json',dict(rows=operator_rows,not_end_to_end_speed=True))
                result=dict(rows=rows,profiles=profiles,warmup=3,repeats=5,same_physical_gpu=1,model_only_latency_excludes_image_preprocessing=True)
            if hooks!=hook_footprint(model) or llama.eager_attention_forward is not original:raise AssertionError('Policy hook leakage')
            result.update(binding=binding,sample_sha256=digest(s),image_id=s['image_identity']);atomic_json(path,result)
            print(stage,physical,i+1,'/',len(samples),name,flush=True)
        except BaseException as e:
            atomic_json(folder/f'failure_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),binding=binding));raise
    if versions!={n:p._version for n,p in model.named_parameters()}:raise AssertionError('Model modified')
    verify();atomic_json(folder/'completion.json',dict(status='complete',binding=binding,images=len(samples),rank=rank,physical_gpu=physical,
        wall_seconds=time.perf_counter()-started,hooks_clean=True,weights_unchanged=True,memory=gpu_memory()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','quality_smoke','quality','risk','latency']);p.add_argument('--rank',type=int,choices=range(3),default=0);a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:worker(a.stage,a.rank)
