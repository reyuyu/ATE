import argparse,fcntl,time,traceback
import numpy as np
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from vlm_common import load_model,gpu_memory
from gqa3000_token_visual_writing.engine import generate
from gqa3000_token_visual_writing.common import verify as verify_history
from relay_source_causality.common import hardware
from semantic_writer_error_cases.run import smoke as historical_smoke
from .common import *
def freeze():
    import unittest
    from .tests import Audit
    r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));assert r.wasSuccessful()
    p=verify_history();assert read(OUT/'cpu_validation.json')['passed'];state=hardware();busy={l.split(',')[0].strip() for l in state['compute_processes'].splitlines()};idle=[]
    for l in state['table'].splitlines():
        i,u,n,used,free,util=[x.strip() for x in l.split(',')]
        if u not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(index=int(i),uuid=u,name=n))
    gpu=idle[0] if idle else None;state['selected']=gpu;atomic_json(OUT/'gpu_selection.json',state)
    atomic_json(OUT/'protocol.json',dict(base_commit=BASE,error_cases_commit='425286a2566cc6b0387460492a0f99566b142a38',sources=sources(),plan_sha256=bind(),parser_environment_sha256=sha256(OUT/'parser_environment.json'),rules_sha256=sha256(OUT/'selection_rules.json'),model_revision=REVISION,model_files=p['model_files'],transformers_sources=p['transformers_sources'],scorer_sources=p['scorer_sources'],gpu=gpu,smoke_ids=['00150746','11697551','11127723'],precision='float16',attention='eager',transformers='4.49.0',max_new_tokens=32,logits_to_keep=1,scope='All32 P-1/Decode imageblocked; only selected original Prefill R reads Image Key, hard mask, no o_proj ablation, dense native KV/MLP/residual',reuse='Exact history Semantic-k and All-Writer; identical writer sets reused per question after audit',bootstrap_repetitions=20000,bootstrap_seed=20261104,cpu_tests=r.testsRun))
    print('FROZEN GPU',gpu,flush=True)
def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources() and p['plan_sha256']==bind();verify_history();active();return p
def run():
    started=time.perf_counter();torch.set_num_threads(4);p=verify();assert p['gpu']
    state=hardware();u=p['gpu']['uuid'];fields=[x.strip() for x in next(l for l in state['table'].splitlines() if l.split(',')[1].strip()==u).split(',')]
    ok=not any(l.split(',')[0].strip()==u for l in state['compute_processes'].splitlines()) and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0
    atomic_json(OUT/'gpu_preflight.json',dict(**state,passed=ok,selected=p['gpu']));assert ok and os.environ.get('CUDA_VISIBLE_DEVICES')==u
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(OUT/'worker_started.json',dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),plan_sha256=bind()))
    model,proc,env=load_model(MODEL);atomic_json(OUT/'environment.json',dict(**env,pid=os.getpid(),gpu=p['gpu']))
    # Reuse the strict three-case smoke implementation, rebinding its output only.
    import semantic_writer_error_cases.run as oldrun
    oldrun.OUT=OUT;oldrun.RAW=RAW
    historical_smoke(model,proc,p)
    new=0;labels=0
    for qid,v in plan()['cases'].items():
        s=v['sample'];cache={};inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==v['input_hashes'];fp=hook_footprint(model)
        versions={n:x._version for n,x in model.named_parameters()}
        for label,w in v['writers'].items():
            if w is None:continue
            uid=digest(w)[:20];path=OUT/'checkpoints'/qid/(uid+'.json')
            if uid in cache:row=cache[uid]
            else:
                old=next((h for h in v['history'].values() if h['generation']['writer_positions']==w),None)
                if old is not None:
                    assert sha256(HIST/old['path'])==old['sha256'];row=dict(question_id=qid,writers=w,generation=old['generation'],reused=True,history=old)
                elif path.exists():
                    row=read(path);assert row['plan_sha256']==bind() and row['writers']==w and sha256(row['raw']['path'])==row['raw']['sha256']
                else:
                    g,f,t,a,_=generate(model,proc.tokenizer,inputs,w,capture=True);g['correct']=int(g['prediction']==s['answer'])
                    assert g['eos'] and not g['empty'] and not g['truncated'] and not g['repeated_trigram']
                    assert fp==hook_footprint(model) and tensor_hashes(inputs)==v['input_hashes'] and versions=={n:x._version for n,x in model.named_parameters()}
                    raw=atomic_npz(RAW/'main'/qid/(uid+'.npz'),first_logits=f,valid_logits=t,audit=a)
                    row=dict(question_id=qid,plan_sha256=bind(),writers=w,generation=g,raw=raw,input_hashes=v['input_hashes'],reused=False);atomic_json(path,row);new+=1
                cache[uid]=row
            atomic_json(OUT/'records'/qid/(label+'.json'),dict(condition=label,cohort=v['cohort'],**row));labels+=1
        print('DONE',qid,v['cohort'],'new',new,'labels',labels,'seconds',round(time.perf_counter()-started,1),flush=True)
    verify();atomic_json(OUT/'gpu_completion.json',dict(passed=True,pid=os.getpid(),new_generations=new,condition_labels=labels,seconds=time.perf_counter()-started,gpu=p['gpu'],memory=gpu_memory()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        if args.stage=='run':atomic_json(OUT/'STOP.json',dict(traceback=traceback.format_exc(),hardware=hardware()))
        raise
