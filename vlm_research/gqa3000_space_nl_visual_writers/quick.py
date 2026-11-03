"""Bounded quick24 plus four diagnostic cases; immutable parent token design."""
import argparse,collections,datetime,fcntl,io,os,subprocess,time,traceback,unittest
import numpy as np
import torch
from .common import *
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz,verify as verify_history
from gqa3000_token_visual_writing.engine import generate
from relay_source_causality.common import hardware
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from knockout_probe import model_layers
from vlm_common import load_model
Q=OUT/'quick24'
RAW=Path('/data/lk/ATE-resources/experiments/gqa3000_space_nl_visual_writers/quick24')
DIAG=['001020372','09662582','10763612','0160501']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def source_hashes():
    paths=[Path(__file__),ROOT/'gqa3000_token_visual_writing/engine.py',ROOT/'semantic_visual_writers/engine.py',ROOT/'semantic_writer_error_cases/run.py',ROOT/'query_role_smoke_fix/run.py',ROOT/'vlm_common.py',ROOT/'paper_rq1/scorers.py',Path(__file__).parent/'common.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}
def writecsv(name,rows):
    rows=list(rows);assert rows
    with (Q/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def freeze():
    from .tests import Audit
    Q.mkdir(exist_ok=True);assert not (Q/'protocol.json').exists()
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));(Q/'cpu_tests.log').write_text(stream.getvalue());assert tests.wasSuccessful()
    parent=read(OUT/'token_plan.json');cases=parent['cases'];cohort=read(OUT/'cohorts.json');chosen=[]
    for task in TASKS:
        pool=[q for q in cohort['pilot120'] if cases[q]['task']==task]
        chosen+=sorted(pool,key=lambda q:hashlib.sha256(('space-nl-quick24-v1|'+task+'|'+q+'|'+cases[q]['image_sha256']).encode()).hexdigest())[:4]
    assert len(chosen)==24 and not set(chosen)&set(DIAG)
    assert len({cases[q]['image_sha256'] for q in chosen+DIAG})==28
    samples={s['question_id']:s for s in read(OUT/'sample_manifest.json')['samples']};selected={q:dict(design=cases[q],sample=samples[q],cohort='quick24' if q in chosen else 'diagnostic4') for q in chosen+DIAG}
    histories={};new=0
    for q,v in selected.items():
        d=v['design'];histories[q]={}
        for uid,u in d['unique'].items():
            if u['reuse']:
                ref=u['reuse'];cp=read(ROOT/ref['path']);assert sha256(ROOT/ref['path'])==ref['sha256'];g=cp['generation'];assert g['writer_positions']==u['positions']
                assert g['prediction']==__import__('paper_rq1.scorers',fromlist=['gqa_prediction']).gqa_prediction(g['text']);assert g['correct']==int(g['prediction']==v['sample']['answer'])
                histories[q][uid]=dict(source=ref,generation=g)
            else:new+=1
        for label,w in d['writers'].items():
            assert w is not None and intact(w,d['groups']) and set(w)<=set(d['R'])
            if label in ['S+SPACE+NL','S+Template-Control-2']:assert len(set(w)-set(d['S']))==2
            if label in ['Noun+Article+SPACE+NL','N+D+Template-Control-2']:assert len(set(w)-set(d['ND']))==2
    assert new==168
    atomic_json(Q/'sample_plan.json',dict(selection='First4 per task by SHA256(space-nl-quick24-v1|task|qid|imageSHA) from the original pilot120; no outcome used. Four specified diagnostic cases separate.',main_ids=chosen,diagnostic_ids=DIAG,cases=selected,histories=histories))
    state=hardware();busy={l.split(',')[0].strip() for l in state['compute_processes'].splitlines()};idle=[]
    for row in state['table'].splitlines():
        i,u,n,used,free,util=[x.strip() for x in row.split(',')]
        if u not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(index=int(i),uuid=u,name=n))
    atomic_json(Q/'gpu_selection.json',dict(**state,selected=idle[:1],utc=stamp()))
    hist=verify_history()
    atomic_json(Q/'protocol.json',dict(status='FROZEN' if idle else 'STOP_NO_IDLE_GPU',gpu=idle[0] if idle else None,sources=source_hashes(),plan_sha256=sha256(Q/'sample_plan.json'),parent_plan_sha256=sha256(OUT/'token_plan.json'),parent_commit='ae20e6e9f18cae86eb54773b8890ae57d60e2fba',model_revision=REVISION,model_files=hist['model_files'],transformers_sources=hist['transformers_sources'],scorer_sources=hist['scorer_sources'],conditions=CONDITIONS,comparisons=COMPARISONS,main_n=24,diagnostic_n=4,new_generations=168,logical_labels=252,cpu_tests=tests.testsRun,CPU_quick_set_audit_passed=True,smoke_ids=['00150746','11697551','11127723'],bootstrap_repetitions=20000,bootstrap_seed=20261104,ordinary_percentiles=[2.5,97.5],five_comparison_Bonferroni_percentiles=[.5,99.5],notes='Quick exploratory feasibility/direction check;24 historically explored pilot questions, not independent confirmation. Four diagnostics excluded from main ACC. No NLL; no full3000 launched by this entry.',scope='FP16/Eager/Transformers4.49.0; greedy32; logits_to_keep=1; all32 P-1/Decode imageblocked; selected original Prefill R only; native dense QKV/KV/MLP/residual; no o_proj ablation.'))
    print('QUICK FROZEN',new,'GPU',idle[:1],flush=True)
def verify():
    p=read(Q/'protocol.json');assert p['sources']==source_hashes() and p['plan_sha256']==sha256(Q/'sample_plan.json') and p['parent_plan_sha256']==sha256(OUT/'token_plan.json');assert not (Q/'STOP.json').exists();return p
def run():
    start=time.perf_counter();torch.set_num_threads(4);p=verify();gpu=p['gpu'];assert gpu
    state=hardware();u=gpu['uuid'];row=next(l for l in state['table'].splitlines() if l.split(',')[1].strip()==u).split(',');busy=[l for l in state['compute_processes'].splitlines() if l.split(',')[0].strip()==u]
    ok=not busy and int(row[3])<200 and int(row[4])>22000 and int(row[5])==0;atomic_json(Q/'gpu_preflight.json',dict(**state,passed=ok,selected=gpu,utc=stamp()));assert ok and os.environ.get('CUDA_VISIBLE_DEVICES')==u
    lock=(Q/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip();atomic_json(Q/'worker_started.json',dict(pid=os.getpid(),utc=stamp(),code_commit=code))
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1
    atomic_json(Q/'environment.json',{**env,'pid':os.getpid(),'gpu':gpu,'code_commit':code})
    import semantic_writer_error_cases.run as oldrun
    oldrun.OUT=Q;oldrun.RAW=RAW;oldrun.smoke(model,proc,p)
    plan=read(Q/'sample_plan.json');count=0;reused=0;labels=0
    for q,v in plan['cases'].items():
        verify();s=v['sample'];d=v['design'];inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==d['input_hashes'];cache={}
        for uid,item in d['unique'].items():
            path=Q/'checkpoints'/q/(uid+'.json')
            if uid in plan['histories'][q]:
                h=plan['histories'][q][uid];assert sha256(ROOT/h['source']['path'])==h['source']['sha256'];r=dict(question_id=q,writers=item['positions'],generation=h['generation'],reused=True,source=h['source']);reused+=1
            elif path.exists():
                r=read(path);assert r['plan_sha256']==p['plan_sha256'] and r['writers']==item['positions'] and sha256(r['raw']['path'])==r['raw']['sha256'];count+=1
            else:
                fp=hook_footprint(model);versions={n:x._version for n,x in model.named_parameters()};calls=[0]*32;handles=[];visual=[]
                try:
                    for k,layer in enumerate(model_layers(model)):
                        def mlp(m,args,out,k=k):assert torch.isfinite(out).all();calls[k]+=1
                        handles.append(layer.mlp.register_forward_hook(mlp))
                    def projected(m,args,out):visual.append(tensor_hashes({'projected':out})['projected'])
                    handles.append(model.multi_modal_projector.register_forward_hook(projected))
                    g,f,t,a,_=generate(model,proc.tokenizer,inputs,item['positions'],capture=True)
                finally:
                    for handle in handles:handle.remove()
                assert calls==[g['length']]*32 and len(visual)==1 and fp==hook_footprint(model) and versions=={n:x._version for n,x in model.named_parameters()} and tensor_hashes(inputs)==d['input_hashes']
                g['correct']=int(g['prediction']==s['answer']);metadata=dict(question_id=q,plan_sha256=p['plan_sha256'],code_commit=code,writers=item['positions'],generation=g,input_hashes=d['input_hashes'],MLP_calls=calls,projected_visual_sha256=visual[0],weights_unchanged=True,hook_cleanup=True,reused=False)
                raw=atomic_npz(RAW/'main'/q/(uid+'.npz'),first_logits=f,valid_logits=t,audit=a,record_metadata=np.array(json.dumps(metadata,sort_keys=True)))
                r=dict(**metadata,raw=raw);atomic_json(path,r);count+=1
            cache[uid]=r
        projected={r['projected_visual_sha256'] for r in cache.values() if not r['reused']};assert len(projected)==1
        for label,uid in d['aliases'].items():atomic_json(Q/'records'/q/(label+'.json'),dict(condition=label,cohort=v['cohort'],**cache[uid]));labels+=1
        print('QUICK DONE',q,v['cohort'],'new',count,'labels',labels,'seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert count==168 and reused==84 and labels==252
    atomic_json(Q/'gpu_completion.json',dict(passed=True,pid=os.getpid(),new_generations=count,reused_generations=reused,labels=labels,main_n=24,diagnostic_n=4,seconds=time.perf_counter()-start,code_commit=code,utc=stamp()))
    print('QUICK GPU COMPLETE',count,'seconds',round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        if args.stage=='run':atomic_json(Q/'STOP.json',dict(passed=False,pid=os.getpid(),traceback=traceback.format_exc(),utc=stamp()))
        raise
