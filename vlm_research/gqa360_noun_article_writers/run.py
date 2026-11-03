"""Reuse frozen N/D positions and shared immutable engine; add only Noun+Article."""
import argparse,collections,datetime,fcntl,io,os,subprocess,sys,time,traceback,unittest,shutil
import numpy as np
import torch
from gqa3000_space_nl_visual_writers.common import *
from gqa3000_space_nl_visual_writers.failure_probe import csvout,idle_gpus,stamp
from gqa360_structural_receiver_selection.run import source_hashes as inherited_sources
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz,verify as verify_history
from gqa3000_token_visual_writing.engine import generate,cpu_mask
from relay_source_causality.common import hardware
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from knockout_probe import model_layers
from vlm_common import load_model

BASE=ROOT
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/gqa360_noun_article_writers'
RAW=Path('/data1/likai/ATE-resources/experiments/gqa360_noun_article_writers')
BASE_COMMIT='74afff563d255f23b0859f19def2de64accff109'
NAMES=['All-Writer','Semantic-k','Question-Only','S+SPACE+NL','Noun+Article']
def source_hashes():return {**{str(BASE/n):h for n,h in inherited_sources().items()},str(Path(__file__)):sha256(__file__),str(Path(__file__).with_name('__init__.py')):sha256(Path(__file__).with_name('__init__.py'))}

def freeze():
    from gqa3000_space_nl_visual_writers.tests import Audit
    P.mkdir(parents=True,exist_ok=True);assert not (P/'protocol.json').exists();torch.set_num_threads(4)
    git=GitHistory();old=git.load(BASE/'results/gqa360_structural_receiver_selection/sample_plan.json',BASE_COMMIT);parent=git.load(OUT/'token_plan.json',BASE_COMMIT);shards=git.load(HIST/'shards.json',SEM)['image_assignments']
    parser=git.load(OUT/'parser_environment.json',BASE_COMMIT);histories={};cases={};workernew=collections.Counter();totalreuse=0
    for q,v in old['cases'].items():
        s=v['sample'];d=parent['cases'][q];ND=sorted(set(d['N'])|set(d['D']));assert ND==d['ND']==d['writers']['Noun+Article'] and intact(ND,d['groups']) and set(ND)<=set(d['Q'])<=set(d['R'])
        spec=git.load(HIST/'token_manifests'/(s['task']+'__'+q+'.json'),SEM);match=[c for c,w in spec['writers'].items() if w==ND];histories[q]={}
        for c in NAMES[:4]:
            if c in old['histories'][q]:
                h=old['histories'][q][c];path=BASE/h['source']['path'];commit=h['source']['commit']
            else:path=BASE/f'results/gqa360_structural_receiver_selection/worker{v["worker"]}/records'/q/(c+'.json');commit=BASE_COMMIT
            cp=git.load(path,commit);g=cp['generation'];assert cp['writers']==g['writer_positions']==v['writers'][c] and g['correct']==int(gqa_prediction(g['text'])==s['answer'])
            raw=cp.get('raw') or cp.get('historical_raw');assert sha256(raw['path'])==raw['sha256']
            histories[q][c]=dict(source=dict(path=str(path),sha256=sha256(path),commit=commit),generation=g,raw=raw)
        if match:
            c=match[0];path=HIST/f'shard{shards[s["image_sha256"]]}/checkpoints'/(s['task']+'__'+q)/(spec['aliases'][c]+'.json');cp=git.load(path,SEM);g=cp['generation']
            assert cp['writers']==g['writer_positions']==ND and cp['input_hashes']==d['input_hashes'] and cp['image_sha256']==s['image_sha256'] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed'] and g['correct']==int(gqa_prediction(g['text'])==s['answer'])
            assert sha256(cp['raw']['path'])==cp['raw']['sha256'];histories[q]['Noun+Article']=dict(source=dict(path=str(path),sha256=sha256(path),commit=SEM,historical_labels=match),generation=g,raw=cp['raw']);totalreuse+=1
        else:workernew[v['worker']]+=1
        visualref=git.load(BASE/f'results/gqa360_structural_receiver_selection/worker{v["worker"]}/records'/q/'S+SPACE+NL.json',BASE_COMMIT)['projected_visual_sha256']
        cases[q]=dict(sample=s,design={k:v['design'][k] for k in ['input_hashes','groups','R','S','Q']},N=d['N'],D=d['D'],writers={**{c:v['writers'][c] for c in NAMES[:4]},'Noun+Article':ND},worker=v['worker'],stratum='unchanged_natural360',reference_visual_sha256=visualref)
        n=d['R'][-1]+2;layout=dict(prompt_length=n,I=list(range(5,581)),R=d['R'],A=[n-1],other=list(range(5)));a,b,m=cpu_mask(layout,ND)
        assert torch.equal(a[0,0][~m],b[0,0][~m]) and torch.isneginf(b[0,0][m]).all() and int(m.sum())==(len(d['R'])-len(ND)+1)*576 and m[n-1,layout['I']].all() and not m[layout['I']].any()
        a,b,m=cpu_mask(layout,ND,past=n);assert int(m.sum())==576 and torch.equal(a[0,0][~m],b[0,0][~m])
    assert len(cases)==360 and totalreuse==10 and sum(workernew.values())==350
    plan=dict(cases=cases,histories=histories,selection='Exact unchanged prior360; no resampling/POS rule changes; N+D only, no Semantic union or boundary addition.',parent_plan_sha256=sha256(BASE/'results/gqa360_structural_receiver_selection/sample_plan.json'),parser=parser)
    atomic_json(P/'sample_plan.json',plan)
    shutil.copyfile(BASE/'results/gqa360_structural_receiver_selection/sample_manifest.csv',P/'sample_manifest.csv')
    for i in range(2):atomic_json(P/f'worker{i}/sample_plan.json',dict(cases={q:v for q,v in cases.items() if v['worker']==i},histories={q:h for q,h in histories.items() if cases[q]['worker']==i}))
    git.close();atomic_json(P/'history_byte_audit.json',git.files)
    stream=io.StringIO();t=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));(P/'cpu_tests.log').write_text(stream.getvalue());assert t.wasSuccessful() and not torch.cuda.is_initialized()
    state=hardware();idle=idle_gpus(state);gpus=idle[:2] if len(idle)>=2 else [];atomic_json(P/'gpu_selection.json',dict(**state,selected=gpus,utc=stamp()))
    hist=verify_history();protocol=dict(status='FROZEN' if len(gpus)==2 else 'STOP_NO_TWO_IDLE_GPUS',base_commit=BASE_COMMIT,gpus=gpus,sources=source_hashes(),plan_sha256=sha256(P/'sample_plan.json'),manifest_sha256=sha256(P/'sample_manifest.csv'),shard_plan_sha256={str(i):sha256(P/f'worker{i}/sample_plan.json') for i in range(2)},model_revision=REVISION,model_files=hist['model_files'],transformers_sources=hist['transformers_sources'],scorer_sources=hist['scorer_sources'],conditions=NAMES,new_condition_only='Noun+Article = frozen N union D',new_generations=350,history_reuse=1450,ND_history_reuse=10,worker_new={str(i):workernew[i] for i in range(2)},worker_questions={'0':180,'1':180},smoke_ids=['00150746','11697551','11127723'],bootstrap_repetitions=20000,bootstrap_seed=20261104,CPU_tests=t.testsRun,actual_mask_and_frozen_groups_passed=True,scope='Unchanged LLaVA FP16 Eager4.49.0 greedy32/logits_to_keep1; all32 P-1/Decode imageblocked; original native QKV/MLP/Residual/fullKV; independent true cache; only ND R writers open.',shared_source_worktree=str(BASE.parent),raw_directory=str(RAW),storage_note='Full checkout on data failed before GPU; sparse independent checkout and raw arrays on data1; old code/results read only.',utc=stamp())
    atomic_json(P/'protocol.json',protocol);print('FROZEN ND360',350,'new,10 ND reused',dict(workernew),'GPUS',gpus,flush=True)
    if len(gpus)!=2:atomic_json(P/'completion.json',dict(status='STOP_NO_TWO_IDLE_GPUS'))

def verify():
    p=read(P/'protocol.json');assert p['status']=='FROZEN' and p['sources']==source_hashes() and not (P/'STOP.json').exists()
    assert p['plan_sha256']==sha256(P/'sample_plan.json') and p['manifest_sha256']==sha256(P/'sample_manifest.csv')
    for i,h in p['shard_plan_sha256'].items():assert sha256(P/f'worker{i}/sample_plan.json')==h
    return p

def smoke(model,proc,p,W,rank):
    import semantic_writer_error_cases.run as oldrun
    oldrun.OUT=W;oldrun.RAW=RAW/f'worker{rank}';oldrun.smoke(model,proc,p)
    case=next(iter(read(W/'sample_plan.json')['cases'].values()));inputs=prepared(model,proc,case['sample']);g,f,t,a,_=generate(model,proc.tokenizer,inputs,case['writers']['Noun+Article'],capture=True)
    assert g['audit']['passed'] and g['writer_positions']==case['writers']['Noun+Article'] and tensor_hashes(inputs)==case['design']['input_hashes']
    raw=atomic_npz(RAW/f'worker{rank}'/'smoke_ND.npz',first_logits=f,valid_logits=t,audit=a);atomic_json(W/'structural_smoke_validation.json',dict(passed=True,ND_generation=g,raw=raw))

def worker(rank):
    start=time.perf_counter();torch.set_num_threads(4);p=verify();gpu=p['gpus'][rank];W=P/f'worker{rank}';W.mkdir(exist_ok=True)
    lock=(W/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=hardware();ok=gpu in idle_gpus(state);atomic_json(W/'gpu_preflight.json',dict(**state,passed=ok,selected=gpu,utc=stamp()));assert ok and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip();atomic_json(W/'worker_started.json',dict(pid=os.getpid(),utc=stamp(),code_commit=code,worker=rank))
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1
    atomic_json(W/'environment.json',{**env,'pid':os.getpid(),'gpu':gpu,'code_commit':code})
    if not (W/'structural_smoke_validation.json').exists():smoke(model,proc,p,W,rank)
    assert read(W/'smoke_validation.json')['passed'] and read(W/'structural_smoke_validation.json')['passed'];atomic_json(W/'READY.json',dict(worker=rank,plan_sha256=p['plan_sha256'],passed=True))
    deadline=time.monotonic()+600
    while not (P/'GO.json').exists():
        verify();assert time.monotonic()<deadline,'Other worker smoke barrier timed out';time.sleep(1)
    assert read(P/'GO.json')['plan_sha256']==p['plan_sha256']
    plan=read(W/'sample_plan.json');cases=plan['cases'];assert all(v['worker']==rank for v in cases.values());count=0;reused=0;labels=0
    for q,v in cases.items():
        verify();s=v['sample'];d=v['design'];inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==d['input_hashes'];visual=set()
        for label in NAMES:
            writers=v['writers'][label];uid=digest(writers);path=W/'checkpoints'/q/(uid+'.json')
            if label in plan['histories'][q]:
                h=plan['histories'][q][label];assert sha256(ROOT/h['source']['path'])==h['source']['sha256'];r=dict(question_id=q,writers=writers,generation=h['generation'],reused=True,source=h['source'],historical_raw=h['raw']);reused+=1
            elif path.exists():
                r=read(path);assert r['plan_sha256']==p['plan_sha256'] and r['writers']==writers and sha256(r['raw']['path'])==r['raw']['sha256'];count+=1;visual.add(r['projected_visual_sha256'])
            else:
                fp=hook_footprint(model);versions={n:x._version for n,x in model.named_parameters()};calls=[0]*32;handles=[];features=[]
                try:
                    for k,layer in enumerate(model_layers(model)):
                        def mlp(m,args,out,k=k):assert torch.isfinite(out).all();calls[k]+=1
                        handles.append(layer.mlp.register_forward_hook(mlp))
                    def projected(m,args,out):features.append(tensor_hashes({'projected':out})['projected'])
                    handles.append(model.multi_modal_projector.register_forward_hook(projected))
                    g,f,t,a,_=generate(model,proc.tokenizer,inputs,writers,capture=True)
                finally:
                    for handle in handles:handle.remove()
                assert calls==[g['length']]*32 and len(features)==1 and fp==hook_footprint(model) and versions=={n:x._version for n,x in model.named_parameters()} and tensor_hashes(inputs)==d['input_hashes']
                assert g['writer_positions']==writers and g['own_cache'] and g['own_prefix'] and g['audit']['passed']
                g['correct']=int(gqa_prediction(g['text'])==s['answer']);metadata=dict(question_id=q,worker=rank,plan_sha256=p['plan_sha256'],code_commit=code,writers=writers,generation=g,input_hashes=d['input_hashes'],MLP_calls=calls,projected_visual_sha256=features[0],weights_unchanged=True,hook_cleanup=True,reused=False)
                raw=atomic_npz(RAW/f'worker{rank}'/'main'/q/(uid+'.npz'),first_logits=f,valid_logits=t,audit=a,record_metadata=np.array(json.dumps(metadata,sort_keys=True)))
                r=dict(**metadata,raw=raw);atomic_json(path,r);count+=1;visual.add(features[0])
            atomic_json(W/'records'/q/(label+'.json'),dict(condition=label,stratum=v['stratum'],**r));labels+=1
        assert len(visual)<=1
        if visual:assert visual=={v['reference_visual_sha256']}
        print('CASE DONE',q,v['stratum'],'new',count,'labels',labels,'seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert count==p['worker_new'][str(rank)] and count+reused==5*len(cases) and labels==5*len(cases)
    atomic_json(W/'completion.json',dict(passed=True,worker=rank,pid=os.getpid(),questions=len(cases),new_generations=count,reused_generations=reused,labels=labels,seconds=time.perf_counter()-start,code_commit=code,gpu=gpu,utc=stamp()))
    print('WORKER COMPLETE',rank,count,flush=True)

def driver():
    p=verify();start=time.perf_counter();lock=(P/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);children=[];logs=[]
    atomic_json(P/'driver_started.json',dict(pid=os.getpid(),utc=stamp(),plan_sha256=p['plan_sha256']))
    try:
        for i,gpu in enumerate(p['gpus']):
            w=P/f'worker{i}';w.mkdir(exist_ok=True);log=(w/'run.log').open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONUNBUFFERED':'1'}
            children.append(subprocess.Popen([sys.executable,'-m','gqa360_noun_article_writers.run','worker','--rank',str(i)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(c.poll() is None for c in children):
            if (P/'STOP.json').exists() or any(c.poll() not in [None,0] for c in children):raise RuntimeError('Worker failed; stop only this driver\'s children')
            if not (P/'GO.json').exists() and all((P/f'worker{i}/READY.json').exists() for i in range(2)):
                assert all(read(P/f'worker{i}/smoke_validation.json')['passed'] for i in range(2));atomic_json(P/'GO.json',dict(plan_sha256=p['plan_sha256'],both_smokes_passed=True,utc=stamp()));print('BOTH SMOKES PASSED; OPEN MAIN GATE',flush=True)
            time.sleep(2)
        assert all(c.returncode==0 for c in children)
        for i in range(2):
            done=read(P/f'worker{i}/completion.json');assert done['passed'];atomic_json(P/f'worker{i}_completion.json',done)
        verify();atomic_json(P/'gpu_completion.json',dict(passed=True,seconds=time.perf_counter()-start,new_generations=p['new_generations'],history_reuse=p['history_reuse'],utc=stamp()))
    except BaseException:
        if not (P/'STOP.json').exists():atomic_json(P/'STOP.json',dict(passed=False,utc=stamp(),traceback=traceback.format_exc()))
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        raise
    finally:
        for log in logs:log.close()
        state=hardware();uuids={g['uuid'] for g in p['gpus']};remaining=[l for l in state['compute_processes'].splitlines() if l.split(',')[0].strip() in uuids]
        atomic_json(P/'gpu_release.json',dict(**state,utc=stamp(),all_own_children_exited=all(c.poll() is not None for c in children),selected_devices_no_compute_processes=not remaining,selected_uuids=sorted(uuids)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','driver','worker']);a.add_argument('--rank',type=int,choices=[0,1]);args=a.parse_args()
    try:
        if args.stage=='worker':worker(args.rank)
        else:globals()[args.stage]()
    except BaseException:
        if args.stage=='worker' and not (P/'STOP.json').exists():atomic_json(P/'STOP.json',dict(passed=False,worker=args.rank,pid=os.getpid(),traceback=traceback.format_exc(),utc=stamp()))
        raise
