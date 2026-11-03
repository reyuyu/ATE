"""Frozen historical-failure sampling and two isolated, bounded GPU workers."""
import argparse,collections,datetime,fcntl,io,os,subprocess,sys,time,traceback,unittest
import numpy as np
import torch
from .common import *
from .quick import source_hashes as parent_sources
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz,verify as verify_history
from gqa3000_token_visual_writing.engine import generate,cpu_mask
from relay_source_causality.common import hardware
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from knockout_probe import model_layers
from vlm_common import load_model

BASE_COMMIT='f2fb54df59a1a04bbfc1a59403df4c1155976d79'
P=OUT/'failure_probe144'
RAW=Path('/data/lk/ATE-resources/experiments/gqa3000_space_nl_visual_writers/failure_probe144')
NAMES=['All-Writer','Semantic-k','S+SPACE','S+NL','S+SPACE+NL','S+Template-Control-1','S+Template-Control-2']
PAIRS=[['S+NL','S+SPACE'],['S+NL','S+Template-Control-1'],['S+SPACE','S+Template-Control-1'],['S+SPACE+NL','S+Template-Control-2'],['S+SPACE+NL','S+NL']]

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def source_hashes():return {**parent_sources(),str(Path(__file__).relative_to(ROOT)):sha256(__file__)}
def csvout(path,rows,fields=None):
    rows=list(rows);fields=fields or list(rows[0]);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),path
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def rank_hash(task,stratum,q,image):return hashlib.sha256(f'space-nl-failure-recovery-v1|{task}|{stratum}|{q}|{image}'.encode()).hexdigest()
def idle_gpus(state):
    busy={l.split(',')[0].strip() for l in state['compute_processes'].splitlines()};idle=[]
    for row in state['table'].splitlines():
        i,u,n,used,free,util=[x.strip() for x in row.split(',')]
        if u not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(index=int(i),uuid=u,name=n))
    return idle

def freeze():
    from .tests import Audit
    from transformers import AutoTokenizer
    P.mkdir(exist_ok=True);assert not (P/'protocol.json').exists()
    git=GitHistory()
    parent=git.load(OUT/'token_plan.json',BASE_COMMIT);cohorts=git.load(OUT/'cohorts.json',BASE_COMMIT)
    samples={s['question_id']:s for s in git.load(OUT/'sample_manifest.json',BASE_COMMIT)['samples']}
    quick=git.load(OUT/'quick24/sample_plan.json',BASE_COMMIT)
    baseline=list(csv.DictReader(io.StringIO(git.bytes(OUT/'audited_historical_baselines.csv',BASE_COMMIT).decode())))
    bs={(r['question_id'],r['condition']):r for r in baseline};assert len(bs)==6000
    counts=collections.Counter();strata={}
    for q in samples:
        s=int(bs[q,'Semantic-k']['correct']);a=int(bs[q,'All-Writer']['correct'])
        strata[q]='C' if s else 'A' if a else 'B';counts[samples[q]['task'],strata[q]]+=1
    assert sum(int(bs[q,'Semantic-k']['correct']) for q in samples)==2514
    print('RAW CANDIDATE COUNTS',sorted((t,st,n) for (t,st),n in counts.items()),flush=True)
    hardq=set(quick['cases']);hardim={v['sample']['image_sha256'] for v in quick['cases'].values()}
    prefq=set(cohorts['pilot120'])|set(cohorts['historical12']);prefim={samples[q]['image_sha256'] for q in prefq}
    used=set();selected={};sampling=[]
    # Fixed allocation order resolves cross-task/image conflicts before any new output.
    for t in TASKS:
        for st,target in [('A',10),('B',10),('C',4)]:
            raw=[q for q in samples if samples[q]['task']==t and strata[q]==st]
            eligible=[q for q in raw if q not in hardq and samples[q]['image_sha256'] not in hardim]
            pref=lambda q:q in prefq or samples[q]['image_sha256'] in prefim
            ordered=sorted(eligible,key=lambda q:(pref(q),rank_hash(t,st,q,samples[q]['image_sha256'])))
            chosen=[]
            for q in ordered:
                im=samples[q]['image_sha256']
                if im in used:continue
                chosen.append(q);used.add(im)
                selected[q]=dict(sample=samples[q],design=parent['cases'][q],stratum=st,selection_sha256=rank_hash(t,st,q,im),preference_tier=int(pref(q)))
                if len(chosen)==target:break
            sampling.append(dict(task=t,stratum=st,target=target,raw_candidates=len(raw),after_hard_exclusion=len(eligible),preferred_candidates=sum(not pref(q) for q in eligible),selected=len(chosen),deficit=target-len(chosen),preference_fallback=sum(pref(q) for q in chosen)))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);histories={};rows=[];token_audit=[]
    for q,v in selected.items():
        s=v['sample'];d=v['design'];ids=s['input_mapping']['prompt_ids'];S=d['S'];R=d['R'];control=d['template_control']
        assert d['B']==[581,582] and ids[581:583]==[29871,13] and len(control)==2
        assert not set(S)&set([581,582]+control) and set([581,582]+control)<=set(R)
        assert tok.decode([ids[control[0]]]).strip()=='Answer' and tok.decode([ids[control[1]]]).strip()=='the'
        groups={p:g for g in d['groups'] for p in g['positions']}
        assert all(len(groups[p]['positions'])==1 and not groups[p]['question'] for p in control)
        writers={NAMES[0]:R,NAMES[1]:S,NAMES[2]:sorted(S+[581]),NAMES[3]:sorted(S+[582]),NAMES[4]:sorted(S+[581,582]),NAMES[5]:sorted(S+[control[0]]),NAMES[6]:sorted(S+control)}
        v['writers']=writers;histories[q]={}
        for label in NAMES[:2]:
            ref=d['history'][label];cp=git.load(ROOT/ref['path'],ref['commit']);g=cp['generation']
            assert sha256(ROOT/ref['path'])==ref['sha256'] and g['writer_positions']==writers[label]
            assert g['correct']==int(gqa_prediction(g['text'])==s['answer'])==int(bs[q,label]['correct'])
            assert g['token_ids']==json.loads(bs[q,label]['token_ids']) and g['text']==bs[q,label]['answer']
            assert sha256(cp['raw']['path'])==cp['raw']['sha256']
            with np.load(cp['raw']['path']) as z:assert int(z['first_logits'].argmax())==g['token_ids'][0]
            histories[q][label]=dict(source=ref,generation=g,raw=cp['raw'])
        v['design']={k:d[k] for k in ['S','R','Q','B','template_control','input_hashes','groups','prompt_chat_sha256','prompt_ids_sha256']}
        token_audit.append(dict(question_id=q,positions=[581,582]+control,ids=[ids[p] for p in [581,582]+control],pieces=tok.convert_ids_to_tokens([ids[p] for p in [581,582]+control]),decoded=[tok.decode([ids[p]]) for p in [581,582]+control],offsets=[groups[p]['offsets'] for p in [581,582]+control]))
        rows.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],stratum=v['stratum'],question=s['question'],gold=s['answer'],All_answer=histories[q][NAMES[0]]['generation']['text'],S_answer=histories[q][NAMES[1]]['generation']['text'],All_correct=int(bs[q,NAMES[0]]['correct']),S_correct=int(bs[q,NAMES[1]]['correct']),selection_sha256=v['selection_sha256'],preference_tier=v['preference_tier']))
    assignments={v['sample']['image_sha256']:i%2 for i,(q,v) in enumerate(sorted(selected.items(),key=lambda kv:kv[1]['sample']['image_sha256']))}
    for v in selected.values():v['worker']=assignments[v['sample']['image_sha256']]
    plan=dict(selection='Fixed task order TASKS, strata A/B/C; prefer untested pilot/history image tier, then ascending prescribed SHA; one global unique-image set. Never fill across strata.',sampling=sampling,cases=selected,histories=histories,hard_excluded_ids=sorted(hardq),hard_excluded_images=sorted(hardim),preferred_excluded_ids=sorted(prefq),preferred_excluded_images=sorted(prefim),image_assignments=assignments,smoke_inputs_are_validation_only='Each worker separately replays the same three frozen smoke IDs; scientific shards are disjoint.',conditions=NAMES,comparisons=PAIRS)
    csvout(P/'sample_manifest.csv',rows);csvout(P/'candidate_counts.csv',sampling)
    csvout(P/'historical_S_errors.csv',[dict(question_id=q,task=samples[q]['task'],image_sha256=samples[q]['image_sha256'],stratum=strata[q],gold=samples[q]['answer'],All_answer=bs[q,'All-Writer']['answer'],S_answer=bs[q,'Semantic-k']['answer']) for q in sorted(samples) if strata[q]!='C'])
    atomic_json(P/'sample_plan.json',plan);atomic_json(P/'token_audit.json',token_audit)
    git.close();atomic_json(P/'history_byte_audit.json',git.files)
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));(P/'cpu_tests.log').write_text(stream.getvalue());assert tests.wasSuccessful()
    extra_audit(plan);assert not torch.cuda.is_initialized()
    state=hardware();idle=idle_gpus(state);gpus=idle[:2] if len(idle)>=2 else []
    atomic_json(P/'gpu_selection.json',dict(**state,selected=gpus,utc=stamp(),criterion='No compute process; utilization=0; used<200MiB; free>22000MiB'))
    hist=verify_history()
    protocol=dict(status='FROZEN' if len(gpus)==2 else 'STOP_NO_TWO_IDLE_GPUS',base_commit=BASE_COMMIT,gpus=gpus,sources=source_hashes(),plan_sha256=sha256(P/'sample_plan.json'),manifest_sha256=sha256(P/'sample_manifest.csv'),parent_plan_sha256=sha256(OUT/'token_plan.json'),model_revision=REVISION,model_files=hist['model_files'],transformers_sources=hist['transformers_sources'],scorer_sources=hist['scorer_sources'],conditions=NAMES,comparisons=PAIRS,questions=len(selected),stratum_counts=dict(collections.Counter(v['stratum'] for v in selected.values())),new_generations=5*len(selected),history_reuse=2*len(selected),logical_labels=7*len(selected),worker_questions={str(i):sum(v['worker']==i for v in selected.values()) for i in range(2)},cpu_tests=tests.testsRun,extra_CPU_audit_passed=True,smoke_ids=['00150746','11697551','11127723'],bootstrap_repetitions=20000,bootstrap_seed=20261104,primary_family='Five fixed paired comparisons in combined A+B only; ordinary95% and Bonferroni99% intervals; A/B/task/C exploratory.',scope='FP16/Eager/Transformers4.49.0; greedy32; logits_to_keep=1; all32 P-1/Decode imageblocked; selected original Prefill R only; native dense QKV/KV/MLP/residual; no o_proj ablation.',no_full3000=True,utc=stamp())
    atomic_json(P/'protocol.json',protocol)
    print('FROZEN ACTUAL',protocol['stratum_counts'],'NEW',protocol['new_generations'],'GPUS',gpus,flush=True)
    print('SAMPLING',json.dumps(sampling),flush=True)
    if len(gpus)!=2:atomic_json(P/'completion.json',dict(status='STOP_NO_TWO_IDLE_GPUS',new_generations=0));return

def extra_audit(plan):
    cases=plan['cases'];assert len(cases)==len({v['sample']['image_sha256'] for v in cases.values()})
    assert not set(cases)&set(plan['hard_excluded_ids']);assert not set(plan['image_assignments'])&set(plan['hard_excluded_images'])
    assert abs(sum(v['worker']==0 for v in cases.values())-sum(v['worker']==1 for v in cases.values()))<=1
    for q,v in cases.items():
        d=v['design'];S=set(d['S']);n=d['R'][-1]+2;layout=dict(prompt_length=n,I=list(range(5,581)),R=d['R'],A=[n-1],other=list(range(5)))
        assert v['selection_sha256']==rank_hash(v['sample']['task'],v['stratum'],q,v['sample']['image_sha256'])
        for j,c in enumerate(NAMES):
            w=v['writers'][c];assert intact(w,d['groups']) and set(w)<=set(d['R'])
            if j>=2:assert len(set(w)-S)==(2 if j in [4,6] else 1) and S<=set(w)
        for c in NAMES[2:]:
            w=v['writers'][c];native,changed,blocked=cpu_mask(layout,w)
            assert torch.equal(changed[0,0][~blocked],native[0,0][~blocked]) and torch.isneginf(changed[0,0][blocked]).all()
            assert int(blocked.sum())==(len(d['R'])-len(w)+1)*576 and not blocked[layout['I']].any()
            assert blocked[n-1,layout['I']].all()
            a,b,m=cpu_mask(layout,w,past=n);assert int(m.sum())==576 and torch.equal(b[0,0][~m],a[0,0][~m])
    assert len(PAIRS)==5

def verify():
    p=read(P/'protocol.json');assert p['status']=='FROZEN' and len(p['gpus'])==2
    assert p['sources']==source_hashes() and p['plan_sha256']==sha256(P/'sample_plan.json') and p['manifest_sha256']==sha256(P/'sample_manifest.csv') and p['parent_plan_sha256']==sha256(OUT/'token_plan.json')
    assert not (P/'STOP.json').exists();return p

def worker(rank):
    start=time.perf_counter();torch.set_num_threads(4);p=verify();gpu=p['gpus'][rank];W=P/f'worker{rank}';W.mkdir(exist_ok=True)
    lock=(W/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=hardware();ok=gpu in idle_gpus(state);atomic_json(W/'gpu_preflight.json',dict(**state,passed=ok,selected=gpu,utc=stamp()));assert ok and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip();atomic_json(W/'worker_started.json',dict(pid=os.getpid(),utc=stamp(),code_commit=code,worker=rank))
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1
    atomic_json(W/'environment.json',{**env,'pid':os.getpid(),'gpu':gpu,'code_commit':code})
    import semantic_writer_error_cases.run as oldrun
    oldrun.OUT=W;oldrun.RAW=RAW/f'worker{rank}'
    if not (W/'smoke_validation.json').exists():oldrun.smoke(model,proc,p)
    assert read(W/'smoke_validation.json')['passed'];atomic_json(W/'READY.json',dict(worker=rank,plan_sha256=p['plan_sha256'],passed=True))
    deadline=time.monotonic()+600
    while not (P/'GO.json').exists():
        verify();assert time.monotonic()<deadline,'Other worker smoke barrier timed out';time.sleep(1)
    assert read(P/'GO.json')['plan_sha256']==p['plan_sha256']
    plan=read(P/'sample_plan.json');cases={q:v for q,v in plan['cases'].items() if v['worker']==rank};count=0;reused=0;labels=0
    for q,v in cases.items():
        verify();s=v['sample'];d=v['design'];inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==d['input_hashes'];visual=set()
        for label in NAMES:
            writers=v['writers'][label];uid=digest(writers);path=W/'checkpoints'/q/(uid+'.json')
            if label in NAMES[:2]:
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
        assert len(visual)==1
        print('CASE DONE',q,v['stratum'],'new',count,'labels',labels,'seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert count==5*len(cases) and reused==2*len(cases) and labels==7*len(cases)
    atomic_json(W/'completion.json',dict(passed=True,worker=rank,pid=os.getpid(),questions=len(cases),new_generations=count,reused_generations=reused,labels=labels,seconds=time.perf_counter()-start,code_commit=code,gpu=gpu,utc=stamp()))
    print('WORKER COMPLETE',rank,count,flush=True)

def driver():
    p=verify();start=time.perf_counter();lock=(P/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);children=[];logs=[]
    atomic_json(P/'driver_started.json',dict(pid=os.getpid(),utc=stamp(),plan_sha256=p['plan_sha256']))
    try:
        for i,gpu in enumerate(p['gpus']):
            w=P/f'worker{i}';w.mkdir(exist_ok=True);log=(w/'run.log').open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONUNBUFFERED':'1'}
            children.append(subprocess.Popen([sys.executable,'-m','gqa3000_space_nl_visual_writers.failure_probe','worker','--rank',str(i)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT))
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
