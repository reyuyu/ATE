"""Input-only structural selection; exact historical baselines; two GPU workers."""
import argparse,collections,datetime,fcntl,io,os,subprocess,sys,time,traceback,unittest
import numpy as np
import torch
from gqa3000_space_nl_visual_writers.common import *
from gqa3000_space_nl_visual_writers.failure_probe import csvout,idle_gpus,stamp
from gqa3000_space_nl_visual_writers.failure_probe import source_hashes as inherited_sources
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz,verify as verify_history
from gqa3000_token_visual_writing.engine import generate,cpu_mask
from relay_source_causality.common import hardware
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from knockout_probe import model_layers
from vlm_common import load_model

BASE_COMMIT='1159c2197a6611fa236b520b4653424eaf192125'
P=ROOT/'results/gqa360_structural_receiver_selection'
RAW=Path('/data/lk/ATE-resources/experiments/gqa360_structural_receiver_selection')
NAMES=['All-Writer','Semantic-k','Question-Only','Q+SPACE+NL','Q+PostQuestion-2','S+QuestionTail-2','S+SPACE+NL']
PAIRS=[[NAMES[3],NAMES[2]],[NAMES[4],NAMES[2]],[NAMES[3],NAMES[4]],[NAMES[5],NAMES[6]],[NAMES[6],NAMES[1]]]
HISTORY_LABELS=['All-Writer','Semantic-k','Semantic-first-Q100']

def source_hashes():
    return {**inherited_sources(),str(Path(__file__).relative_to(ROOT)):sha256(__file__),str(Path(__file__).with_name('__init__.py').relative_to(ROOT)):sha256(Path(__file__).with_name('__init__.py'))}

def token_design(s,d,tok,proc):
    """Validate native IDs and frozen offsets; never choose by model outputs."""
    chat=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
    assert digest(chat)==d['prompt_chat_sha256']==s['input_mapping']['chat_prompt_sha256']
    expanded=chat.replace('<image>','<image>'*576);enc=tok(expanded,return_offsets_mapping=True,add_special_tokens=True)
    ids=s['input_mapping']['prompt_ids'];assert enc['input_ids']==ids and digest(ids)==d['prompt_ids_sha256']
    offsets=enc['offset_mapping'];start=d['question_char_start'];end=start+len(s['question']);assert expanded[start:end]==s['question']
    frozen={p:g for g in d['groups'] for p in g['positions']}
    assert sorted(frozen)==d['R']
    for g in d['groups']:assert [list(offsets[p]) for p in g['positions']]==g['offsets']
    Q=[p for p in d['R'] if offsets[p][0]<end and offsets[p][1]>start];assert Q==d['Q']
    assert Q==[p for g in d['groups'] if g['question'] for p in g['positions']]
    final=Q[-1];post=final+1;answer=final+2;B=d['B'];S=d['S'];R=d['R'];reasons=[]
    if B!=[581,582] or ids[581:583]!=[29871,13]:reasons.append('pre_boundary_native_ID_or_position_mismatch')
    if not set(B+[final,post,answer])<=set(R):reasons.append('structural_position_outside_R')
    if list(offsets[final])[1]!=end:reasons.append('question_final_offset_not_at_question_end')
    if tuple(offsets[post])!=(end,end+1) or ids[post]!=13 or expanded[end:end+1]!='\n':reasons.append('post_question_not_native_single_NL')
    if tuple(offsets[answer])!=(end+1,end+7) or tok.decode([ids[answer]]).strip()!='Answer' or frozen[answer]['positions']!=[answer]:reasons.append('Answer_not_adjacent_single_native_token')
    if set([final,post])&set(S):reasons.append('QuestionTail_already_selected_by_S')
    writers={NAMES[0]:R,NAMES[1]:S,NAMES[2]:Q,NAMES[3]:sorted(set(Q)|set(B)),NAMES[4]:sorted(set(Q)|{post,answer}),NAMES[5]:sorted(set(S)|{final,post}),NAMES[6]:sorted(set(S)|set(B))}
    for c,base in [(NAMES[3],Q),(NAMES[4],Q),(NAMES[5],S),(NAMES[6],S)]:
        if len(set(writers[c])-set(base))!=2:reasons.append(c+'_budget_not_two')
        if not intact(writers[c],d['groups']):reasons.append(c+'_breaks_frozen_group')
    positions=dict(SPACE=581,NL=582,Question_final=final,PostQuestion_NL=post,Instruction_Answer=answer)
    audit=[dict(question_id=s['question_id'],task=s['task'],role=role,position=p,token_id=ids[p],piece=tok.convert_ids_to_tokens(ids[p]),decoded=tok.decode([ids[p]]),offset_start=offsets[p][0],offset_end=offsets[p][1],original_text=expanded[offsets[p][0]:offsets[p][1]],in_R=p in R,in_Q=p in Q,in_S=p in S,question_char_start=start,question_char_end=end) for role,p in positions.items()]
    return dict(writers=writers,positions=positions,feasible=not reasons,reasons=reasons),audit

def freeze():
    from transformers import AutoProcessor
    from gqa3000_space_nl_visual_writers.tests import Audit
    P.mkdir(exist_ok=True);assert not (P/'protocol.json').exists();torch.set_num_threads(4)
    git=GitHistory();parent=git.load(OUT/'token_plan.json',BASE_COMMIT);cohorts=git.load(OUT/'cohorts.json',BASE_COMMIT)
    samples={s['question_id']:s for s in git.load(OUT/'sample_manifest.json',BASE_COMMIT)['samples']}
    quick=git.load(OUT/'quick24/sample_plan.json',BASE_COMMIT);probe=git.load(OUT/'failure_probe144/sample_plan.json',BASE_COMMIT)
    hardim={v['sample']['image_sha256'] for v in [*quick['cases'].values(),*probe['cases'].values()]}
    prefim=set(cohorts['old240_image_sha256'])|{samples[q]['image_sha256'] for q in cohorts['pilot120']}
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);tok=proc.tokenizer;designs={};audits=[];candidate=[];specs={}
    for i,(q,s) in enumerate(samples.items()):
        d=parent['cases'][q];td,a=token_design(s,d,tok,proc);designs[q]=td;audits+=a
        # Audit actual complete Question writer identity, not just its label.
        spec=git.load(HIST/'token_manifests'/(s['task']+'__'+q+'.json'),SEM);assert spec['writers']['Semantic-first-Q100']==d['Q']==spec['question_positions'],'Question-only history differs: '+q
        specs[q]=spec
        h=hashlib.sha256(f'structural-receiver-gqa360-v1|{s["task"]}|{q}|{s["image_sha256"]}'.encode()).hexdigest()
        candidate.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],selection_sha256=h,token_feasible=td['feasible'],token_exclusion_reasons=json.dumps(td['reasons']),hard_excluded_image=s['image_sha256'] in hardim,preference_tier=int(s['image_sha256'] in prefim),selected=False,selection_reason=''))
        if (i+1)%500==0:print('CPU TOKEN AND Q HISTORY AUDIT',i+1,flush=True)
    chosen=[];used=set();counts=[]
    for task in TASKS:
        raw=[r for r in candidate if r['task']==task];pool=[r for r in raw if r['token_feasible'] and not r['hard_excluded_image']]
        n=0
        for r in sorted(pool,key=lambda r:(r['preference_tier'],r['selection_sha256'])):
            if r['image_sha256'] in used:r['selection_reason']='image_already_allocated';continue
            if n==60:r['selection_reason']='beyond_fixed_hash_quota';continue
            r['selected']=True;r['selection_reason']='selected';chosen.append(r['question_id']);used.add(r['image_sha256']);n+=1
        counts.append(dict(task=task,raw=len(raw),token_infeasible=sum(not r['token_feasible'] for r in raw),hard_excluded_images=sum(r['hard_excluded_image'] for r in raw),eligible=len(pool),selected=n,preference_fallback=sum(r['selected'] and r['preference_tier']==1 for r in raw)))
    for r in candidate:
        if not r['selection_reason']:r['selection_reason']='token_infeasible' if not r['token_feasible'] else 'historically_tested_image'
    csvout(P/'candidate_audit.csv',candidate);csvout(P/'candidate_counts.csv',counts);csvout(P/'token_audit_all3000.csv',audits)
    if len(chosen)!=360:
        atomic_json(P/'completion.json',dict(status='STOP_INSUFFICIENT_FEASIBLE_SAMPLES',actual=len(chosen),counts=counts,new_generations=0));git.close();print('STOP',counts,flush=True);return
    # Only now read old generation/score records: selection above cannot use correctness.
    shards=git.load(HIST/'shards.json',SEM)['image_assignments'];histories={};selected={};manifest=[]
    hashes=git.load(HIST/'input_hashes.json',SEM)
    for q in chosen:
        s=samples[q];d=parent['cases'][q];td=designs[q];spec=specs[q];sk=s['task']+'__'+q;histories[q]={}
        for c,oldlabel in zip(NAMES[:3],HISTORY_LABELS):
            path=HIST/f'shard{shards[s["image_sha256"]]}/checkpoints'/sk/(spec['aliases'][oldlabel]+'.json');cp=git.load(path,SEM);g=cp['generation']
            assert cp['image_sha256']==s['image_sha256'] and cp['sample_sha256']==digest(s) and cp['input_hashes']==d['input_hashes']==hashes[sk]
            assert cp['writers']==g['writer_positions']==td['writers'][c] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
            assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
            assert sha256(cp['raw']['path'])==cp['raw']['sha256']
            with np.load(cp['raw']['path']) as z:assert int(z['first_logits'].argmax())==g['token_ids'][0]
            histories[q][c]=dict(source=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),commit=SEM,historical_label=oldlabel),generation=g,raw=cp['raw'])
        cr=next(r for r in candidate if r['question_id']==q)
        design={k:d[k] for k in ['S','R','Q','B','input_hashes','groups','prompt_chat_sha256','prompt_ids_sha256']};design['structural_positions']=td['positions']
        selected[q]=dict(sample=s,design=design,writers=td['writers'],selection_sha256=cr['selection_sha256'],preference_tier=cr['preference_tier'],stratum='natural_unconditioned_on_accuracy')
        manifest.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],selection_sha256=cr['selection_sha256'],preference_tier=cr['preference_tier'],All_answer=histories[q][NAMES[0]]['generation']['text'],S_answer=histories[q][NAMES[1]]['generation']['text'],Q_answer=histories[q][NAMES[2]]['generation']['text']))
        assert len({digest(w) for w in td['writers'].values()})==7,'Unexpected duplicate writer set; audit before dedup'
    assign={im:i%2 for i,im in enumerate(sorted(used))}
    for v in selected.values():v['worker']=assign[v['sample']['image_sha256']]
    smoke_ids=['00150746','11697551','11127723'];smoke_refs={}
    for q in smoke_ids:
        s=samples[q];d=parent['cases'][q];spec=specs[q];td=designs[q];assert td['feasible'];sk=s['task']+'__'+q
        path=HIST/f'shard{shards[s["image_sha256"]]}/checkpoints'/sk/(spec['aliases']['Semantic-first-Q100']+'.json');cp=git.load(path,SEM)
        smoke_refs[q]=dict(sample=s,writers=td['writers'],input_hashes=d['input_hashes'],Q_source=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)),generation=cp['generation'],raw=cp['raw'])
    plan=dict(cases=selected,histories=histories,conditions=NAMES,comparisons=PAIRS,selection='No correctness or model output used; fixed TASKS allocation, prefer outside Semantic240/Pilot120 images, then ascending prescribed SHA; global image uniqueness; stop if any task<60.',image_assignments=assign,hard_excluded_images=sorted(hardim),preferred_excluded_images=sorted(prefim),candidate_counts=counts,smoke_refs=smoke_refs)
    atomic_json(P/'sample_plan.json',plan);csvout(P/'sample_manifest.csv',manifest);csvout(P/'token_audit.csv',[a for a in audits if a['question_id'] in selected])
    for i in range(2):atomic_json(P/f'worker{i}/sample_plan.json',dict(cases={q:v for q,v in selected.items() if v['worker']==i},histories={q:h for q,h in histories.items() if selected[q]['worker']==i}))
    git.close();atomic_json(P/'history_byte_audit.json',git.files)
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));(P/'cpu_tests.log').write_text(stream.getvalue());assert test.wasSuccessful();extra_audit(plan)
    state=hardware();idle=idle_gpus(state);gpus=idle[:2] if len(idle)>=2 else [];atomic_json(P/'gpu_selection.json',dict(**state,selected=gpus,utc=stamp(),criterion='no compute processes; utilization=0; used<200MiB; free>22000MiB'))
    hist=verify_history();assert hist['model_revision']==REVISION
    protocol=dict(status='FROZEN' if len(gpus)==2 else 'STOP_NO_TWO_IDLE_GPUS',base_commit=BASE_COMMIT,gpus=gpus,sources=source_hashes(),plan_sha256=sha256(P/'sample_plan.json'),manifest_sha256=sha256(P/'sample_manifest.csv'),parent_plan_sha256=sha256(OUT/'token_plan.json'),shard_plan_sha256={str(i):sha256(P/f'worker{i}/sample_plan.json') for i in range(2)},model_revision=REVISION,model_files=hist['model_files'],transformers_sources=hist['transformers_sources'],scorer_sources=hist['scorer_sources'],conditions=NAMES,comparisons=PAIRS,primary_exploratory_comparison=PAIRS[0],questions=360,new_generations=1440,history_reuse=1080,logical_labels=2520,worker_questions={'0':180,'1':180},cpu_tests=test.testsRun,extra_CPU_audit_passed=True,smoke_ids=smoke_ids,bootstrap_repetitions=20000,bootstrap_seed=20261104,ordinary_CI='paired whole-image cluster, question weighted, percentile95%',multiplicity='all five prespecified comparisons also report Bonferroni99% intervals; no NI claims',scope='FP16/Eager/Transformers4.49.0; Greedy32; logits_to_keep=1; hard -inf R->originalImage only, all32 P-1/Decode imageblocked; native dense QKV, fullKV,MLP,Residual; no o_proj ablation',no_full3000=True,token_feasibility_counts=counts,utc=stamp())
    atomic_json(P/'protocol.json',protocol);print('FROZEN360',counts,'GPUS',gpus,flush=True)
    if len(gpus)!=2:atomic_json(P/'completion.json',dict(status='STOP_NO_TWO_IDLE_GPUS',new_generations=0))

def extra_audit(plan):
    cases=plan['cases'];assert len(cases)==360 and len({v['sample']['image_sha256'] for v in cases.values()})==360
    assert not set(plan['image_assignments'])&set(plan['hard_excluded_images'])
    assert all(sum(v['sample']['task']==t for v in cases.values())==60 for t in TASKS)
    assert all(sum(v['worker']==i for v in cases.values())==180 for i in range(2))
    for q,v in cases.items():
        d=v['design'];n=d['R'][-1]+2;layout=dict(prompt_length=n,I=list(range(5,581)),R=d['R'],A=[n-1],other=list(range(5)))
        for c,base in [(NAMES[3],d['Q']),(NAMES[4],d['Q']),(NAMES[5],d['S']),(NAMES[6],d['S'])]:
            w=v['writers'][c];assert len(set(w)-set(base))==2 and set(base)<=set(w)<=set(d['R']) and intact(w,d['groups'])
            native,changed,blocked=cpu_mask(layout,w);assert torch.equal(changed[0,0][~blocked],native[0,0][~blocked]) and torch.isneginf(changed[0,0][blocked]).all()
            assert int(blocked.sum())==(len(d['R'])-len(w)+1)*576 and not blocked[layout['I']].any() and blocked[n-1,layout['I']].all()
            a,b,m=cpu_mask(layout,w,past=n);assert int(m.sum())==576 and torch.equal(b[0,0][~m],a[0,0][~m])
    assert not torch.cuda.is_initialized()

def verify():
    p=read(P/'protocol.json');assert p['status']=='FROZEN' and len(p['gpus'])==2 and not (P/'STOP.json').exists()
    assert p['sources']==source_hashes() and p['plan_sha256']==sha256(P/'sample_plan.json') and p['manifest_sha256']==sha256(P/'sample_manifest.csv') and p['parent_plan_sha256']==sha256(OUT/'token_plan.json')
    for i,h in p['shard_plan_sha256'].items():assert sha256(P/f'worker{i}/sample_plan.json')==h
    return p

def smoke(model,proc,p,W,rank):
    import semantic_writer_error_cases.run as oldrun
    oldrun.OUT=W;oldrun.RAW=RAW/f'worker{rank}';oldrun.smoke(model,proc,p)
    plan=read(P/'sample_plan.json');checks=[]
    for q in p['smoke_ids']:
        ref=plan['smoke_refs'][q];inputs=prepared(model,proc,ref['sample']);assert tensor_hashes(inputs)==ref['input_hashes'];fp=hook_footprint(model);versions={n:x._version for n,x in model.named_parameters()}
        g,f,t,a,_=generate(model,proc.tokenizer,inputs,ref['writers']['Question-Only'],capture=True)
        assert sha256(ROOT/ref['Q_source']['path'])==ref['Q_source']['sha256'] and sha256(ref['raw']['path'])==ref['raw']['sha256']
        assert g['token_ids']==ref['generation']['token_ids'] and g['text']==ref['generation']['text']
        with np.load(ref['raw']['path']) as z:assert torch.equal(torch.from_numpy(f),torch.from_numpy(z['first_logits']))
        assert g['audit']['passed'] and fp==hook_footprint(model) and versions=={n:x._version for n,x in model.named_parameters()} and tensor_hashes(inputs)==ref['input_hashes']
        r=atomic_npz(RAW/f'worker{rank}'/'smoke_Q'/(q+'.npz'),first_logits=f,valid_logits=t,audit=a);checks.append(dict(question_id=q,historical_Q_token_ids_and_first_logits_strict=True,raw=r))
    q=p['smoke_ids'][0];ref=plan['smoke_refs'][q];inputs=prepared(model,proc,ref['sample']);new=[]
    for c in NAMES[3:]:
        g,f,t,a,_=generate(model,proc.tokenizer,inputs,ref['writers'][c],capture=True);assert g['writer_positions']==ref['writers'][c] and g['audit']['passed'] and g['own_cache'] and g['own_prefix']
        raw=atomic_npz(RAW/f'worker{rank}'/'smoke_structural'/q/(c+'.npz'),first_logits=f,valid_logits=t,audit=a);new.append(dict(condition=c,generation=g,raw=raw))
    atomic_json(W/'structural_smoke_validation.json',dict(passed=True,Q_historical_replay=checks,four_frozen_new_writer_conditions=new))

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
            if label in NAMES[:3]:
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
    verify();assert count==4*len(cases) and reused==3*len(cases) and labels==7*len(cases)
    atomic_json(W/'completion.json',dict(passed=True,worker=rank,pid=os.getpid(),questions=len(cases),new_generations=count,reused_generations=reused,labels=labels,seconds=time.perf_counter()-start,code_commit=code,gpu=gpu,utc=stamp()))
    print('WORKER COMPLETE',rank,count,flush=True)

def driver():
    p=verify();start=time.perf_counter();lock=(P/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);children=[];logs=[]
    atomic_json(P/'driver_started.json',dict(pid=os.getpid(),utc=stamp(),plan_sha256=p['plan_sha256']))
    try:
        for i,gpu in enumerate(p['gpus']):
            w=P/f'worker{i}';w.mkdir(exist_ok=True);log=(w/'run.log').open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONUNBUFFERED':'1'}
            children.append(subprocess.Popen([sys.executable,'-m','gqa360_structural_receiver_selection.run','worker','--rank',str(i)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT))
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
