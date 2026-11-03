"""Freeze real legacy-selector behavior; change only STOP membership of to/can."""
import argparse,collections,datetime,fcntl,io,os,subprocess,sys,time,traceback,unittest,shutil,inspect
import numpy as np
import torch
from gqa3000_space_nl_visual_writers.common import *
from gqa3000_space_nl_visual_writers.failure_probe import csvout,idle_gpus,stamp
from gqa360_structural_receiver_selection.run import source_hashes as inherited_sources
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz,verify as verify_history
from gqa3000_token_visual_writing.engine import generate,cpu_mask,ReceiverMask
from relay_source_causality.common import hardware
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from paper_rq1.engine import cache_fingerprint
from knockout_probe import model_layers
from vlm_common import load_model
import semantic_visual_writers.groups as original
import query_role_smoke_fix.run as repaired
from . import groups_variant as variant

BASE=ROOT
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/gqa360_semantic_stop_to_keep_can'
RAW=Path('/data1/likai/ATE-resources/experiments/gqa360_semantic_stop_to_keep_can')
BASE_COMMIT='74afff563d255f23b0859f19def2de64accff109'
NEW='Semantic-STOP-to-KEEP-can'
NAMES=['All-Writer','Semantic-k','Question-Only',NEW]

def source_hashes():
    return {**{str(BASE/n):h for n,h in inherited_sources().items()},**{str(Path(__file__).with_name(n)):sha256(Path(__file__).with_name(n)) for n in ['run.py','__init__.py','groups_variant.py']}}

class SelectorTests(unittest.TestCase):
    def test_only_two_STOP_entries_changed(self):
        self.assertEqual(variant.STOP,(original.STOP|{'to'})-{'can'})
        self.assertEqual(Path(variant.__file__).read_text(),Path(original.__file__).read_text().replace('having can could','having could').replace('for as then there','to for as then there'))
    def test_all_algorithms_and_regex_identical(self):
        for name in ['flatten','subsets','intact','token_groups','select']:self.assertEqual(inspect.getsource(getattr(original,name)),inspect.getsource(getattr(variant,name)))
        self.assertEqual(original.WORD.pattern,variant.WORD.pattern);self.assertEqual(original.RANDOM_SEEDS,variant.RANDOM_SEEDS)
    def budget_case(self,word):
        words=[word,'red','small','car','?'];q=list(range(1,6));r=list(range(1,10))
        def groups(stop):return [dict(positions=[i+1],words=[w],core=w not in stop and w!='?',question=True,decoded=w) for i,w in enumerate(words)]+[dict(positions=[6,7,8,9],words=['ASSISTANT'],core=False,question=False,decoded='ASSISTANT')]
        return original.select(groups(original.STOP),q,r,'synthetic '+word),variant.select(groups(variant.STOP),q,r,'synthetic '+word)
    def test_to_removal_can_reduce_k_four_to_zero(self):
        a,b=self.budget_case('to');self.assertEqual((a['k'],b['k']),(4,0));self.assertEqual(b['writers']['Semantic-k'],[])
    def test_can_addition_can_increase_k_zero_to_four(self):
        a,b=self.budget_case('can');self.assertEqual((a['k'],b['k']),(0,4));self.assertEqual(b['writers']['Semantic-k'],[1,2,3,4])

def harray(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def freeze():
    from transformers import AutoProcessor
    from gqa3000_space_nl_visual_writers.tests import Audit
    P.mkdir(parents=True,exist_ok=True);assert not (P/'protocol.json').exists();torch.set_num_threads(4)
    disk=dict(output=str(P),raw=str(RAW),output_free_bytes=shutil.disk_usage(P).free,git_volume_free_bytes=shutil.disk_usage(BASE).free,min_output_free_bytes=512*1024**2,min_git_free_bytes=256*1024**2,estimated_new_data_upper_bytes=360*100000+30*1024**2,full_step_logits_archived=False)
    disk['passed']=disk['output_free_bytes']>disk['min_output_free_bytes'] and disk['git_volume_free_bytes']>disk['min_git_free_bytes'];atomic_json(P/'disk_preflight.json',disk)
    if not disk['passed']:atomic_json(P/'completion.json',dict(status='STOP_DISK_SPACE_INSUFFICIENT'));print('STOP DISK',disk,flush=True);return
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Audit),unittest.defaultTestLoader.loadTestsFromTestCase(SelectorTests)]);stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(P/'cpu_tests.log').write_text(stream.getvalue());assert tests.wasSuccessful()
    git=GitHistory();old=git.load(BASE/'results/gqa360_structural_receiver_selection/sample_plan.json',BASE_COMMIT);parent=git.load(OUT/'token_plan.json',BASE_COMMIT);pos=git.load(PILOT/'pos_parse_all3000.json',BASE_COMMIT)
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);cases={};histories={};changes=[];summary=collections.Counter();group_changes=[]
    for q,v in old['cases'].items():
        s=v['sample'];d=parent['cases'][q];spec=git.load(HIST/'token_manifests'/(s['task']+'__'+q+'.json'),SEM)
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True);assert digest(chat)==s['input_mapping']['chat_prompt_sha256']
        args=(proc.tokenizer,s['question'],chat,s['input_mapping']['prompt_ids'],s['input_mapping']['visual_positions'],d['R'])
        og,oq,ow=original.token_groups(*args);ng,nq,nw=variant.token_groups(*args);a=original.select(og,oq,d['R'],s['question']);b=variant.select(ng,nq,d['R'],s['question'])
        assert a==spec['natural'] and a['writers']['Semantic-k']==d['S']==v['writers']['Semantic-k'];assert oq==nq==d['Q']
        assert [{k:x for k,x in g.items() if k!='core'} for g in og]==[{k:x for k,x in g.items() if k!='core'} for g in ng]
        assert [{k:x for k,x in w.items() if k!='core'} for w in ow]==[{k:x for k,x in w.items() if k!='core'} for w in nw]
        oldS=a['writers']['Semantic-k'];newS=b['writers']['Semantic-k'];added=sorted(set(newS)-set(oldS));removed=sorted(set(oldS)-set(newS));changed=oldS!=newS
        target={t:[w for w in ow if w['in_question'] and w['text'].lower()==t] for t in ['to','can']};targetpos={t:{p for w in ws for p in w['positions']} for t,ws in target.items()};alltarget=targetpos['to']|targetpos['can']
        has=bool(alltarget)
        if not has:assert oldS==newS and a['k']==b['k']
        assert newS==sorted(set(newS)) and set(newS)<=set(d['Q']) and intact(newS,og)
        cpstart=d['question_char_start'];canpos=[]
        for w in target['can']:
            relstart=w['start']-cpstart;relend=w['end']-cpstart;tokens=[t for t in pos[q]['tokens'] if t['start']<relend and t['end']>relstart]
            canpos.append(dict(word=w['text'],positions=w['positions'],question_start=relstart,question_end=relend,POS=[t['pos'] for t in tokens],tags=[t['tag'] for t in tokens],dep=[t['dep'] for t in tokens],head=[t['head_text'] for t in tokens]))
        histories[q]={}
        for c in NAMES[:3]:
            h=old['histories'][q][c];path=BASE/h['source']['path'];cp=git.load(path,h['source']['commit']);g=cp['generation'];assert cp['input_hashes']==d['input_hashes'] and cp['image_sha256']==s['image_sha256'] and cp['writers']==g['writer_positions']==v['writers'][c] and g['correct']==int(gqa_prediction(g['text'])==s['answer']) and g['own_cache'] and g['own_prefix'] and g['audit']['passed']
            assert sha256(cp['raw']['path'])==cp['raw']['sha256'];histories[q][c]=dict(source=dict(path=str(path),sha256=sha256(path),commit=h['source']['commit']),generation=g,raw=cp['raw'])
        if not changed:histories[q][NEW]=histories[q]['Semantic-k']
        visual=git.load(BASE/f'results/gqa360_structural_receiver_selection/worker{v["worker"]}/records'/q/'S+SPACE+NL.json',BASE_COMMIT)['projected_visual_sha256']
        writers={c:v['writers'][c] for c in NAMES[:3]};writers[NEW]=newS
        other_added=sorted(set(added)-alltarget);other_removed=sorted(set(removed)-alltarget)
        cases[q]=dict(sample=s,design={k:d[k] for k in ['input_hashes','R','Q','S','groups']},writers=writers,stratum='unchanged_natural360',reference_visual_sha256=visual,changed=changed,has_to=bool(target['to']),has_can=bool(target['can']),can_POS=canpos,old_k=a['k'],new_k=b['k'],budget_changed=a['k']!=b['k'],other_positions_changed=bool(other_added or other_removed))
        changes.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],has_to=bool(target['to']),has_can=bool(target['can']),writer_changed=changed,old_k=a['k'],new_k=b['k'],delta_k=b['k']-a['k'],R=len(d['R']),old_R_ratio=a['k']/len(d['R']),new_R_ratio=b['k']/len(d['R']),old_core_candidate_tokens=a['candidate_core_tokens'],new_core_candidate_tokens=b['candidate_core_tokens'],old_budget_reduction=a['budget_reduction_for_whole_word_suffix'],new_budget_reduction=b['budget_reduction_for_whole_word_suffix'],old_writers=json.dumps(oldS),new_writers=json.dumps(newS),to_positions=json.dumps(sorted(targetpos['to'])),can_positions=json.dumps(sorted(targetpos['can'])),to_removed=json.dumps(sorted(set(removed)&targetpos['to'])),can_added=json.dumps(sorted(set(added)&targetpos['can'])),added=json.dumps(added),removed=json.dumps(removed),other_added=json.dumps(other_added),other_removed=json.dumps(other_removed),can_POS=json.dumps(canpos)))
        for oldg,newg in zip(og,ng):
            if set(oldg['positions'])&set(added+removed):group_changes.append(dict(question_id=q,task=s['task'],question=s['question'],words=json.dumps(oldg['words']),decoded=oldg['decoded'],positions=json.dumps(oldg['positions']),old_core=oldg['core'],new_core=newg['core'],old_selected=set(oldg['positions'])<=set(oldS),new_selected=set(oldg['positions'])<=set(newS),target_group=bool(set(oldg['positions'])&alltarget)))
        summary['changed']+=changed;summary['unchanged']+=not changed;summary['no_to_or_can_unchanged']+=not has;summary['has_to']+=bool(target['to']);summary['has_can']+=bool(target['can']);summary['budget_changed']+=a['k']!=b['k'];summary['other_positions_changed']+=bool(other_added or other_removed)
        n=d['R'][-1]+2;layout=dict(prompt_length=n,I=list(range(5,581)),R=d['R'],A=[n-1],other=list(range(5)));native,m,blocked=cpu_mask(layout,newS);assert torch.equal(native[0,0][~blocked],m[0,0][~blocked]) and torch.isneginf(m[0,0][blocked]).all() and int(blocked.sum())==(len(d['R'])-len(newS)+1)*576 and not blocked[layout['I']].any() and blocked[n-1,layout['I']].all()
        aa,bb,mm=cpu_mask(layout,newS,past=n);assert int(mm.sum())==576 and torch.equal(aa[0,0][~mm],bb[0,0][~mm])
    # Deterministic whole-image shards, balancing changed generations first.
    newcounts=[0,0];totals=[0,0]
    for q,v in sorted(cases.items(),key=lambda kv:(not kv[1]['changed'],kv[1]['sample']['image_sha256'])):
        rank=min(range(2),key=lambda i:(newcounts[i],totals[i],i));v['worker']=rank;newcounts[rank]+=v['changed'];totals[rank]+=1
    plan=dict(cases=cases,histories=histories,selection='Exact same360; no sampling or outcome filtering; only two STOP entries differ; legacy selector and feasible-budget behavior preserved.',summary=dict(summary),old_STOP=sorted(original.STOP),new_STOP=sorted(variant.STOP),parent_plan_sha256=sha256(BASE/'results/gqa360_structural_receiver_selection/sample_plan.json'))
    atomic_json(P/'sample_plan.json',plan);csvout(P/'token_changes.csv',changes);csvout(P/'word_group_changes.csv',group_changes,['question_id','task','question','words','decoded','positions','old_core','new_core','old_selected','new_selected','target_group']);shutil.copyfile(BASE/'results/gqa360_structural_receiver_selection/sample_manifest.csv',P/'sample_manifest.csv')
    for i in range(2):atomic_json(P/f'worker{i}/sample_plan.json',dict(cases={q:v for q,v in cases.items() if v['worker']==i},histories={q:h for q,h in histories.items() if cases[q]['worker']==i}))
    git.close();atomic_json(P/'history_byte_audit.json',git.files);assert not torch.cuda.is_initialized()
    state=hardware();idle=idle_gpus(state);gpus=idle[:2] if len(idle)>=2 else idle[:1];atomic_json(P/'gpu_selection.json',dict(**state,selected=gpus,utc=stamp()))
    hist=verify_history();protocol=dict(status='FROZEN' if len(gpus)==2 else 'STOP_GPU_RESOURCES',base_commit=BASE_COMMIT,gpus=gpus,sources=source_hashes(),plan_sha256=sha256(P/'sample_plan.json'),manifest_sha256=sha256(P/'sample_manifest.csv'),shard_plan_sha256={str(i):sha256(P/f'worker{i}/sample_plan.json') for i in range(2)},model_revision=REVISION,model_files=hist['model_files'],transformers_sources=hist['transformers_sources'],scorer_sources=hist['scorer_sources'],conditions=NAMES,new_condition_only=NEW,new_generations=summary['changed'],history_reuse=1440-summary['changed'],worker_new={str(i):newcounts[i] for i in range(2)},worker_questions={str(i):totals[i] for i in range(2)},smoke_ids=['00150746','11697551','11127723'],CPU_tests=tests.testsRun,all360_legacy_S_and_natural_selection_exact=True,all360_grouping_and_budget_mask_audits_passed=True,bootstrap_repetitions=20000,bootstrap_seed=20261104,scope='Original LLaVA FP16 Eager4.49.0 greedy32/logits_to_keep1; all32 P-1/Decode imageblocked; only Prefill R writer sets change; independent true KV and native all other computation.',summary=dict(summary),new_archive='Answers/tokenIDs, first-logit SHA256 and compact mask/metadata NPZ only; no full per-step or first-token logit arrays archived.',shared_source_worktree=str(BASE.parent),utc=stamp())
    atomic_json(P/'protocol.json',protocol);print('FROZEN STOP PILOT',dict(summary),'worker_new',newcounts,'GPUS',gpus,flush=True)
    if len(gpus)!=2:atomic_json(P/'completion.json',dict(status='STOP_GPU_RESOURCES',new_generations=0))

def verify():
    p=read(P/'protocol.json');assert p['status']=='FROZEN' and p['sources']==source_hashes() and not (P/'STOP.json').exists()
    assert sha256(P/'sample_plan.json')==p['plan_sha256'] and sha256(P/'sample_manifest.csv')==p['manifest_sha256']
    for i,h in p['shard_plan_sha256'].items():assert sha256(P/f'worker{i}/sample_plan.json')==h
    return p

@torch.inference_mode()
def smoke(model,proc,p,W,rank):
    samples=read(HIST/'sample_manifest.json')['samples'];hashes=read(HIST/'input_hashes.json');shards=read(HIST/'shards.json')['image_assignments'];checks=[]
    for q in p['smoke_ids']:
        s=next(s for s in samples if s['question_id']==q);sk=s['task']+'__'+q;spec=read(HIST/'token_manifests'/(sk+'.json'));inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==hashes[sk];fp=hook_footprint(model);versions={n:x._version for n,x in model.named_parameters()}
        native_ids,rec=repaired.native(model,inputs);reference=[v[-1].float() for v in rec.logits]
        g,f,t,a,parent=generate(model,proc.tokenizer,inputs,spec['writers']['Semantic-k'],noop=True,capture=True,retain_cache=True)
        assert native_ids==g['token_ids'] and len(reference)==len(t) and all(torch.equal(torch.from_numpy(x).float(),y) for x,y in zip(t,reference));ph=cache_fingerprint(parent);replays=[]
        for c in ['All-Writer','Semantic-k']:
            old=read(HIST/f'shard{shards[s["image_sha256"]]}/checkpoints'/sk/(spec['aliases'][c]+'.json'));gg,ff,tt,aa,child=generate(model,proc.tokenizer,inputs,spec['writers'][c],capture=True,retain_cache=True)
            assert old['generation']['token_ids']==gg['token_ids'] and old['generation']['text']==gg['text'] and sha256(old['raw']['path'])==old['raw']['sha256']
            with np.load(old['raw']['path']) as z:assert torch.equal(torch.from_numpy(ff),torch.from_numpy(z['first_logits']))
            assert cache_fingerprint(parent)==ph and all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),child.to_legacy_cache()) for x,y in zip(a,b));replays.append(dict(condition=c,token_ids=gg['token_ids'],first_logits_sha256=harray(ff),all_valid_logits_sha256=harray(tt),historical_token_ids_and_first_logits_strict=True));del child
        del parent
        try:
            with ReceiverMask(model,inputs,spec['writers']['Semantic-k']):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert fp==hook_footprint(model) and versions=={n:x._version for n,x in model.named_parameters()} and tensor_hashes(inputs)==hashes[sk]
        checks.append(dict(question_id=q,Native_noop_all_effective_logits_torch_equal=True,independent_cache=True,exception_and_normal_hook_cleanup=True,weights_inputs_exact=True,replays=replays))
    own=list(read(W/'sample_plan.json')['cases'].values());case=next((v for v in own if v['changed']),own[0]);inputs=prepared(model,proc,case['sample']);counts=[0]*32;handles=[]
    try:
        for k,layer in enumerate(model_layers(model)):
            def mlp(m,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
            handles.append(layer.mlp.register_forward_hook(mlp))
        g,f,t,a,_=generate(model,proc.tokenizer,inputs,case['writers'][NEW],capture=True)
    finally:
        for h in handles:h.remove()
    assert counts==[g['length']]*32 and g['writer_positions']==case['writers'][NEW] and tensor_hashes(inputs)==case['design']['input_hashes']
    raw=atomic_npz(RAW/f'worker{rank}'/'smoke_audit.npz',audit=a,token_ids=np.array(g['token_ids']));atomic_json(W/'smoke_validation.json',dict(passed=True,cases=checks,all_valid_logits_strict=True,original_S_reproduced=True,mask_audit_passed=True));atomic_json(W/'structural_smoke_validation.json',dict(passed=True,new_generation=g,MLP_counts=counts,first_logits_sha256=harray(f),valid_logits_sha256=harray(t),raw=raw,no_logit_arrays_archived=True))

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
                    g,f,t,a,_=generate(model,proc.tokenizer,inputs,writers,capture=False)
                finally:
                    for handle in handles:handle.remove()
                assert calls==[g['length']]*32 and len(features)==1 and fp==hook_footprint(model) and versions=={n:x._version for n,x in model.named_parameters()} and tensor_hashes(inputs)==d['input_hashes']
                assert g['writer_positions']==writers and g['own_cache'] and g['own_prefix'] and g['audit']['passed']
                g['correct']=int(gqa_prediction(g['text'])==s['answer']);metadata=dict(question_id=q,worker=rank,plan_sha256=p['plan_sha256'],code_commit=code,writers=writers,generation=g,input_hashes=d['input_hashes'],MLP_calls=calls,projected_visual_sha256=features[0],weights_unchanged=True,hook_cleanup=True,first_logits_sha256=harray(f),reused=False)
                raw=atomic_npz(RAW/f'worker{rank}'/'main'/q/(uid+'.npz'),audit=a,token_ids=np.array(g['token_ids']),record_metadata=np.array(json.dumps(metadata,sort_keys=True)))
                r=dict(**metadata,raw=raw);atomic_json(path,r);count+=1;visual.add(features[0])
            atomic_json(W/'records'/q/(label+'.json'),dict(condition=label,stratum=v['stratum'],**r));labels+=1
        assert len(visual)<=1
        if visual:assert visual=={v['reference_visual_sha256']}
        print('CASE DONE',q,v['stratum'],'new',count,'labels',labels,'seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert count==p['worker_new'][str(rank)] and count+reused==4*len(cases) and labels==4*len(cases)
    atomic_json(W/'completion.json',dict(passed=True,worker=rank,pid=os.getpid(),questions=len(cases),new_generations=count,reused_generations=reused,labels=labels,seconds=time.perf_counter()-start,code_commit=code,gpu=gpu,utc=stamp()))
    print('WORKER COMPLETE',rank,count,flush=True)

def driver():
    p=verify();start=time.perf_counter();lock=(P/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);children=[];logs=[]
    atomic_json(P/'driver_started.json',dict(pid=os.getpid(),utc=stamp(),plan_sha256=p['plan_sha256']))
    try:
        for i,gpu in enumerate(p['gpus']):
            w=P/f'worker{i}';w.mkdir(exist_ok=True);log=(w/'run.log').open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONUNBUFFERED':'1'}
            children.append(subprocess.Popen([sys.executable,'-m','gqa360_semantic_stop_to_keep_can.run','worker','--rank',str(i)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT))
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
