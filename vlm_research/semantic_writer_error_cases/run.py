"""One idle GPU, dense historical hard mask, bounded exploratory repairs only."""
import argparse,fcntl,time,traceback
import numpy as np
import transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.scorers import gqa_prediction
from vlm_common import load_model,gpu_memory
from gqa3000_token_visual_writing.engine import generate,ReceiverMask
from gqa3000_token_visual_writing.common import verify as verify_history
import query_role_smoke_fix.run as repaired
from .common import *

def freeze():
    p=verify_history();assert read(OUT/'selection_validation.json')['passed'];assert not torch.cuda.is_initialized()
    state=hardware();busy={x.split(',')[0].strip() for x in state['compute_processes'].splitlines()};candidates=[]
    for line in state['table'].splitlines():
        index,uuid,name,used,free,util=[s.strip() for s in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:candidates.append(dict(index=int(index),uuid=uuid,name=name))
    gpu=candidates[0] if candidates else None;state['selected']=gpu;atomic_json(OUT/'gpu_selection.json',state)
    protocol=dict(base_commit=BASE,sources=source_hashes(),candidate_plan_sha256=bind(),case_manifest_sha256=sha256(OUT/'case_manifest.json'),selection_rules_sha256=sha256(OUT/'selection_rules.json'),
        model_revision=REVISION,model_files=p['model_files'],transformers_sources=p['transformers_sources'],scorer_sources=p['scorer_sources'],
        history_protocol_sha256=sha256(HIST/'protocol.json'),gpu=gpu,precision='float16',attention='eager',transformers='4.49.0',max_new_tokens=32,logits_to_keep=1,
        experiment='Add or replace only frozen Prefill R image-key receiver positions; all layers unchanged, P-1/Decode always image-blocked.',
        baseline='Strictly reuse original official All/Semantic/position/random records; smoke replays are validation only.',
        stages='At most240 new greedy evaluations, plus three fixed validation-only smoke cases. Conditional replacements fixed before outcomes.',
        smoke_ids=['00150746','11697551','11127723'],no_gpu_if_busy=True,exploratory_only=True)
    atomic_json(OUT/'protocol.json',protocol);print('FROZEN',gpu,'plan',bind(),flush=True)
def verify():
    p=read(OUT/'protocol.json');assert p['sources']==source_hashes() and p['candidate_plan_sha256']==bind() and p['case_manifest_sha256']==sha256(OUT/'case_manifest.json')
    verify_history();active();return p
def preflight(p):
    state=hardware();gpu=p['gpu'];assert gpu is not None
    fields=[s.strip() for s in next(x for x in state['table'].splitlines() if x.split(',')[1].strip()==gpu['uuid']).split(',')]
    apps=[x for x in state['compute_processes'].splitlines() if x.split(',')[0].strip()==gpu['uuid']]
    state.update(passed=not apps and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0,selected=gpu)
    atomic_json(OUT/'gpu_preflight.json',state);assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid']

@torch.inference_mode()
def smoke(model,proc,p):
    samples=read(HIST/'sample_manifest.json')['samples'];hashes=read(HIST/'input_hashes.json');shards=read(HIST/'shards.json')['image_assignments'];out=[]
    for qid in p['smoke_ids']:
        s=next(s for s in samples if s['question_id']==qid);sk=s['task']+'__'+qid;spec=read(HIST/'token_manifests'/(sk+'.json'));inputs=prepared(model,proc,s)
        assert tensor_hashes(inputs)==hashes[sk];fp=hook_footprint(model);versions={n:v._version for n,v in model.named_parameters()}
        native_ids,rec=repaired.native(model,inputs);reference=[v[-1].float() for v in rec.logits]
        g,f,trace,a,parent=generate(model,proc.tokenizer,inputs,spec['writers']['Semantic-k'],noop=True,capture=True,retain_cache=True)
        assert native_ids==g['token_ids'] and len(reference)==len(trace)
        assert all(torch.equal(torch.from_numpy(x).float(),y) for x,y in zip(trace,reference))
        parenthash=cache_fingerprint(parent);checks={}
        for label in ['All-Writer','Semantic-k']:
            uid=spec['aliases'][label];rank=shards[s['image_sha256']];old=read(HIST/f'shard{rank}/checkpoints'/sk/(uid+'.json'))
            gg,ff,tt,aa,child=generate(model,proc.tokenizer,inputs,spec['writers'][label],capture=True,retain_cache=True)
            assert gg['token_ids']==old['generation']['token_ids'] and gg['text']==old['generation']['text']
            assert sha256(old['raw']['path'])==old['raw']['sha256']
            with np.load(old['raw']['path']) as z:assert torch.equal(torch.from_numpy(ff),torch.from_numpy(z['first_logits']))
            assert cache_fingerprint(parent)==parenthash
            assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),child.to_legacy_cache()) for x,y in zip(a,b))
            checks[label]=dict(generation=gg,original_ids_and_first_logits_exact=True,cache_independent=True)
            del child
        writer=sorted(set(spec['writers']['Semantic-k'])|set(next(x['positions'] for x in spec['groups'] if x['question'] and not set(x['positions'])&set(spec['writers']['Semantic-k']))))
        counts=[0]*32;handles=[]
        try:
            for k,layer in enumerate(model_layers(model)):
                def mlp(m,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
                handles.append(layer.mlp.register_forward_hook(mlp))
            gg,ff,tt,aa,_=generate(model,proc.tokenizer,inputs,writer,capture=True)
        finally:
            for h in handles:h.remove()
        assert counts==[gg['length']]*32 and cache_fingerprint(parent)==parenthash
        del parent
        try:
            with ReceiverMask(model,inputs,writer):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert fp==hook_footprint(model) and versions=={n:v._version for n,v in model.named_parameters()} and tensor_hashes(inputs)==hashes[sk]
        raw=atomic_npz(RAW/'smoke'/(qid+'.npz'),modified_valid_logits=tt,modified_audit=aa)
        row=dict(passed=True,question_id=qid,all_noop_logits_torch_equal=True,checks=checks,modified=gg,mlp_counts=counts,
            independent_cache=True,hook_cleanup=True,weights_and_inputs_exact=True,raw=raw)
        atomic_json(OUT/'smoke_cases'/(qid+'.json'),row);out.append(dict(question_id=qid,passed=True));print('SMOKE PASS',qid,flush=True)
    atomic_json(OUT/'smoke_validation.json',dict(passed=True,cases=out,strict_torch_equal=True,forbidden_actual_probability_zero=True,non_target_masks_exact=True,all32_MLP=True,independent_cache=True))

def historical_baseline(case):
    r=case['rows']['Semantic-k'];path=HIST/r['source_checkpoint'];old=read(path)
    assert old['generation']['token_ids']==json.loads(r['token_ids']) and old['generation']['text']==r['text'] and old['generation']['correct']==0
    assert sha256(path)==gitsha(path) and sha256(old['raw']['path'])==r['raw_logits_sha256']
    return old
def run():
    started=time.perf_counter();torch.set_num_threads(4);p=verify();preflight(p);assert transformers.__version__=='4.49.0'
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(OUT/'worker_started.json',dict(pid=os.getpid(),plan_sha256=bind(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()))
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable);atomic_json(OUT/'environment.json',env)
    smoke(model,proc,p);cases=[c for c in selected() if c['kind']=='harm'];plans=read(OUT/'candidate_plan.json')['cases'];hashes=read(HIST/'input_hashes.json');records=[];conditional=[];new=0
    for case in cases:
        s=case['sample'];qid=s['question_id'];sk=s['task']+'__'+qid;spec=case['spec'];base=historical_baseline(case)
        inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==hashes[sk];cache={}
        def evaluate(label,writers,added,removed=None):
            nonlocal new
            active();assert intact(writers,spec['groups']) and set(writers)<=set(spec['relay_positions'])
            uid=digest(writers)[:20];path=OUT/'checkpoints'/qid/(uid+'.json')
            if uid in cache:row=cache[uid]
            elif path.exists():
                row=read(path);assert row['plan_sha256']==bind() and row['writers']==writers and sha256(row['raw']['path'])==row['raw']['sha256'];cache[uid]=row
            else:
                fp=hook_footprint(model);versions={n:v._version for n,v in model.named_parameters()};before=tensor_hashes(inputs)
                g,first,trace,a,_=generate(model,proc.tokenizer,inputs,writers,capture=True)
                assert fp==hook_footprint(model) and before==tensor_hashes(inputs)==hashes[sk] and versions=={n:v._version for n,v in model.named_parameters()}
                g['correct']=int(g['prediction']==s['answer']);assert g['prediction']==gqa_prediction(g['text'])
                metadata=dict(question_id=qid,plan_sha256=bind(),sample_sha256=digest(s),writers=writers,generation=g,input_hashes=before)
                raw=atomic_npz(RAW/'main'/qid/(uid+'.npz'),first_logits=first,valid_logits=trace,audit=a,record_metadata=np.array(json.dumps(metadata,sort_keys=True)))
                row=dict(**metadata,raw=raw);atomic_json(path,row);cache[uid]=row;new+=1
            g=row['generation'];records.append(dict(question_id=qid,task=s['task'],condition=label,answer=g['text'],prediction=g['prediction'],gold=s['answer'],correct=g['correct'],
                baseline_answer=base['generation']['text'],baseline_correct=0,token_ids=json.dumps(g['token_ids']),added=json.dumps(added),removed=json.dumps(removed or []),
                writers=json.dumps(writers),k=len(writers),R=len(spec['relay_positions']),k_R=len(writers)/len(spec['relay_positions']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                first_top1_flip_vs_Semantic=int(g['token_ids'][0]!=base['generation']['token_ids'][0]),checkpoint=str(path.relative_to(OUT)),raw_logits_path=row['raw']['path'],raw_logits_sha256=row['raw']['sha256']))
            return g['correct']
        singles=plans[qid]['singles'];success={}
        for c in singles:
            success[c['id']]=evaluate(c['id'],c['writers'],c['added'])
            if c['control_writers'] is not None:evaluate(c['id']+'_matched_control',c['control_writers'],c['control_added'])
        for c in plans[qid]['grouped']:evaluate(c['id'],c['writers'],c['added'])
        for c in singles:
            allowed=bool(success[c['id']] and c['replacement_writers'] is not None)
            conditional.append(dict(question_id=qid,candidate=c['id'],single_correct=success[c['id']],replacement_feasible=c['replacement_writers'] is not None,executed=allowed,
                reason='Successful single; execute prelocked replacement' if allowed else 'Single failed or exact whole-group replacement unavailable'))
            if allowed:evaluate(c['id']+'_equal_budget_replace',c['replacement_writers'],c['added'],c['remove'])
        print('CASE COMPLETE',qid,'new unique',new,'seconds',round(time.perf_counter()-started,1),flush=True)
    assert new<=240
    csvwrite(OUT/'supplementation_results.csv',records);csvwrite(OUT/'conditional_replacement_audit.csv',conditional)
    verify();atomic_json(OUT/'gpu_completion.json',dict(passed=True,pid=os.getpid(),plan_sha256=bind(),case_count=12,condition_labels=len(records),unique_new_generations=new,
        seconds=time.perf_counter()-started,model_instances=1,gpu=p['gpu'],baseline_records_reused=12,original_results_overwritten=False,memory=gpu_memory()))
    print('BOUNDED GPU COMPLETE',new,'unique,',len(records),'labels',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        if args.stage=='run':atomic_json(OUT/'STOP.json',dict(status='FAILED',pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware()))
        raise
