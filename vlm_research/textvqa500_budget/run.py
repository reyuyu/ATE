"""Single model load: idle check, correctness/reuse smoke, resumable fixed grid."""
import gzip,json,os,platform,subprocess,traceback,datetime
from pathlib import Path
from unittest.mock import patch
import torch,transformers
from .common import *
from .engine import generate
from ..p1_layerwise_accuracy import engine as adapter_module
from ..middle10_attribution.engine import prepare
from ..benchmarks.scoring import textvqa_score

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def idle_check():
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
 assert not jobs,('GPU conflict; leave existing processes untouched',jobs)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
 assert 'RTX 3090' in gpu
 return gpu

def smoke(a,samples,historical):
 report=dict(status='RUNNING',cases=[],budget_checks=[],started=now())
 for key in list(samples)[:2]:
  layout,_,binding=prepare(a,samples[key]);expected=historical['Native'][key]
  assert digest(binding)==read(AUDIT/'input_bindings.json')[key]
  native,ln,_=generate(a,layout,None,audit=True)
  noop,lno,observed=generate(a,layout,list(range(32)),audit=True)
  assert native['token_ids']==noop['token_ids'] and all(torch.equal(x,y) for x,y in zip(ln,lno))
  assert native['token_ids']==json.loads(expected['token_ids']),('historical Native mismatch',key,native['token_ids'],expected['token_ids'])
  assert native['first_top1_token']==int(expected['first_top1_token'])
  # A second instrumented no-op confirms complete native cache and projection identity.
  other,lo,observed2=generate(a,layout,list(range(32)),audit=True)
  assert observed.cache_hashes==observed2.cache_hashes and observed.prefill_hashes==observed2.prefill_hashes
  assert all(torch.equal(x,y) for x,y in zip(lno,lo))
  rr={}
  for condition in ['Early21','Middle14','Late21']:
   g,lg,c=generate(a,layout,CONDITIONS[condition],audit=True)
   h=historical[condition][key]
   assert g['token_ids']==json.loads(h['token_ids']),('historical masked mismatch',condition,key,g['token_ids'],h['token_ids'])
   assert torch.equal(lg[0],ln[0]),('first-token logits',condition,key)
   assert c.prefill_hashes==observed.prefill_hashes and c.cache_hashes[0]==observed.cache_hashes[0]
   assert abs(textvqa_score(g['text'],samples[key]['answer'])-float(h['score']))<1e-12
   rr[condition]=dict(token_ids=g['token_ids'],score=float(h['score']),audit=c.summary())
  report['cases'].append(dict(key=key,native_tokens=native['token_ids'],native_history_exact=True,native_noop_all_logits_exact=True,full_native_KV_exact=True,first_native_logit_hash_historical_exact=native['first_logits_sha256']==expected['first_logits_sha256'],reuse_checks=rr))
 # All predefined new windows tested before formal generation. Fixed four steps are
 # diagnostic only; they never enter accuracy estimates or change the cohort.
 key=next(iter(samples));layout,_,_=prepare(a,samples[key])
 native,ln,base=generate(a,layout,list(range(32)),audit=True,max_tokens=4,fixed=True)
 for condition,keep in CONDITIONS.items():
  if condition in REUSE:continue
  g,lg,c=generate(a,layout,keep,audit=True,max_tokens=4,fixed=True)
  assert torch.equal(lg[0],ln[0]) and g['first_top1_token']==native['first_top1_token']
  assert c.prefill_hashes==base.prefill_hashes and c.cache_hashes[0]==base.cache_hashes[0]
  report['budget_checks'].append(dict(condition=condition,keep=keep,first_logits_native_exact=True,prefill_QKV_and_full_KV_native_exact=True,audit=c.summary()))
 report.update(status='PASS',finished=now(),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
 atomic(OUT/'gpu_smoke.json',report)
 return report

def main():
 assert read(OUT/'input_audit.json')['status']=='PASS'
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 protocol=read(OUT/'protocol.json');assert protocol['decode_image_layers']==CONDITIONS
 sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'vlm_research/textvqa500_budget').glob('*.py'))}
 freeze=dict(protocol_sha256=sha(OUT/'protocol.json'),input_audit_sha256=sha(OUT/'input_audit.json'),source_hashes=sources)
 existing=OUT/'execution_freeze.json'
 if existing.exists():assert read(existing)==freeze,'Frozen source/protocol changed; refuse checkpoint reuse'
 else:atomic(existing,freeze)
 run_id=digest(freeze)
 gpu=idle_check()
 atomic(OUT/'run_status.json',dict(status='LOADING',gpu=gpu,at=now(),run_id=run_id))
 samples=load_samples();paths=read(AUDIT/'runtime_mapping.json');bindings=read(AUDIT/'input_bindings.json')
 for key,s in samples.items():s['image_path']=paths[key]
 with gzip.open(OUT/'reused_predictions.json.gz','rt',encoding='utf-8') as f:
  historical={c:{r['key']:r for r in rr} for c,rr in json.load(f).items()}
 with patch.object(adapter_module,'OLD',ROOT/'vlm_research'),patch.object(adapter_module,'model_path',lambda kind:MODEL):
  a=adapter_module.Adapter('llava')
 atomic(OUT/'runtime_environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,model_path=str(MODEL),revision=REV,gpu=gpu,environment=a.environment))
 print('MODEL LOADED; correctness smoke',flush=True)
 smoke(a,samples,historical)
 print('SMOKE PASS; formal missing conditions only',flush=True)
 checkpoint=OUT/'checkpoints';checkpoint.mkdir(exist_ok=True)
 total=0;new=0
 for condition,keep in CONDITIONS.items():
  if condition in REUSE:continue
  completed=0
  for key,sample in samples.items():
   target=checkpoint/condition/(key+'.json')
   if target.exists():
    wrap=read(target);p=wrap['payload'];assert wrap['sha256']==digest(p) and p['run_id']==run_id and p['input_binding_sha256']==bindings[key]
   else:
    layout,_,binding=prepare(a,sample);assert digest(binding)==bindings[key]
    g,_,_=generate(a,layout,keep)
    assert g['first_top1_token']==int(historical['Native'][key]['first_top1_token']),('first answer token differs from historical Native',condition,key)
    assert 0<len(g['token_ids'])<=32 and g['kv']['allocations']=='native DynamicCache'
    score=textvqa_score(g['text'],sample['answer']);assert 0<=score<=1
    p=dict(run_id=run_id,key=key,condition=condition,question_id=sample['question_id'],image_sha256=sample['image_sha256'],input_binding_sha256=digest(binding),generation={k:v for k,v in g.items() if k not in ['kv','audit','visual_audit']},score=score,at=now())
    atomic(target,dict(payload=p,sha256=digest(p)));new+=1
   total+=1;completed+=1
   if completed%25==0 or completed==500:
    atomic(OUT/'run_status.json',dict(status='RUNNING',condition=condition,condition_completed=completed,total_completed=total,new_this_invocation=new,target=9500,at=now(),run_id=run_id))
    print(f'{now()} {condition} {completed}/500 total {total}/9500',flush=True)
  print('CONDITION COMPLETE',condition,flush=True)
 assert total==9500
 atomic(OUT/'run_status.json',dict(status='COMPLETE',total_completed=total,new_this_invocation=new,at=now(),run_id=run_id,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
 print('ALL FIXED CONDITIONS COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  atomic(OUT/'failure.json',dict(status='STOPPED',at=now(),error=repr(error),traceback=traceback.format_exc()))
  print(traceback.format_exc(),flush=True)
  raise
