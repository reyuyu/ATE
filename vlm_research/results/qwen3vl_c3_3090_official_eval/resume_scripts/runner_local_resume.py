"""Single GPU, single model load; auditable checkpoints, no algorithm tuning."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');sys.path.insert(0,str(W))
O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval'
sys.path.insert(0,str(O/'scripts'))
import torch,transformers
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from vlm_research.qwen3vl_early_windows.experiment import prepare,hook_counts,gqa_prediction
from runtime import generate
from benchmark_inputs import digest,prepare as prepare_benchmark

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader'],text=True).strip()
def checkpoint(path,binding,run):
 if path.exists():
  old=read(path);assert old['binding']==binding,('Checkpoint binding mismatch',str(path));return old['result']
 result=run();write(path,dict(binding=binding,result=result,at_UTC=now()));return result
def main():
 lock=(O/'GPU_run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1)
 protocol=read(O/'protocol.json');source=read(O/'source_manifest.json')
 for name,h in source['files'].items():assert sha(W/name)==h,name
 for name,spec in read(O/'model_manifest.json')['files'].items():
  st=(Path(protocol['model_path'])/name).stat();assert st.st_size==spec['bytes'] and st.st_mtime_ns==spec['mtime_ns'],name
 assert os.environ['CUDA_VISIBLE_DEVICES']==protocol['gpu_UUID']
 assert transformers.__version__=='4.57.6'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already busy; do not stop external processes'
 binding=dict(protocol_SHA256=sha(O/'protocol.json'),source_manifest_SHA256=sha(O/'source_manifest.json'),model_manifest_SHA256=sha(O/'model_manifest.json'),GPU_UUID=protocol['gpu_UUID'])
 historical=read(O/'historical_inputs.json');rows=historical['rows'];by={s['question_id']:s for s in rows}
 assert len(rows)==1000 and len({s['image_sha256'] for s in rows})==918
 write(O/'progress.json',dict(status='LOADING_MODEL',pid=os.getpid(),binding=binding,at_UTC=now(),gpu=gpu()))
 processor=AutoProcessor.from_pretrained(protocol['model_path'],local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(protocol['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert len(model.model.language_model.layers)==36 and all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
 assert model.config._attn_implementation=='eager' and model.config.text_config._attn_implementation=='eager' and model.config.vision_config._attn_implementation=='eager'
 hooks=hook_counts(model);started=time.monotonic();peak=int(read(O/'local_disk_migration.json')['prior_observed_peak_allocated'])
 def call(x,l,c,**kw):
  nonlocal peak
  r,v=generate(model,processor,x,l,c,**kw);assert hook_counts(model)==hooks,'Hook leak'
  peak=max(peak,r['memory']['peak_allocated']);return r,v
 def hb(s,l):return dict(**binding,question_id=s['question_id'],input_SHA256=digest(l),sample_SHA256=digest(s))
 # Storage-only restart: verified prior structural/historical completion is retained.
 prior_validation=read(O/'validation.json');prior_historical=read(O/'historical_completion.json')
 assert prior_validation['structural_status']=='PASS' and prior_validation['request_cache_isolation_and_hook_cleanup']=='PASS'
 assert prior_validation['binding']==binding and prior_historical['binding']==binding
 assert prior_historical['status']=='COMPLETE' and prior_historical['questions']==1000 and prior_historical['generations']==3000
 assert prior_validation['thresholds_unchanged'] is True
 print('LOCAL_DISK_RESUME_REUSES_COMPLETED_SMOKE_AND_3000_HISTORICAL_PREDICTIONS',flush=True)
 # Complete available seven-benchmark protocols before efficiency, as authorized.
 from seven_stage_resume import run_seven
 run_seven(model,processor,call,binding)
 # Same 48 images, fresh KV, Latin rotation, real fixed16 and fixed32 decode.
 for j,q in enumerate(historical['benchmark_question_ids']):
  s=by[q];x,l=prepare(processor,s);assert l==historical['input_bindings'][q];b=hb(s,l)
  refs={}
  for regime in ['natural','fixed16','fixed32']:
   cap=16 if regime=='fixed16' else 32;fixed=regime!='natural'
   for c in protocol['conditions']:
    if not fixed:refs[regime,c]=read(O/'historical_predictions'/f'{q}__{c}.json')['result']['token_ids']
    else:refs[regime,c]=call(x,l,c,max_tokens=cap,fixed_length=True)[0]['token_ids']
   rotation=int(digest(q)[:8],16)%3
   for rep in range(-2,5):
    k=(rotation+rep)%3;order=protocol['conditions'][k:]+protocol['conditions'][:k]
    for c in order:
     path=O/'efficiency_raw'/f'{q}__{regime}__{c}__{rep}.json'
     if rep>=0 and path.exists():assert read(path)['binding']==b;continue
     r,_=call(x,l,c,max_tokens=cap,fixed_length=fixed,check_finite=False)
     assert r['token_ids']==refs[regime,c],('Nonrepeatable efficiency tokens',q,c,regime,rep)
     if fixed:assert r['generated_tokens']==cap and r['incremental_decode_steps']==cap-1
     if rep>=0:write(path,dict(binding=b,question_id=q,image_sha256=s['image_sha256'],regime=regime,repeat=rep,condition_order=order,result=r,reference_finite_checked=True,at_UTC=now()))
  write(O/'progress.json',dict(status='RUNNING',phase='EFFICIENCY48',completed_questions=j+1,total_questions=48,binding=binding,peak_allocated=peak,at_UTC=now(),gpu=gpu()))
  print('EFFICIENCY',j+1,'/48',flush=True);del x;gc.collect()
 write(O/'efficiency_completion.json',dict(status='COMPLETE',inputs=48,measured_requests=48*3*3*5,binding=binding))
 del model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
 write(O/'progress.json',dict(status='GPU_STAGES_FINISHED_PENDING_CPU_AUDIT',binding=binding,peak_allocated=peak,elapsed_seconds=time.monotonic()-started,at_UTC=now(),gpu=gpu()))
 print('GPU_STAGES_FINISHED',gpu(),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(O/'failure.json',dict(status='BLOCKED',error=repr(e),traceback=traceback.format_exc(),at_UTC=now(),gpu=gpu()))
  write(O/'progress.json',dict(status='BLOCKED',reason=repr(e),at_UTC=now()))
  raise
