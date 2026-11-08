"""Full authorized MME Paper accuracy; unchanged prefix+SVH, one GPU load."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
import fcntl,gc,gzip,hashlib,json,sqlite3,subprocess,sys,traceback
import torch,transformers
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';F=B/'v2drop_prefix_svh_accuracy_3090';O=B/'v2drop_prefix_svh_mme_full_3090';M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(F),str(O),str(R)]
from fusion_engine import PrefixSVH,greedy
from run_fusion import trace_execution,structural
from vlm_research.information_horizon_alignment import engine as frozen,score as scorer
from vlm_research.mme_seed_quality import engine as loader
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def old_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
def save(n,v):
 p=O/n;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def score(s,text):
 value=scorer.score(s,text);low=text.lower()
 if low in ('yes','no'):label=low
 else:
  if 'yes' in low[:4]:label='yes'
  elif 'no' in low[:4]:label='no'
  else:label='other'
 assert value['parsed']==label and value['score']==float(label==s['answer'])
 return value
def main():
 guard=(O/'run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'failure.json').exists(),'Do not retry a critical failure without review'
 audit=read(O/'input_audit.json');assert audit['status']=='PASS'
 for name,h in audit['frozen_sources'].items():assert sha(R/name)==h,name
 for name,h in audit['fusion_sources'].items():assert sha(F/name)==h,name
 assert not cmd('git','status','--porcelain','--untracked-files=no')
 samples=gz(O/'input_manifest.jsonl.gz');lookup={s['key']:s for s in samples};assert len(samples)==len(lookup)==2374
 reference={r['key']:r for r in gz(O/'SVH_reference_predictions.jsonl.gz')}
 protocol=dict(condition='V2Drop-prefix+SVH',benchmark='MME-paper',questions=2374,pairs=1187,model=audit['model'],revision=audit['revision'],
  code_commit=audit['code_commit'],dtype='float16',attention='eager',torch='2.6.0+cu124',transformers='4.49.0',batch_size=1,greedy=True,num_beams=1,EOS=2,max_new_tokens=128,
  input_audit_sha256=sha(O/'input_audit.json'),manifest_sha256=sha(O/'input_manifest.jsonl.gz'),sources=dict(run_mme=sha(O/'run_mme.py'),**audit['fusion_sources']),
  inherited_strict_numerical_smoke=dict(path=str(F/'physical_smoke.json'),sha256=audit['source_physical_smoke_sha256'],status='PASS',unchanged_thresholds=True,same_GPU_and_implementation=True),
  no_extra_algorithm_or_Native_generation=True,known_MME_image_difference=audit['known_MME_image_difference'])
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
 else:save('protocol.json',protocol)
 binding=digest(protocol);db=sqlite3.connect(O/'predictions.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS predictions(key TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)');db.commit();existing={}
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(payload);assert digest(r)==h and r['binding']==binding and r['sample_sha256']==digest(lookup[key]);assert score(lookup[key],r['generation']['text'])==r['scoring'];existing[key]=r
 state=dict(status='RUNNING',phase='GPU_LOAD',completed=len(existing),expected=2374,model_loads_this_process=1,pid=os.getpid(),at=now(),binding=binding,MME_integration_smoke='NOT_VALIDATED')
 a=None;ctl=None;cases=read(O/'MME_integration_smoke.json')['cases'] if (O/'MME_integration_smoke.json').exists() else []
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU occupied'
  assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
  assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090' and torch.cuda.mem_get_info()[0]>20*2**30
  state['GPU']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader');save('progress.json',state)
  loader.MODELS=dict(loader.MODELS,llava=M);print('MME_MODEL_LOAD_ONCE',flush=True);a=frozen.load('B')
  assert {str(p.device) for p in a.model.parameters()}=={'cuda:0'} and {p.dtype for p in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager' and a.model.generation_config.eos_token_id==2
  state['phase']='MME_ACCURACY';save('progress.json',state)
  for index,s in enumerate(samples):
   if s['key'] in existing:continue
   assert not (O/'STOP_REQUESTED').exists(),'Explicit stop request'
   layout,input_binding=frozen.prepare(a,s);r=reference[s['key']]
   assert old_digest(input_binding['prompt'])==r['prompt_sha256'] and old_digest(input_binding['input_ids'])==r['input_ids_sha256']
   assert input_binding['pixel_values_sha256']==r['pixel_values_sha256'] and input_binding['expanded_length']==r['expanded_length']
   assert layout.cutoff==20 and len(layout.image)==576 and layout.text[-1]==layout.length-1
   ctl=PrefixSVH(a,layout,128,True,audit=index<2)
   if index<2:
    with trace_execution(a,layout) as trace:g,_=greedy(a,layout,ctl,128)
    kv=structural(ctl,layout,trace,len(g['token_ids']))
    cases.append(dict(key=s['key'],status='PASS',generation=g,selection_events=ctl.selection_events,layer_events=ctl.layer_events,
     visual_access=ctl.visual_events,attention_audit=ctl.events,KV_positions=kv,execution_trace=trace,
     input_tensor_binding_matches_historical_SVH=True,counted_once_in_full_accuracy=True))
    status='PASS' if len(cases)==2 else 'RUNNING';save('MME_integration_smoke.json',dict(status=status,cases=cases,upstream_strict_numerical_smoke='PASS unchanged'))
    state['MME_integration_smoke']=status
   else:g,_=greedy(a,layout,ctl,128)
   scoring=score(s,g['text']);record=dict(key=s['key'],benchmark='MME',question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],task=s['task'],pair_id=s['pair_id'],
    condition=protocol['condition'],input_view='MME-paper',gold=s['answer'],sample_sha256=digest(s),binding=binding,
    prompt_sha256=r['prompt_sha256'],input_ids_sha256=r['input_ids_sha256'],pixel_values_FP32_sha256=input_binding['pixel_values_sha256'],
    same_input_binding_as_SVH=True,known_MME_image_difference=bool(s.get('known_recovered_image_version_difference')),
    generation=g,scoring=scoring,pruning_events=ctl.selection_events,at=now())
   payload=json.dumps(record,ensure_ascii=False,separators=(',',':'))
   with db:db.execute('INSERT INTO predictions VALUES(?,?,?)',(s['key'],payload,digest(record)))
   existing[s['key']]=record;ctl=None;a.inputs={};state.update(completed=len(existing),last_key=s['key'],at=now());save('progress.json',state)
   if len(existing)%100==0 or len(existing) in (2,2374):print('MME_PROGRESS',len(existing),'/2374',flush=True)
  assert len(existing)==2374 and read(O/'MME_integration_smoke.json')['status']=='PASS'
  state.update(status='COMPLETE',phase='MME_ACCURACY_DONE',at=now());save('progress.json',state)
 except BaseException as e:
  state.update(status='BLOCKED' if len(existing)<2 else 'PARTIAL',at=now(),error=repr(e))
  save('failure.json',dict(phase=state['phase'],error=repr(e),traceback=traceback.format_exc()[-10000:],at=now()))
  print('MME_STOPPED',traceback.format_exc(),flush=True)
 finally:
  with gzip.GzipFile(filename=str(O/'predictions.jsonl.gz'),mode='wb',mtime=0) as out:
   for s in samples:
    if s['key'] in existing:out.write((json.dumps(existing[s['key']],ensure_ascii=False,separators=(',',':'))+'\n').encode())
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();ctl=None;a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  state['process_exit_releases_CUDA_context']=True;save('progress.json',state);print('MME_FINAL_STATUS',state['status'],len(existing),'/2374',flush=True)
if __name__=='__main__':main()
