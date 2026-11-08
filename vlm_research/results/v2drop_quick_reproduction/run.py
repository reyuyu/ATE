"""Single GPU/model load: gated smoke, 400 exploratory generations, efficiency."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
import sys,json,gzip,hashlib,sqlite3,subprocess,time,traceback,gc,fcntl
import torch,transformers
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_quick_reproduction';D=Path('/home/featurize/data/ate-recovery');M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(O),str(R)]
from vlm_research.information_horizon_alignment import engine as frozen,score as scorer
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3.common import prediction
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from runtime import generate,Layout

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(name,x):
 p=O/name;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
def score(s,text):
 if s['benchmark']=='GQA':
  parsed=prediction(text);return {'parsed':parsed,'score':float(parsed==s['answer'])}
 result=scorer.score(s,text);e=scorer.text_evaluator();p=e.answer_processor(text);answers=[e.answer_processor(v) for v in s['answer']]
 assert len(answers)==10;count=answers.count(p)
 independent=sum(min((count-int(v==p))/3,1) for v in answers)/10
 assert abs(result['score']-independent)<1e-12,'Independent VQA soft-score mismatch'
 return result
def verify(p):
 assert cmd('git','rev-parse','HEAD')==p['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
 for name,h in p['frozen_ATE_sources'].items():assert sha(R/name)==h,name
 assert sha(O/'input_manifest.jsonl.gz')==p['manifest_sha256']
def write(db,table,key,record):
 payload=json.dumps(record,ensure_ascii=False,separators=(',',':'))
 with db:db.execute('INSERT INTO '+table+' VALUES (?,?,?)',(key,payload,digest(record)))
def load_table(db,table):
 result={}
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM '+table):
  r=json.loads(payload);assert digest(r)==h;result[key]=r
 return result

def main():
 guard=(O/'run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(O/'protocol.json');verify(p)
 with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 assert len(rows)==200 and len({s['key'] for s in rows})==200
 bykey={s['key']:s for s in rows};groot=O.parent/'featurize_4090_gqa_official_aligned_threeway';questions={}
 with gzip.open(groot/'input_manifest.jsonl.gz','rt') as f:
  for s in map(json.loads,f):
   if 'GQA:'+s['question_id'] in bykey:questions[s['question_id']]=s['question_positions']
 controls={}
 with sqlite3.connect('file:'+str(groot/'results.sqlite')+'?mode=ro',uri=True) as old:
  for (payload,) in old.execute('SELECT payload FROM results'):
   r=json.loads(payload)
   if 'GQA:'+r['question_id'] in bykey:controls[r['question_id'],r['condition']]=r
 assert len(controls)==300
 config={'protocol_sha256':sha(O/'protocol.json'),'sources':{n:sha(O/n) for n in ['run.py','runtime.py','hf_adapter.py','prepare.py']},'frozen_ATE_unchanged':True}
 if (O/'execution_config.json').exists():assert read(O/'execution_config.json')==config
 else:save('execution_config.json',config)
 db=sqlite3.connect(O/'results.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 for table in ('predictions','timing','counts'):db.execute('CREATE TABLE IF NOT EXISTS '+table+' (key TEXT PRIMARY KEY,payload TEXT,sha256 TEXT)')
 db.commit();predictions=load_table(db,'predictions');timings=load_table(db,'timing');counts=load_table(db,'counts')
 state={'status':'RUNNING','phase':'GPU_LOAD','started_at':now(),'pid':os.getpid(),'model_loads':1,'accuracy_completed':len(predictions),'accuracy_expected':400,
        'efficiency_completed':len(timings),'efficiency_expected':240,'counter_completed':len(counts),'counter_expected':120,'protocol_sha256':config['protocol_sha256']}
 a=None;started=time.perf_counter()
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU already occupied'
  assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090' and torch.cuda.mem_get_info()[0]>20*2**30
  assert transformers.__version__=='4.49.0';state['GPU_before']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader');save('progress.json',state)
  loader.MODELS=dict(loader.MODELS,llava=M);a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
  scorer.AUTHOR=D/'sources/Information-Horizon';scorer.text_evaluator()
  pixels={}
  def prepare(s):
   if s['image_sha256'] not in pixels:pixels[s['image_sha256']]=frozen.paper_pixels(s['image_path'],a.image_processor).half()
   x={'input_ids':torch.tensor([s['input_ids']],dtype=torch.long),'attention_mask':torch.ones((1,len(s['input_ids'])),dtype=torch.long),'pixel_values':pixels[s['image_sha256']]}
   assert {k:tensor_hash(v) for k,v in x.items()}==s['input_hashes'];a.inputs={k:v.to('cuda:0') for k,v in x.items()}
   image=tuple(i for i,v in enumerate(s['input_ids']) if v==32000);text=tuple(i for i,v in enumerate(s['input_ids']) if v!=32000);relay=tuple(range(image[-1]+1,len(s['input_ids'])-1))
   q=tuple(questions[s['question_id']]) if s['benchmark']=='GQA' else relay
   return Layout(len(s['input_ids']),image,text,relay,q,20)
  state['phase']='SMOKE';save('progress.json',state);smoke=[]
  for idx,key in enumerate(p['smoke_keys']):
   s=bykey[key];l=prepare(s)
   native,nl=generate(a,l,'Native',4,fixed=True,capture=True)
   disabled,dl=generate(a,l,'V2Drop-NoPrune',4,fixed=True,capture=True,trace=True)
   assert native['token_ids']==disabled['token_ids'] and len(nl)==len(dl)==4 and all(torch.equal(x,y) for x,y in zip(nl,dl))
   pruned,pl=generate(a,l,'V2Drop',4,fixed=True,capture=True,trace=True,reference=True)
   assert [r['visual_after'] for r in pruned['pruning_events']]==[288,144,1]
   assert [r['visual_before'] for r in pruned['pruning_events']]==[576,288,144]
   assert sum(r['visual_tokens'] for r in pruned['cache']['layers'])==6490
   if idx==0:
    hf=a.model.generate(**a.inputs,max_new_tokens=4,do_sample=False,num_beams=1,logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
    ids=hf.sequences[0,l.length:].tolist();assert native['token_ids'][:len(ids)]==ids
    assert all(torch.equal(x,y.float().cpu()) for x,y in zip(nl,hf.scores));del hf
    plain,_=generate(a,l,'V2Drop',4,fixed=True,capture=True)
    assert plain['token_ids']==pruned['token_ids']
   smoke.append({'key':key,'native_no_prune_tokens_and_logits_exact':True,'official_selection_kernel_exact':True,'text_positions_preserved':True,
                 'mask_positions_RoPE_KV':'PASS','image_start':s['image_token_start'],'native':native,'pruned':pruned,'scorer':score(s,pruned['text'])})
   print('SMOKE_PASS',key,flush=True)
  save('smoke_full.json',smoke);save('smoke_summary.json',{'status':'PASS','images':len(smoke),'checks':[{k:r[k] for k in ['key','native_no_prune_tokens_and_logits_exact','official_selection_kernel_exact','text_positions_preserved','mask_positions_RoPE_KV','image_start']} for r in smoke],
    'physical_visual_counts':[576,288,144,1],'actual_visual_KV_per_layer':[576]*3+[288]*14+[144]*5+[1]*10,'visual_KV_total_tokens':6490,'Native_visual_KV_total_tokens':18432,'no_dense_mask_substitution':True})
  methods=['Native','V2Drop','C3-V2','C3-V2+L11-L24']
  try:
   for key in (p['smoke_keys'][0],p['smoke_keys'][2]):
    s=bykey[key];l=prepare(s)
    for method in methods[2:]:
     g,_=generate(a,l,method,4,fixed=True)
     assert g['cache']['visual_KV_residency_percent']==(43.75 if '+' in method else 0)
     if s['benchmark']=='GQA':
      old=controls[s['question_id'],method]['generation']['token_ids'];assert g['token_ids'][:min(4,len(old))]==old[:4]
   state['optional_ATE_efficiency']='PASS; unchanged frozen MiddleWriter reused'
  except Exception as exc:
   if isinstance(exc,torch.cuda.OutOfMemoryError) or 'Nonfinite' in str(exc):raise
   methods=methods[:2];state['optional_ATE_efficiency']='SKIPPED: '+repr(exc)
   state['efficiency_expected']=120;state['counter_expected']=60
  state['executed_efficiency_methods']=methods;state['phase']='ACCURACY';save('progress.json',state)
  for s in rows:
   for method in ('Native','V2Drop'):
    key=s['key']+'|'+method
    if key in predictions:continue
    l=prepare(s);g,_=generate(a,l,method,s['max_new_tokens'])
    if method=='Native' and s['benchmark']=='GQA':assert g['token_ids']==controls[s['question_id'],'Native']['generation']['token_ids'],'Native historical same-input mismatch'
    r={'key':s['key'],'benchmark':s['benchmark'],'question_id':s['question_id'],'condition':method,'sample_sha256':s['sample_sha256'],'input_hashes':s['input_hashes'],'generation':g,**score(s,g['text'])}
    write(db,'predictions',key,r);predictions[key]=r;state['accuracy_completed']=len(predictions)
   if len(predictions)%20==0:state['updated_at']=now();save('progress.json',state);print('ACCURACY',len(predictions),400,flush=True)
  assert len(predictions)==400;verify(p)
  state['phase']='EFFICIENCY_WARMUP';save('progress.json',state)
  first=bykey[p['efficiency_keys'][0]]
  for method in methods:
   for length in (32,128):generate(a,prepare(first),method,length,fixed=True)
  state['phase']='EFFICIENCY';save('progress.json',state)
  for index,key in enumerate(p['efficiency_keys']):
   s=bykey[key];order=methods[index%len(methods):]+methods[:index%len(methods)];lengths=(32,128) if index%2==0 else (128,32)
   for length in lengths:
    for method in order:
     item=key+'|'+method+'|'+str(length)
     if item in timings:continue
     g,_=generate(a,prepare(s),method,length,fixed=True)
     assert len(g['token_ids'])==length
     r={'key':key,'benchmark':s['benchmark'],'condition':method,'length':length,'input_hashes':s['input_hashes'],'sample_sha256':s['sample_sha256'],'result':g}
     write(db,'timing',item,r);timings[item]=r;state['efficiency_completed']=len(timings)
   for method in order:
    item=key+'|'+method
    if item in counts:continue
    g,_=generate(a,prepare(s),method,128,fixed=True,count=True,trace='layout' if method=='V2Drop' else False)
    for length in (32,128):
     expected=timings[item+'|'+str(length)]['result']['token_ids'];assert g['token_ids'][:length]==expected,'Counter changed greedy outputs'
     assert g['counter_checkpoints'][str(length)]['full_request_flops']>g['counter_checkpoints'][str(length)]['prefill_flops']>g['counter_checkpoints'][str(length)]['vision_flops']>0
    r={'key':key,'benchmark':s['benchmark'],'condition':method,'input_hashes':s['input_hashes'],'result':g}
    write(db,'counts',item,r);counts[item]=r;state['counter_completed']=len(counts)
   state['updated_at']=now();save('progress.json',state);print('EFFICIENCY_INPUT',index+1,30,len(timings),len(counts),flush=True)
  assert len(timings)==state['efficiency_expected'] and len(counts)==state['counter_expected'];verify(p)
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');state['status']='GENERATIONS_AND_MEASUREMENTS_COMPLETE';state['phase']='AWAITING_CPU_SUMMARY'
 except BaseException as exc:
  state.update(status='BLOCKED',failure={'error':repr(exc),'traceback':traceback.format_exc()[-10000:]});print('STOPPED',traceback.format_exc(),flush=True)
 finally:
  state.update(finished_at=now(),elapsed_seconds=time.perf_counter()-started)
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  db.close();save('progress.json',state);print('RUN_FINAL',state['status'],flush=True)
 if state['status']=='BLOCKED':raise SystemExit(1)

if __name__=='__main__':main()
