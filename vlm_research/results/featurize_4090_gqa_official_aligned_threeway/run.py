"""One GPU/model; audited physical smoke, then durable sequential GQA cohorts."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime,timezone
import sys,json,gzip,hashlib,sqlite3,subprocess,time,traceback,gc,fcntl,signal
import torch,transformers
from PIL import Image
from types import SimpleNamespace
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_4090_gqa_official_aligned_threeway';A=O.parent/'featurize_4090_gqa_native_protocol_audit'
D=Path('/home/featurize/data/ate-recovery');M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(A),str(R)]
from protocol_helpers import official_functions,make_inputs
from transformers import LlamaTokenizer
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3 import engine as engine,common
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.fastp2_bypass_writer.run import operators
from vlm_research.physical_visual_handoff.engine import equivalence
from vlm_research.physical_visual_handoff.common import TOLERANCES
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
def save(n,x):
 p=O/n;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def stopcheck():
 assert not (O/'STOP_REQUESTED').exists(),'Stop requested'
 assert os.statvfs(O).f_bavail*os.statvfs(O).f_frsize>2*2**30,'Result filesystem safety floor'
def verify_sources(p):
 for name,h in p['frozen_sources'].items():assert sha(R/name)==h,name
 assert sha(A/'protocol_helpers.py')==p['audit_helper_sha256']
 assert cmd('git','rev-parse','HEAD')==p['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
def hashes(x):return {k:tensor_hash(v.half() if v.is_floating_point() else v) for k,v in x.items()}
def layout_from(ids,question,cutoff=20):
 image=tuple(i for i,t in enumerate(ids) if t==32000);text=tuple(i for i,t in enumerate(ids) if t!=32000);relay=tuple(range(image[-1]+1,len(ids)-1))
 assert len(image)==576 and question and set(question)<=set(relay)
 return Layout(len(ids),image,text,relay,tuple(question),cutoff)

@contextmanager
def structural_trace(a,l):
 out={'vision_calls':0,'projector_calls':0,'image_embedding_exact':False,'layer_events':[],
  'norm_calls':[0]*32,'post_norm_calls':[0]*32,'mlp_calls':[0]*32,'skip_residual_checks':0}
 handles=[];projected=[];before={};n=l.length;image_mask=a.inputs['input_ids'][0]==32000
 def vision(m,args,v):out['vision_calls']+=1;assert v.last_hidden_state.is_cuda
 def project(m,args,v):out['projector_calls']+=1;assert list(v.shape)==[1,576,4096];projected.append(v.detach())
 def language(m,args,kw):
  if projected:assert torch.equal(kw['inputs_embeds'][0,image_mask],projected.pop().reshape(576,4096));out['image_embedding_exact']=True
 try:
  handles.extend([a.model.vision_tower.register_forward_hook(vision),a.model.multi_modal_projector.register_forward_hook(project),a.model.language_model.register_forward_pre_hook(language,with_kwargs=True)])
  for i,layer in enumerate(a.layers):
   def pre(m,args,kw,i=i):
    step=sum(e['layer']==i for e in out['layer_events']);pos=list(range(n)) if step==0 and i<l.cutoff else list(l.text) if step==0 else [n+step-1]
    assert args[0].shape[1]==len(pos) and kw['cache_position'].tolist()==pos and kw['position_ids'][0].tolist()==pos
    if i in (2,3,4,5):before[i]=args[0].detach().clone()
    out['layer_events'].append({'layer':i,'step':step,'rows':len(pos),'original_positions_sha256':digest(pos)})
   def norm(m,args,v,i=i):out['norm_calls'][i]+=1
   def post(m,args,i=i):
    out['post_norm_calls'][i]+=1
    if i in (2,3,4,5):assert torch.equal(args[0][:,-1],before.pop(i)[:,-1]);out['skip_residual_checks']+=1
   def mlp(m,args,v,i=i):out['mlp_calls'][i]+=1
   handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm.register_forward_hook(norm),layer.post_attention_layernorm.register_forward_pre_hook(post),layer.mlp.register_forward_hook(mlp)])
  yield out
 finally:
  for h in handles:h.remove()
  projected.clear();before.clear()

def migration_references(p):
 wanted=set(p['cross_hardware_old_protocol_question_ids']);refs={}
 H7=O.parent/'featurize_new3090_c3v2_reaccess_sevenbench'
 with gzip.open(H7/'gqa_historical_controls.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);q=r['question_id']
   if q in wanted:refs[q,'C3-V2']=r['controls']['C3-V2']
   if len(refs)==len(wanted):break
 with gzip.open(H7/'gqa_verified_records.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line);r=row['record'];q=r['question_id']
   if q in wanted:
    assert common.digest(r)==row['record_sha256'] and r['condition']=='C3-V2+L11-L24'
    refs[q,r['condition']]=dict(r['generation'],input_binding=r['input_binding'])
   if len(refs)==2*len(wanted):break
 assert set(refs)=={(q,label) for q in wanted for label in p['conditions'][1:]}
 return refs

def physical_smoke(a,entries,p,prepare,refs):
 byid={s['question_id']:s for s in entries};checks=[];migrations=[]
 # Independent migration reference: old inputs and complete saved token outputs.
 H=O.parent/'featurize_new3090_recovery/history/c3v2-gqa-testdev-balanced-12578/vlm_research/results/c3v2_gqa_testdev_balanced_12578'
 with gzip.open(H/'sample_manifest.jsonl.gz','rt') as f:old={s['question_id']:s for s in map(json.loads,f)}
 conv,functions=official_functions();tok=LlamaTokenizer.from_pretrained(D/'audit_support/gqa_native_protocol/official_tokenizer',local_files_only=True,legacy=True)
 for q in p['cross_hardware_old_protocol_question_ids']:
  s=dict(old[q],image_path=byid[q]['image_path']);x,_=make_inputs(s,'HISTORICAL',a.processor,tok,conv,functions)
  for label in p['conditions'][1:]:
   ref=refs[q,label];assert hashes(x)==ref['input_binding']['input_hashes']
   a.inputs={k:v.to('cuda:0',dtype=torch.float16) if v.is_floating_point() else v.to('cuda:0') for k,v in x.items()};l=layout_from(x['input_ids'][0].tolist(),s['question_positions'])
   generated=engine.generate(a,l,label,32)[0];assert generated['token_ids']==ref['token_ids'] and generated['text']==ref['text'],('Cross-hardware migration mismatch',q,label)
   migrations.append({'question_id':q,'condition':label,'historical_text':ref['text'],'current_text':generated['text'],'token_ids_exact':True,'input_tensors_exact':True,'source_commit':'57e37ce024b1b2adf9fa24fc24aabec32a1fed08'})
   print('MIGRATION_MATCH',q,label,repr(generated['text']),flush=True)
 for q in p['physical_smoke_question_ids']:
  s=byid[q];l=prepare(s)
  if not checks:
   native=engine.generate(a,l,'Native',32,capture=True)
   with torch.inference_mode():hf=a.model.generate(**a.inputs,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
   assert hf.sequences[0,l.length:].tolist()==native[0]['token_ids']
   assert len(hf.scores)==len(native[1]) and all(torch.equal(x,y.float().cpu()) for x,y in zip(native[1],hf.scores))
   save('native_smoke.json',{'status':'PASS','question_id':q,'generation':native[0],'HF_generate_tokens_and_scores_exact':True})
   del native,hf
  for label in p['conditions'][1:]:
   keep=common.KEEP[label];cutoff=common.CUTOFF[label];ll=replace(l,cutoff=cutoff);before=hashes(a.inputs)
   with operators(a) as (calls,mlp,shapes),structural_trace(a,ll) as trace:
    actual=engine.generate(a,ll,label,3,capture=True,audit=True,fixed=True)
   g=actual[0];steps=len(g['token_ids']);assert steps==3
   assert trace['vision_calls']==trace['projector_calls']==1 and trace['image_embedding_exact']
   assert trace['norm_calls']==trace['post_norm_calls']==trace['mlp_calls']==[steps]*32 and trace['skip_residual_checks']==4*steps
   for i in range(32):assert mlp[i]==steps
   for i in (2,3,4,5):
    assert all(calls[i,op]==1 for op in ('q_proj','k_proj','v_proj','o_proj'))
    assert {r['op']:r['shape'][1] for r in shapes if r['layer']==i}=={'q_proj':ll.length-1,'k_proj':ll.length,'v_proj':ll.length,'o_proj':ll.length-1}
   assert all(r['blocked_probability_max']==0 for r in g['attention_audit'])
   events=g['visual_audit'];assert events
   assert all(e['image_keys_permitted']==(e['layer'] in keep) and ((e['answer_image_mass']>0)==(e['layer'] in keep)) for e in events)
   assert all(e['relay_image_mass']==0 for e in events if e['prefill'] and e['layer']>=20)
   assert any(e['relay_image_mass']>0 for e in events if e['prefill'] and e['layer']<20)
   assert all(not e['image_keys_present'] for e in events if (not e['prefill'] and e['layer'] not in keep) or (e['prefill'] and e['layer']>=cutoff))
   kv=g['kv_audit'];assert kv['retained_image_layers']==list(keep) and kv['historical_cache_copies']==kv['append_concatenations']==0
   for row in kv['layers']:
    if row['layer'] in (2,3,4,5):assert row['used_tokens']==row['storage_bytes']==0
    else:
     expected=(list(range(ll.length)) if row['layer'] in keep else list(ll.text))+list(range(ll.length,ll.length+steps-1))
     assert row['original_positions']==expected and row['used_tokens']==len(expected)
   # Same-semantics numerical reference only, never a formal dense cohort.
   dense=engine.generate(a,ll,label,3,capture=True,audit=True,dense=True,fixed=True)
   gate=equivalence('llava',dense[1],actual[1],dense[3],actual[3])
   assert g['token_ids']==dense[0]['token_ids'] and gate['all_argmax_equal'],'Short output mismatch'
   assert len(actual[4])==len(dense[4])==3
   assert all(x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x) for x,y in zip(actual[4],dense[4]))
   assert hashes(a.inputs)==before,'Input changed during physical execution'
   record={'question_id':q,'condition':label,'structural_status':'PASS','strict_numerical_status':'PASS' if gate['passed'] else 'FAIL','numerical_equivalence':gate,
    'short_tokens':g['token_ids'],'short_text':g['text'],'short_output_dense_exact':True,'rope_exact':True,'input_tensors_unchanged':True,
    'visual_hidden_exit':cutoff,'retained_visual_KV_layers':list(keep),'attention_L2_L5_physically_skipped':True,'norm_MLP_residual_checks':'PASS',
    'trace':trace,'projection_shapes':shapes,'visual_audit':events,'KV_audit':kv,'all_logits_finite':g['all_effective_logits_finite']}
   checks.append(record);save('physical_smoke.json',{'status':'RUNNING','checks':checks,'cross_hardware_migration':migrations,'strict_thresholds':TOLERANCES['llava']})
   print('PHYSICAL_CHECK',q,label,json.dumps(gate),flush=True)
   del actual,dense
 result={'status':'AWAITING_REVIEW','checks':checks,'cross_hardware_migration':migrations,'strict_thresholds':TOLERANCES['llava'],'historical_strict_failure_preserved':'featurize_4090_c3v2_handoff/physical_smoke.json'}
 save('physical_smoke.json',result);return result

def main():
 lock=(O/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(O/'protocol.json');binding=sha(O/'protocol.json');verify_sources(p)
 assert sha(O/'input_manifest.jsonl.gz')==p['manifest_sha256'] and torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:entries=[json.loads(l) for l in f]
 assert len(entries)==len({s['question_id'] for s in entries})==12578
 refs=migration_references(p)
 config={'protocol_sha256':binding,'runner_sha256':sha(__file__),'prepare_sha256':sha(O/'prepare.py'),'cluster_bootstrap':{'unit':'accepted Image SHA256','resamples':10000,'seed':20261108},'strict_thresholds_unchanged':TOLERANCES['llava'],
  'historical_reaccess_records_sha256':sha(O.parent/'featurize_new3090_c3v2_reaccess_sevenbench/gqa_verified_records.jsonl.gz')}
 if (O/'execution_config.json').exists():assert read(O/'execution_config.json')==config
 else:save('execution_config.json',config)
 db=sqlite3.connect(O/'results.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS results (qid TEXT, condition TEXT, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(qid,condition))');db.commit()
 old={}
 for payload,h in db.execute('SELECT payload,sha256 FROM results'):
  r=json.loads(payload);assert digest(r)==h and r['protocol_sha256']==binding;old[r['question_id'],r['condition']]=r
 assert len(old)<=37734
 previous=read(O/'progress.json');state={'status':'RUNNING','phase':'GPU_LOAD','started_at':now(),'pid':os.getpid(),'protocol_sha256':binding,'expected':37734,
  'completed':len(old),'counts':{label:sum(c==label for q,c in old) for label in p['conditions']},'correct':{label:sum(r['score'] for (q,c),r in old.items() if c==label) for label in p['conditions']},'model_loads':previous.get('model_loads',0)+1}
 a=None;vision=[0];handle=None;started=time.monotonic();phase='GPU_LOAD'
 try:
  stopcheck();assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')
  assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090' and torch.cuda.mem_get_info()[0]>20*2**30
  state['GPU_before_load']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader');save('progress.json',state)
  loader.MODELS=dict(loader.MODELS,llava=M);torch.cuda.reset_peak_memory_stats();a=frozen.load('B')
  assert {str(v.device) for v in a.model.parameters()}=={'cuda:0'} and {v.dtype for v in a.model.parameters()}=={torch.float16}
  assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
  def observe(m,args,out):assert out.last_hidden_state.is_cuda;vision[0]+=1
  handle=a.model.vision_tower.register_forward_hook(observe)
  conv,functions=official_functions();pixels={}
  def prepare(s):
   if s['image_id'] not in pixels:
    with Image.open(s['image_path']) as im:v=functions.process_images([im.convert('RGB')],a.processor.image_processor,SimpleNamespace(image_aspect_ratio='pad')).half()
    assert tensor_hash(v)==s['input_hashes']['pixel_values'];pixels[s['image_id']]=v
   x={'input_ids':torch.tensor([s['input_ids']],dtype=torch.long),'attention_mask':torch.ones((1,len(s['input_ids'])),dtype=torch.long),'pixel_values':pixels[s['image_id']]}
   assert hashes(x)==s['input_hashes'];a.inputs={k:v.to('cuda:0') for k,v in x.items()}
   return layout_from(s['input_ids'],s['question_positions'])
  phase='PHYSICAL_SMOKE';state['phase']=phase;save('progress.json',state)
  if not (O/'physical_smoke.json').exists() or read(O/'physical_smoke.json')['status']!='AWAITING_REVIEW':physical_smoke(a,entries,p,prepare,refs)
  phase='AWAITING_REVIEW';state['phase']=phase;save('progress.json',state)
  print('AWAITING_AGENT_PHYSICAL_REVIEW',flush=True)
  while not (O/'formal_release.json').exists():stopcheck();time.sleep(1)
  release=read(O/'formal_release.json');assert release['protocol_sha256']==binding and release['physical_smoke_sha256']==sha(O/'physical_smoke.json') and release['allow_formal']
  labels=release['conditions'];assert labels==p['conditions'],'Only the three authorized conditions in frozen order'
  phase='FORMAL';state['phase']=phase;state['formal_started_at']=now();save('progress.json',state)
  fresh=0;formalstart=time.monotonic()
  for label in labels:
   state['current_condition']=label;save('progress.json',state);print('FORMAL_CONDITION_START',label,state['counts'][label],flush=True)
   for s in entries:
    stopcheck();q=s['question_id']
    if (q,label) in old:
     r=old[q,label];assert r['sample_binding']==s['sample_binding'] and r['input_hashes']==s['input_hashes'] and r['score']==common.score(s,r['generation']['text']);continue
    l=prepare(s);before=vision[0];g=engine.generate(a,l,label,32)[0]
    assert vision[0]==before+1 and g['all_effective_logits_finite']
    r={'index':s['index'],'question_id':q,'condition':label,'image_id':s['image_id'],'image_sha256':s['image_sha256'],'gold':s['answer'],
     'question':s['question'],'sample_binding':s['sample_binding'],'protocol_sha256':binding,'input_hashes':s['input_hashes'],'prompt_sha256':s['prompt_sha256'],
     'generation':g,'parsed':common.prediction(g['text']),'score':common.score(s,g['text']),'vision_forward_calls':1,'at':now()}
    raw=json.dumps(r,ensure_ascii=False,separators=(',',':'))
    with db:db.execute('INSERT INTO results VALUES (?,?,?,?)',(q,label,raw,digest(r)))
    old[q,label]=r;fresh+=1;state['counts'][label]+=1;state['correct'][label]+=r['score'];state['completed']+=1
    if state['completed']%25==0 or state['counts'][label]==12578:
     state.update(last_question_id=q,last_index=s['index'],updated_at=now(),elapsed_seconds=time.monotonic()-started,fresh_formal_generations=fresh,formal_generations_per_second=fresh/max(time.monotonic()-formalstart,1e-6))
     save('progress.json',state)
    if state['completed']%250==0:print('FORMAL_PROGRESS',state['completed'],37734,label,state['counts'][label],round(state.get('formal_generations_per_second',0),2),flush=True)
    if state['completed']%500==0:verify_sources(p)
    a.inputs={}
   assert state['counts'][label]==12578;print('FORMAL_CONDITION_COMPLETE',label,12578,'CORRECT',state['correct'][label],flush=True)
  assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==37734
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');state['status']='GENERATIONS_COMPLETE';state['phase']='AWAITING_CPU_SCORING'
 except BaseException as e:
  state.update(status='BLOCKED',phase=phase,failure={'error':repr(e),'traceback':traceback.format_exc()[-12000:]});print('STOPPED',traceback.format_exc(),flush=True)
 finally:
  if handle:handle.remove()
  state.update(finished_at=now(),vision_forward_calls_this_process=vision[0],elapsed_seconds=time.monotonic()-started)
  if torch.cuda.is_initialized():state.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  db.close();save('progress.json',state);print('RUN_FINAL',state['status'],state['completed'],flush=True)
 if state['status']=='BLOCKED':raise SystemExit(1)

if __name__=='__main__':main()
