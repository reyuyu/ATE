from common import *
from types import SimpleNamespace
from contextlib import nullcontext
from dataclasses import replace
import torch,subprocess,platform,time,traceback,gc
from vlm_research.vlm_common import load_model
from vlm_research.physical_handoff_fastpath.engine import Layout,FastHandoff,native_cache_info
from engine import TerminalMiddleWriter as MiddleWriter
def gpu_check(own=False):
 g=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert GPU in g and '3090' in g
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
 assert not p if not own else all(int(v.strip())==os.getpid() for v in p.splitlines()),p
 return g
def footprint(m):return [(id(x),len(x._forward_hooks),len(x._forward_pre_hooks)) for x in m.modules()]
def inputs_for(a,s):
 b=read(OUT/'input_bindings.json')[str(s['question_id'])]
 assert sha(s['image_path'])==s['image_sha256']==b['image_sha256']
 inputs=torch.load(RUN/'prepared'/(str(s['question_id'])+'.pt'),map_location='cpu',weights_only=True)
 assert set(inputs)=={'input_ids','attention_mask','pixel_values'}
 for k,v in inputs.items():assert th(v)==b['tensors'][k]['sha256']
 a.inputs={k:v.to('cuda:0') for k,v in inputs.items()}
 return Layout(s['input_tokens'],tuple(s['image']),tuple(s['text']),tuple(s['relay']),tuple(s['question_positions']),26)
def kv_audit(a,cache,ctrl,l,method,steps):
 info=ctrl.cache.audit() if ctrl else native_cache_info(cache);assert len(info['layers'])==32
 pairs=ctrl.cache.entries if ctrl else list(zip(cache.key_cache,cache.value_cache))
 for i,(r,pair) in enumerate(zip(info['layers'],pairs)):
  image=576 if method in ('Native','NoOp','Dense') or i in KEEP else 0
  absent=method=='SVH-R' and 2<=i<=5
  expected=0 if absent else l.length-576+image+steps-1
  assert r['used_tokens']==expected,(method,i,r['used_tokens'],expected)
  r['image_tokens']=0 if absent else image
  if absent:assert pair is None and r['storage_bytes']==0;continue
  k,v=pair;assert k.dtype==v.dtype==torch.float16 and k.shape==v.shape and k.shape[:2]==(1,32) and k.shape[-1]==128
  used=r['used_tokens'];assert used<=k.shape[-2]
  assert r['live_bytes']==used*2*32*128*2
  assert r['storage_bytes']==sum(x.untyped_storage().nbytes() for x in pair)
  if 'original_positions' in r:
   positions=(list(range(l.length)) if image else list(l.text))+list(range(l.length,l.length+steps-1))
   assert positions==r['original_positions']
  r.update(K_shape=list(k.shape),V_shape=list(v.shape),dtype=str(k.dtype))
 return info
class Observer:
 def __init__(self,a,l,method,ctrl):self.a=a;self.l=l;self.method=method;self.ctrl=ctrl;self.step=0;self.rows={};self.handles=[]
 def start(self):
  for i,layer in enumerate(self.a.layers):
   def block(m,args,kw,i=i):
    h=args[0];pre=self.step==0;physical=self.method=='SVH-R'
    want=list(self.l.text) if pre and physical and i>=26 else list(range(self.l.length)) if pre else [self.l.length+self.step-1]
    assert kw['position_ids'][0].tolist()==want and h.shape[1]==len(want),(self.method,i,self.step)
    rope=self.a.model.language_model.model.rotary_emb(h,kw['position_ids'])
    assert all(torch.equal(x,y) for x,y in zip(rope,kw['position_embeddings']))
    self.rows[(self.step,i)]=dict(step=self.step,layer=i,hidden_rows=int(h.shape[1]),positions_sha256=th(kw['position_ids']),q_rows=0,k_rows=0,v_rows=0,o_rows=0,mlp_calls=0,norm_calls=0)
   self.handles.append(layer.register_forward_pre_hook(block,with_kwargs=True))
   for name in ('q_proj','k_proj','v_proj','o_proj'):
    def projection(m,args,name=name,i=i):
     self.rows[(self.step,i)][name[0]+'_rows']+=int(args[0].shape[-2])
    self.handles.append(getattr(layer.self_attn,name).register_forward_pre_hook(projection))
   for name in ('input_layernorm','post_attention_layernorm','mlp'):
    def count(m,args,name=name,i=i):self.rows[(self.step,i)]['mlp_calls' if name=='mlp' else 'norm_calls']+=1
    self.handles.append(getattr(layer,name).register_forward_pre_hook(count))
 def close(self): 
  for h in self.handles:h.remove()
  self.handles=[]
 def validate(self):
  for (step,i),r in self.rows.items():
   assert r['mlp_calls']==1 and r['norm_calls']==2
   n=r['hidden_rows'];skip=self.method=='SVH-R' and 2<=i<=5
   assert r['q_rows']==(n-1 if skip and step==0 else 0 if skip else n)
   assert r['o_rows']==r['q_rows']
   assert r['k_rows']==r['v_rows']==(0 if skip and step else n)
  return list(self.rows.values())
@torch.inference_mode()
def generate(a,l,method,max_tokens=32,fixed=False,capture=False,audit=False):
 ctrl=MiddleWriter(a,l,KEEP,max_tokens,audit,method=='Dense') if method in ('Dense','SVH-R') else FastHandoff(a,l,'Native-Noop',max_tokens,False) if method=='NoOp' else None
 before=footprint(a.model);obs=Observer(a,l,method,ctrl) if audit else None
 cache=None;ids=[];logits=[];mask=torch.ones((1,l.length+max_tokens-1),device='cuda:0',dtype=torch.long);positions=torch.arange(l.length+max_tokens-1,device='cuda:0')
 try:
  if obs:obs.start()
  with ctrl.context() if ctrl else nullcontext():
   for step in range(max_tokens):
    if obs:obs.step=step
    kw=dict(a.inputs) if not step else dict(input_ids=token.reshape(1,1),attention_mask=mask[:,:l.length+step])
    kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:l.length] if not step else positions[l.length+step-1:l.length+step],logits_to_keep=1)
    out=a.model(**kw);last=out.logits[:,-1];assert out.logits.shape[1]==1 and torch.isfinite(last).all()
    token=last.argmax(-1);ids.append(int(token));cache=out.past_key_values
    if capture:logits.append(last.detach().float().cpu().clone())
    del out,last
    if not fixed and ids[-1]==2:break
  info=kv_audit(a,cache,ctrl,l,method,len(ids));text=a.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
  result=dict(method=method,text=text,token_ids=ids,generated_tokens=len(ids),first_token=ids[0],eos=ids[-1]==2,truncated=len(ids)==max_tokens and ids[-1]!=2,empty=not bool(text),repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2)),all_logits_finite=True,real_decode_calls=len(ids)-1,KV=info,fixed=fixed,first_logits_sha256=th(logits[0]) if logits else None)
  traces=dict(physical_exit_events=[] if not ctrl else getattr(ctrl,'physical_exit_events',[]),events=[] if not ctrl else ctrl.audit_rows,visual=[] if not ctrl else getattr(ctrl,'visual_events',[]),operators=obs.validate() if obs else [])
  hidden=[] if not ctrl else ctrl.hidden
  return result,logits,hidden,traces
 finally:
  if obs:obs.close()
  del cache,ctrl
  assert footprint(a.model)==before,'Hook leakage'
def metrics(x,y):
 d=(x-y).double();rmse=float(d.square().mean().sqrt()/y.double().square().mean().sqrt().clamp_min(1e-12));cos=float(1-torch.nn.functional.cosine_similarity(x.double().reshape(1,-1),y.double().reshape(1,-1)).item())
 return dict(MaxAbs=float(d.abs().max()),MeanAbs=float(d.abs().mean()),relative_RMSE=rmse,cosine_error=cos)
def smoke(a,smoke_samples):
 records=[];numerical=[];operators=[];prob=[]
 for s in smoke_samples:
  l=inputs_for(a,s);qid=str(s['question_id'])
  n,nlog,_,_=generate(a,l,'Native',9,True,True)
  z,zlog,_,_=generate(a,l,'NoOp',9,True,True)
  assert n['token_ids']==z['token_ids'] and all(torch.equal(x,y) for x,y in zip(nlog,zlog)),'Native/NoOp not strictly equal'
  # Independent stock HF greedy generate checks the shared Native loop.
  official=a.model.generate(**a.inputs,max_new_tokens=32,do_sample=False,num_beams=1,use_cache=True,logits_to_keep=1)
  native,_,_,_=generate(a,l,'Native',32,False,False)
  assert official[0,l.length:].tolist()==native['token_ids'],'Native manual loop vs stock generate tokens'
  d,dl,dh,dt=generate(a,l,'Dense',9,True,True,True);p,pl,ph,pt=generate(a,l,'SVH-R',9,True,True,True)
  fixed_tokens_equal=d['token_ids']==p['token_ids']
  dense_full,_,_,_=generate(a,l,'Dense',32,False)
  physical_full,_,_,_=generate(a,l,'SVH-R',32,False)
  assert dense_full['token_ids']==physical_full['token_ids'],'Complete natural-EOS Smoke generation differs'
  valid_steps=len(dense_full['token_ids'])
  for step,(x,y) in enumerate(zip(dl,pl)):
   v=metrics(x,y);passed=bool(torch.allclose(x,y,atol=.10,rtol=.003));v.update(qid=qid,type='logits',step=step,numerical_scope='natural_EOS_valid' if step<valid_steps else 'synthetic_post_EOS_structure_only',original_threshold_pass=passed);numerical.append(v)
   if step<valid_steps:assert passed,(qid,step,v)
   # The requested quality protocol stops at EOS. Synthetic post-EOS steps
   # remain finite/position/cache/shape tests; any numeric failure is retained.
  for i,(x,y) in enumerate(zip(dh,ph)):
   v=metrics(x,y);v.update(qid=qid,type='text_hidden',layer=i);numerical.append(v)
   assert v['relative_RMSE']<=.01 and v['cosine_error']<=1e-4,(qid,i,v)
  for row in pt['operators']:operators.append(dict(qid=qid,**row))
  for row in pt['visual']:prob.append(dict(qid=qid,**row))
  assert len(pt['operators'])==32*9 and len(p['KV']['layers'])==32
  assert len(pt['physical_exit_events'])==1 and pt['physical_exit_events'][0]['before_layer']==26 and pt['physical_exit_events'][0]['removed']==576
  assert all(x['hidden_rows']==(l.length if x['layer']<26 else l.length-576) for x in pt['operators'] if x['step']==0)
  records.append(dict(qid=qid,status='PASS',Native_NoOp_valid_logits_torch_equal=True,stock_HF_generation_matches_Native_loop=True,Dense_Physical_fixed9_tokens_equal=fixed_tokens_equal,Dense_Physical_complete_tokens_equal=True,valid_natural_EOS_steps=valid_steps,synthetic_post_EOS_original_threshold_failures=[v for v in numerical if v['qid']==qid and v['type']=='logits' and not v['original_threshold_pass']],physical_exit_events=pt['physical_exit_events'],full_tokens=physical_full['token_ids'],text=physical_full['text'],max_logit_absolute=max(v['MaxAbs'] for v in numerical if v['qid']==qid and v['type']=='logits'),physical_KV=p['KV']))
 write(OUT/'smoke_validation.json',dict(status='PASS_NATURAL_EOS_QUALITY_AND_9STEP_STRUCTURE',numerical_policy='same parent v2: unchanged thresholds on natural-EOS valid steps; all9 forced steps structural/finite; retain postEOS numeric failures',samples=records,thresholds=read(OUT/'protocol.json')['smoke_gate'],structural=True))
 csvwrite(OUT/'smoke_numerical.csv',numerical);csvwrite(OUT/'smoke_operator_audit.csv',operators);write(OUT/'smoke_probability_audit.json',prob)
def run():
 gpu=gpu_check();verify_sources();assert read(OUT/'cpu_tests.json')['status']=='PASS'
 samples=read(OUT/'sample_manifest.json');db=connect();before_count=db.execute('select count(*) from predictions').fetchone()[0]
 loaded=0;model=None
 try:
  model,processor,environment=load_model(str(MODEL));loaded+=1;assert len(model.language_model.model.layers)==32
  a=SimpleNamespace(kind='llava',model=model,processor=processor,layers=model.language_model.model.layers,inputs=None)
  params=[(p._version,p.data_ptr()) for p in model.parameters()]
  import transformers
  from transformers.models.llama import modeling_llama
  from transformers.models.llava import modeling_llava
  environment.update(GPU=gpu,torch_path=torch.__file__,transformers_path=transformers.__file__,python_path=sys.executable,HF_source={str(Path(m.__file__)):sha(m.__file__) for m in (modeling_llama,modeling_llava)},cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),official_model_parameters_unchanged=True)
  write(OUT/'environment.json',environment)
  smoke(a,samples['Smoke']);gpu_check(True)
  old=sqlite3.connect('file:/home/featurize/ate-runs/llava-svh-r-textvqa128-3090/predictions.sqlite?mode=ro',uri=True)
  calibrations=[]
  byid={str(x['question_id']):x for x in samples['Main']}
  for qid in read(OUT/'protocol.json')['Native_reuse_calibration_ids']:
   sample=byid[qid];layout=inputs_for(a,sample)
   current,_,_,_=generate(a,layout,'Native')
   checksum,raw=old.execute("select checksum,record from predictions where qid=? and method='Native'",(qid,)).fetchone();history=json.loads(raw)
   assert digest(history)==checksum and history['input_binding']==sample['input_binding']
   assert history['token_ids']==current['token_ids'] and history['text']==current['text'],('Native reuse calibration failed',qid)
   calibrations.append(dict(qid=qid,status='PASS',complete_tokens=current['token_ids'],text=current['text'],parent_checkpoint_checksum=checksum))
  old.close();write(OUT/'Native_reuse_calibration.json',dict(status='PASS',records=calibrations,parent_commit='177c631521d5c9d0436768e11e529ec492c2cbc8'))
  for i,s in enumerate(samples['Main']):
   layout=inputs_for(a,s);qid=str(s['question_id'])
   for label in LABELS:
    old=db.execute('select checksum,record from predictions where qid=? and method=?',(qid,label)).fetchone()
    if old:assert digest(json.loads(old[1]))==old[0];continue
    r,_,_,_=generate(a,layout,'SVH-R');r['method']=label
    r.update(question_id=qid,image_sha256=s['image_sha256'],input_binding=s['input_binding'],prompt_sha256=s['prompt_sha256'],protocol_sha256=sha(OUT/'protocol.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),GPU_UUID=GPU)
    with db:db.execute('insert into predictions values(?,?,?,?)',(qid,label,digest(r),json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))))
   if (i+1)%8==0 or i==127:
    assert params==[(p._version,p.data_ptr()) for p in model.parameters()];gpu_check(True);verify_sources()
    n=db.execute('select count(*) from predictions').fetchone()[0];write(OUT/'progress.json',dict(status='RUNNING',completed=n,total=128,paired_questions=i+1));print(json.dumps(dict(paired=i+1,records=n,total=128)),flush=True)
  assert db.execute('select count(*) from predictions').fetchone()[0]==128
  write(OUT/'run_completion.json',dict(status='COMPLETE',formal_records=128,new_records=128-before_count,model_loads=loaded,source_manifest_sha256=verify_sources(),GPU_UUID=GPU))
 except BaseException as e:
  write(OUT/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  db.close()
  if model is not None:
   a.inputs=None;a.model=None;a.layers=None;del model,a;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  write(OUT/'gpu_release.json',dict(model_objects_deleted=True,memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),UUID=GPU,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':run()
