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
 hooks=hook_counts(model);started=time.monotonic();peak=0
 def call(x,l,c,**kw):
  nonlocal peak
  r,v=generate(model,processor,x,l,c,**kw);assert hook_counts(model)==hooks,'Hook leak'
  peak=max(peak,r['memory']['peak_allocated']);return r,v
 def hb(s,l):return dict(**binding,question_id=s['question_id'],input_SHA256=digest(l),sample_SHA256=digest(s))
 # Structural checks must pass before any formal/exploratory quality generation.
 validation=[]
 ordered=sorted(rows,key=lambda s:(historical['input_bindings'][s['question_id']]['input_length'],s['question_id']))
 smoke=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
 for s in smoke:
  q=s['question_id'];x,l=prepare(processor,s);assert l==historical['input_bindings'][q]
  results={};vectors={}
  for c in protocol['conditions']:results[c],vectors[c]=call(x,l,c,audit=True,capture=True)
  noop,nv=call(x,l,'Native',audit=True,capture=True,noop=True)
  assert results['Native']['token_ids']==noop['token_ids'] and len(vectors['Native'])==len(nv) and all(torch.equal(a,b) for a,b in zip(vectors['Native'],nv)),'Native/noop logits differ'
  with torch.inference_mode():
   hf=model.generate(**{k:v.to('cuda:0') for k,v in x.items()},do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True,logits_to_keep=1)
  assert hf.sequences[0,l['input_length']:].tolist()==results['Native']['token_ids']
  assert len(hf.scores)==len(nv) and all(torch.equal(v,b[0].float().cpu()) for v,b in zip(nv,hf.scores)),'Manual Native differs from HF generate'
  del hf
  plain,pv=call(x,l,'Physical-Original',audit=False,capture=True)
  assert plain['token_ids']==results['Physical-Original']['token_ids'] and len(pv)==len(vectors['Physical-Original']) and all(torch.equal(a,b) for a,b in zip(pv,vectors['Physical-Original'])),'Audit hooks alter Physical math'
  for c in ['Dense-F','Physical-Original']:
   a=results[c]['audit'];n=results[c]['generated_tokens']
   assert sum(e.get('mlp_and_norm_executed',False) for e in a['attention_events'])==36*n
   assert len(a['probability'])>0 and all(e['forbidden_probability']==0 for e in a['probability'])
   assert all(e['logical_positions']==(list(range(l['input_length'])) if e['layer']<22 or c=='Dense-F' else [i for i in range(l['input_length']) if i not in set(l['image_positions'])]) for e in a['layers'] if e['step']==0)
  a=results['Physical-Original']['audit'];n=results['Physical-Original']['generated_tokens'];text=l['input_length']-l['image_tokens']
  assert a['deepstack']==[0,1,2]
  assert all(e['hidden_rows']==(l['input_length'] if e['layer']<22 else text) for e in a['layers'] if e['step']==0)
  for i in range(1,7):
   assert results['Physical-Original']['kv']['layers'][i]['live_bytes']==0
   shapes=[e for e in a['projection_shapes'] if e['layer']==i]
   assert not any(e['step']>0 and e['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV'] for e in shapes)
   assert all(e['input'][1]==l['input_length']-1 for e in shapes if e['step']==0 and e['op'] in ['q_proj','k_proj','v_proj','o_proj'])
  assert all(e['live_tokens']==text+n-1 for e in results['Physical-Original']['kv']['layers'] if e['live_tokens'])
  assert all(e['key'][-2]==text+e['step'] for e in a['projection_shapes'] if e['step']>0 and e['op'] in ['QK','AV'])
  differences=[];prefix=True
  for i,(f,c) in enumerate(zip(vectors['Dense-F'],vectors['Physical-Original'])):
   delta=(f-c).abs();differences.append(dict(step=i,max_abs=float(delta.max()),mean_abs=float(delta.mean()),top1_equal=int(f.argmax())==int(c.argmax()),identical_previous_tokens=prefix));prefix=prefix and results['Dense-F']['token_ids'][i]==results['Physical-Original']['token_ids'][i]
  same_tokens=results['Dense-F']['token_ids']==results['Physical-Original']['token_ids']
  strict=same_tokens and len(vectors['Dense-F'])==len(vectors['Physical-Original']) and all(d['max_abs']<=.5 and d['mean_abs']<=.02 and d['top1_equal'] for d in differences)
  item=dict(question_id=q,input_length=l['input_length'],image_tokens=l['image_tokens'],binding=hb(s,l),structural_status='PASS',native_noop_all_logits_strict=True,native_manual_HF_all_logits_strict=True,audit_on_off_exact=True,numerical_status='PASS' if strict else 'NUMERICAL_EQUIVALENCE_FAILED',complete_tokens_equal=same_tokens,logit_differences=differences,results=results)
  write(O/'smoke'/f'{q}.json',item);validation.append({k:v for k,v in item.items() if k!='results'})
  print('GPU_PHYSICAL_SMOKE_PASS',q,l['input_length'],item['numerical_status'],flush=True)
  del x,vectors,nv,pv,results;gc.collect()
 write(O/'validation.json',dict(structural_status='PASS',numerical_status='NUMERICAL_EQUIVALENCE_FAILED' if any(r['numerical_status']!='PASS' for r in validation) else 'PASS',exploratory_only=any(r['numerical_status']!='PASS' for r in validation),thresholds_unchanged=True,smoke=validation,CPU=read(O/'cpu_validation.json'),request_cache_isolation_and_hook_cleanup='PASS',binding=binding))
 # Fresh new-GPU predictions for all 1000 frozen questions, all three conditions.
 for j,s in enumerate(rows):
  q=s['question_id'];x,l=prepare(processor,s);assert l==historical['input_bindings'][q]
  b=hb(s,l)
  for c in protocol['conditions']:
   def run(c=c):
    result,_=call(x,l,c);pred=gqa_prediction(result['text']);result.update(prediction=pred,gold=s['gold'],score=float(pred==s['gold']),task=s['task'],types=s['types'],image_sha256=s['image_sha256']);return result
   checkpoint(O/'historical_predictions'/f'{q}__{c}.json',b,run)
  write(O/'progress.json',dict(status='RUNNING',phase='HISTORICAL_GQA1000_REPRODUCTION',completed_questions=j+1,total_questions=1000,new_GPU_predictions=3*(j+1),binding=binding,peak_allocated=peak,elapsed_seconds=time.monotonic()-started,at_UTC=now(),gpu=gpu() if j%50==0 else None))
  if j%10==0:print('HISTORICAL_QUALITY',j+1,'/1000',round(time.monotonic()-started,1),flush=True)
  del x;gc.collect()
 write(O/'historical_completion.json',dict(status='COMPLETE',questions=1000,generations=3000,binding=binding))
 # Complete available seven-benchmark protocols before efficiency, as authorized.
 from seven_stage import run_seven
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
