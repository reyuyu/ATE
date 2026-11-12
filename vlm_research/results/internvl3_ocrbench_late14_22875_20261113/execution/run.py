"""One additional fixed Late14 window; frozen model/input/scorer/core; resume each (method,id)."""
import os,sys,json,time,gc,traceback
from pathlib import Path
OLD=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-late14-22875-20261113')
sys.path.insert(0,str(OLD));from run_common import sha,atomic,score,gpu_idle
sys.path.insert(0,str(OLD/'late9_sources'))
import torch
from vlm_research.internvl3_svh_seven_benchmarks import worker as original
from vlm_research.internvl3_svh_r_terminal_windows_textvqa import engine as late
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((O/'selected_rows.json').read_text());byid={r['question_id']:r for r in rows}
paths=json.loads((O/'input_paths.json').read_text());bindings={r['question_id']:r for r in json.loads((O/'expected_bindings.json').read_text())}
records={};start=time.time();model=None;current=None;checks={}
for method in P['configs']:
 folder=O/'predictions'/method;folder.mkdir(parents=True,exist_ok=True)
 records[method]={p.stem:json.loads(p.read_text()) for p in folder.glob('*.json')}
 for q,x in records[method].items():assert q in byid and x['protocol_sha256']==pb and x['method']==method
def progress(status,**kw):
 atomic(O/'progress.json',dict(status=status,complete={k:len(v) for k,v in records.items()},expected_per_method=100,elapsed_s=time.time()-start,**kw))
def check_audit(result,r,K):
 assert result['independent_cache'] and result['reaccess_layers']==K and result['cutoff']==28
 assert [x['layer'] for x in result['kv_layout'] if x['visual_kv']]==K
 a=result['audit'];assert a['hidden_exit']==[dict(before=r['input_tokens'],after=r['input_tokens']-256,removed=256)]
 assert a['exit_stage']=='after_L27_before_final_norm' and a['R_image_permission_boundary']==20
 assert a['allowed_image_probability_samples'] and all(x[2]>0 for x in a['allowed_image_probability_samples'])
 assert all(x[2]==0 for x in a['blocked_image_probability_samples']+a['R_image_probability_samples'])
 assert a['finite'] and a['original_logical_positions'] and a['non_target_queries_preserved']
 assert a['prefill_visual_rows_removed_from_language_blocks']==0
def numerical(ps,pc,ds,dc,steps,physical_ids):
 metrics=[];threshold=P['numerical_thresholds'];all_hidden=[]
 for step in range(steps):
  x,y=ps[step].float(),ds[step].float();d=(x-y).abs()
  metrics.append(dict(step=step,max_abs=float(d.max()),mean_abs=float(d.mean()),exact=torch.equal(x,y),top1_equal=int(x.argmax())==int(y.argmax()),physical_top1=int(x.argmax()),dense_top1=int(y.argmax())))
  assert int(x.argmax())==physical_ids[step]
  for layer in range(28):
   xp,xh=pc.hidden[step,layer];yp,yh=dc.hidden[step,layer]
   assert torch.equal(xp,yp), 'SHORT_REFERENCE_POSITION_MISMATCH'
   xh=xh.float();yh=yh.float();rel=float((xh-yh).square().mean().sqrt()/yh.square().mean().sqrt().clamp_min(1e-12))
   assert torch.isfinite(torch.tensor(rel));all_hidden.append(rel)
 passed=all(x['max_abs']<=threshold['logit_max_abs'] and x['mean_abs']<=threshold['logit_mean_abs'] and x['top1_equal'] for x in metrics) and max(all_hidden)<=threshold['shared_hidden_relative_RMS']
 return dict(status='PASS_SHORT_REFERENCE' if passed else 'FAIL_STRICT_RETAINED_EXPLORATORY_NUMERICAL_WAIVER',steps=steps,logits=metrics,max_shared_hidden_relative_RMS=max(all_hidden),thresholds_unchanged=threshold,scope='First smoke question only, up to8 same-prefix steps; no global numerical equivalence claim')
try:
 for path,h in P['source_SHA256'].items():assert sha(OLD/'late9_sources'/path)==h,path
 assert sha(O/'selected_rows.json')==P['selected_rows_sha256'] and sha(O/'baselines.json')==P['baselines_sha256']
 progress('LOADING_MODEL',gpu=gpu_idle());original.MODEL=Path('/home/featurize/ate-local-models/InternVL3-8B');original.GPU=os.environ['CUDA_VISIBLE_DEVICES']
 model,tok=original.load_model();inputs=original.Inputs(model,tok)
 atomic(O/'environment.json',dict(original.environment(model,tok),GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'],model_loads=1,runner_sha256=sha(O/'run.py')))
 def infer(q,method,audit):
  global current
  current=dict(question_id=q,method=method)
  if q in records[method]:return
  r=byid[q];ready,ib,b=inputs.ready(dict(r,image_path=paths[r['image_sha256']]))
  v=bindings[q]
  for key in ('pixels_bf16_sha256','input_ids_sha256','prompt_sha256','image_sha256'):assert b[key]==v[key],(q,key)
  K=P['configs'][method];result,scores,control=late.generate(model,tok,ready,method,K,max_tokens=128,audit=audit,physical=True,cutoff=28)
  assert result['independent_cache'] and result['reaccess_layers']==K and result['cutoff']==28
  assert [x['layer'] for x in result['kv_layout'] if x['visual_kv']]==K
  if audit:check_audit(result,r,K)
  rec=dict(method=method,question_id=q,category=r['category'],dataset_name=r['dataset_name'],image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding=ib,input_binding_details=b,protocol_sha256=pb,generation=result,**score(q,result['text']))
  atomic(O/'predictions'/method/(q+'.json'),rec);records[method][q]=rec
  if audit and q==P['smoke_ids'][0]:
   steps=min(P['numerical_reference_max_steps'],len(result['token_ids']))
   dense,ds,dc=late.generate(model,tok,ready,method,K,max_tokens=steps,fixed=True,forced=result['token_ids'][:steps],audit=True,physical=False,cutoff=28)
   n=numerical(scores,control,ds,dc,steps,result['token_ids']);atomic(O/'numerical'/f'{method}.json',n);del ds,dc,dense
  if audit:checks.setdefault(method,[]).append(q)
  del scores,control,ready;progress('SMOKE' if audit else 'RUNNING',last=current)
  print('PROGRESS',method,len(records[method]),'/100',q,flush=True)
 for method in P['configs']:
  for q in P['smoke_ids']:infer(q,method,True)
  assert all(q in records[method] for q in P['smoke_ids'])
 atomic(O/'smoke.json',dict(status='PASS_STRUCTURAL',ids=P['smoke_ids'],configs=list(P['configs']),quality_generations=4,reused_as_quality=True,checks=checks,numerical_status='See numerical/*.json; historical waiver/failures remain unchanged',no_baseline_regeneration=True))
 for r in rows:
  for method in P['configs']:infer(r['question_id'],method,False)
 assert all(len(v)==100 for v in records.values());progress('COMPLETE_PENDING_CPU_AUDIT');print('LATE14_ALL100_COMPLETE',flush=True)
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',current=current,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));progress('BLOCKED',error=str(e),current=current);traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
