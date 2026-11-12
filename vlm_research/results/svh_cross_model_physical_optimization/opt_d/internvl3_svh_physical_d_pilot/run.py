"""Single-GPU gated pilot. No formal generation occurs unless all preregistered smoke gates pass."""
import gc,json,os,platform,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoModel,AutoTokenizer
from .common import *
from . import engine as p
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as b
from vlm_research.internvl3_svh_joint_3000 import engine as joint

@torch.no_grad()
def load():
 model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
  torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,
  device_map={'':'cuda:0'},use_safetensors=True).eval()
 assert len(model.language_model.model.layers)==28 and model.num_image_token==256
 assert model.language_model.config._attn_implementation=='eager'
 assert {x.dtype for x in model.parameters()}=={torch.bfloat16}
 assert {str(x.device) for x in model.parameters()}=={'cuda:0'}
 return model

def diff_record(sample,kind,step,layer,dense,physical,gate,same_prefix=True):
 x=dense.float();y=physical.float();diff=(x-y).abs()
 relative=float(diff.square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-6))
 row=dict(question_id=sample['question_id'],kind=kind,step=step,layer=layer,
  max_abs=float(diff.max()),mean_abs=float(diff.mean()),relative_RMS=relative,
  same_autoregressive_prefix=same_prefix,top1_equal=True)
 if kind=='logits':
  row['top1_equal']=int(x.argmax())==int(y.argmax())
  row['pass']=row['max_abs']<=gate['per_step_logits_max_abs_le'] and row['mean_abs']<=gate['per_step_logits_mean_abs_le'] and row['top1_equal'] and same_prefix
 else:row['pass']=relative<=gate['per_layer_shared_hidden_relative_RMS_le'] and same_prefix
 return row

@torch.inference_mode()
def smoke(model,tok,loader,manifest,protocol):
 oldm=read(OLD/'sample_manifest.json');olds=oldm['samples'] if isinstance(oldm,dict) else oldm
 old_idx=[525,1928,1127]
 # Three historical replay inputs + three new frozen inputs = six smoke samples.
 newids=manifest['smoke_question_ids'];chosen=[next(s for s in manifest['samples'] if s['question_id']==qid) for qid in [newids[0],newids[2],newids[-1]]]
 smoke_samples=[olds[i] for i in old_idx]+chosen
 rows=[];checks=[];layouts=[];audit=[];all_ok=True;multi=False
 gate=protocol['numerical_gate']['physical_vs_dense'];first_mismatch=None
 for index,s in enumerate(smoke_samples):
  prepared=b.prepare(model,tok,loader,s);ready=p.gpu_ready(model,tok,loader,s)
  if index>=3:
   assert prepared['prompt']==s['prompt'] and prepared['binding']['input_ids']==s['input_ids']
   assert prepared['roles']==s['roles']
  official,official_scores,_=b.generate(model,tok,prepared,native=True)
  native,native_scores,ncontrol=p.generate(model,tok,ready,'Native',audit=True,prepared=prepared)
  noop,noop_scores,noop_control=p.generate(model,tok,ready,'No-op',audit=True,prepared=prepared)
  strict=(native['token_ids']==noop['token_ids']==official['token_ids'] and
   len(native_scores)==len(noop_scores)==len(official_scores) and
   all(torch.equal(a,c) and torch.equal(a,d) for a,c,d in zip(native_scores,noop_scores,official_scores)))
  assert strict,'Official Native / No-op effective logits mismatch'
  if index==0:
   with b.OfficialChatCapture(model) as capture:
    answer=model.chat(tok,prepared['pixels'],prepared['request'],dict(do_sample=False,num_beams=1,
     max_new_tokens=32,logits_to_keep=1,pad_token_id=tok.convert_tokens_to_ids(model.conv_template.sep.strip())))
   assert capture.ids==native['token_ids'] and answer.strip()==native['text']
   assert all(torch.equal(x,y) for x,y in zip(native_scores,capture.scores))
  frozen_d,frozen_scores,_=joint.generate(model,tok,prepared,'D',audit=True,trace=True)
  dense,ds,dc=p.generate(model,tok,ready,'Dense-D',audit=True,prepared=prepared)
  assert dense['token_ids']==frozen_d['token_ids'] and len(ds)==len(frozen_scores)
  assert all(torch.equal(x,y) for x,y in zip(ds,frozen_scores)),'Clean Dense-D differs from frozen runtime'
  historical={}
  if index<3:
   cp=OLD/'checkpoints'/(s['task']+'__'+s['question_id']+'.json')
   record=read(cp);prior=record['conditions']['D']
   historical=dict(checkpoint_sha256=sha(cp),input_binding_equal=record['input_binding']==prepared['input_binding'],
    token_ids_equal=prior['token_ids']==dense['token_ids'],
    all_logits_SHA_equal=prior['all_step_logits_sha256']==[b.tensor_hash(x.float()) for x in ds])
   assert all(historical[x] for x in ['input_binding_equal','token_ids_equal','all_logits_SHA_equal']),historical
  physical,ps,pc=p.generate(model,tok,ready,'Physical-D',audit=True,prepared=prepared)
  assert len(physical['token_ids'])>=2 and not physical['empty']
  multi=multi or len(dense['token_ids'])>=3
  token_match=dense['token_ids']==physical['token_ids'];sample_ok=token_match and len(ds)==len(ps)
  for step,(x,y) in enumerate(zip(ds,ps)):
   same=dense['token_ids'][:step]==physical['token_ids'][:step]
   row=diff_record(s,'logits',step,-1,x,y,gate,same);rows.append(row);sample_ok &= row['pass']
  for (step,i),(ppos,ph) in pc.hidden.items():
   if (step,i) not in dc.hidden:sample_ok=False;continue
   dpos,dh=dc.hidden[(step,i)];lookup={int(z):j for j,z in enumerate(dpos)}
   selected=dh[:,[lookup[int(z)] for z in ppos],:]
   same=dense['token_ids'][:step]==physical['token_ids'][:step]
   row=diff_record(s,'hidden',step,i,selected,ph,gate,same);rows.append(row);sample_ok &= row['pass']
   if first_mismatch is None and not row['pass']:first_mismatch=row
  for result in [native,dense,physical]:
   for row in result['kv_layout']:layouts.append(dict(question_id=s['question_id'],**row))
   audit.append(dict(question_id=s['question_id'],**result['audit']))
  check=dict(question_id=s['question_id'],historical=index<3,input_binding=prepared['input_binding'],
   native_noop_official_strict_equal=strict,clean_dense_vs_frozen_strict_equal=True,historical_replay=historical,
   complete_tokens_equal=token_match,native_tokens=native['token_ids'],dense_tokens=dense['token_ids'],
   physical_tokens=physical['token_ids'],dense_text=dense['text'],physical_text=physical['text'],
   numerical_pass=bool(sample_ok),max_logits_abs=max(r['max_abs'] for r in rows if r['question_id']==s['question_id'] and r['kind']=='logits'),
   prefill_rows=ready['roles']['n'])
  checks.append(check);all_ok &= sample_ok
  save(RUN/'smoke'/(str(index)+'_'+s['question_id']+'.json'),check)
  save(RUN/'numerical_progress.json',dict(samples=checks,numerical_rows=len(rows),all_pass_so_far=bool(all_ok)))
  print('SMOKE',index,s['question_id'],'PASS' if sample_ok else 'FAILED','logit_max',check['max_logits_abs'],flush=True)
  del prepared,ready,official_scores,native_scores,noop_scores,frozen_scores,ds,ps,ncontrol,noop_control,dc,pc
  gc.collect()
 assert multi,'Smoke does not include a multi-token answer'
 # Independent forced 32-token path/call/KV audit outside any efficiency timing.
 # Only execute this extension if all core six-sample numerical gates pass.
 if all_ok:
  s=chosen[1];prepared=b.prepare(model,tok,loader,s);ready=p.gpu_ready(model,tok,loader,s)
  for method in METHODS:
   r,_,control=p.generate(model,tok,ready,method,fixed=True,audit=True,prepared=prepared)
   assert r['generated_tokens']==32 and r['audit']['real_decode']==31
   layouts.extend(dict(question_id=s['question_id'],audit_fixed32=True,**x) for x in r['kv_layout'])
   audit.append(dict(question_id=s['question_id'],audit_fixed32=True,**r['audit']))
   save(RUN/'smoke'/('fixed32_'+method+'.json'),r)
  # Check clean timed vision decomposition has identical native tokens to official extract_feature.
  a,_,_=p.generate(model,tok,ready,'Native',fixed=True,prepared=prepared)
  c,_,_=p.generate(model,tok,ready,'Native',fixed=True,timed=True)
  assert a['token_ids']==c['token_ids'],'Vision decomposition changed native output'
 fields=['question_id','kind','step','layer','max_abs','mean_abs','relative_RMS','same_autoregressive_prefix','top1_equal','pass']
 csvwrite(RUN/'numerical_comparison.csv',rows,fields)
 csvwrite(RUN/'kv_layout.csv',layouts,['question_id','audit_fixed32','method','layer','logical_length','physical_length','visual_kv','k_shape','v_shape','dtype','live_bytes','capacity_bytes','skip_attention'])
 save(RUN/'physical_execution_audit.json',dict(samples=audit,actual_calls_verified_outside_timing=True,hidden_exit_rows=256,skip_layers=[2,3,4,5],no_dense_backing_storage=True))
 summary=dict(status='PASS' if all_ok else 'FAILED',passed=bool(all_ok),samples=checks,
  multi_token_answer_covered=multi,first_failed_hidden_comparison=first_mismatch,numerical_gate=protocol['numerical_gate'],
  protocol_sha256=sha(RUN/'protocol.json'),formal_quality='READY' if all_ok else 'NOT_RUN',
  formal_efficiency='READY' if all_ok else 'NOT_RUN')
 save(RUN/'smoke_validation.json',summary)
 return all_ok

@torch.inference_mode()
def quality(model,tok,loader,manifest,ph,sh):
 total=0
 for index,s in enumerate(manifest['samples']):
  dest=RUN/'quality_checkpoints'/(s['task']+'__'+s['question_id']+'.json')
  record=read(dest) if dest.exists() else dict(question_id=s['question_id'],task=s['task'],gold=s['answer'],
   image_sha256=s['image_sha256'],protocol_sha256=ph,source_manifest_sha256=sh,sample_sha256=digest(s),conditions={})
  assert record['protocol_sha256']==ph and record['source_manifest_sha256']==sh and record['sample_sha256']==digest(s)
  if len(record['conditions'])<3:
   assert sha(s['actual_image_path'])==s['image_sha256']
   prepared=b.prepare(model,tok,loader,s);ready=p.gpu_ready(model,tok,loader,s)
   assert prepared['prompt']==s['prompt'] and prepared['binding']['input_ids']==s['input_ids'] and prepared['roles']==s['roles']
   for method in METHODS:
    if method in record['conditions']:continue
    row,_,_=p.generate(model,tok,ready,method,prepared=prepared)
    row['correct']=score(row['text'],s['answer']);row['input_binding']=prepared['input_binding']
    record['conditions'][method]=row;record['complete']=len(record['conditions'])==3
    save(dest,record)
   del prepared,ready
  total+=len(record['conditions'])
  save(RUN/'progress.json',dict(stage='quality',samples=index+1,generations=total,target=360))
  if (index+1)%10==0:print('QUALITY',index+1,'/120',flush=True)
 save(RUN/'quality_complete.json',dict(samples=120,generations=total))

@torch.inference_mode()
def efficiency(model,tok,loader,manifest,ph,sh):
 samples=[next(s for s in manifest['samples'] if s['question_id']==q) for q in manifest['efficiency_question_ids']]
 ready=p.gpu_ready(model,tok,loader,samples[0])
 for method in METHODS:
  for _ in range(2):p.generate(model,tok,ready,method,fixed=True)
 print('EFFICIENCY_WARMUP_COMPLETE',flush=True)
 total=0
 for index,s in enumerate(samples):
  ready=p.gpu_ready(model,tok,loader,s)
  binding=dict(prompt=ready['prompt'],input_ids=ready['ids'][0].cpu().tolist(),
   pixels_sha256=b.tensor_hash(ready['pixels']),image_sha256=s['image_sha256'])
  assert binding['prompt']==s['prompt'] and binding['input_ids']==s['input_ids']
  tensor_binding=digest(binding)
  for repeat in range(5):
   order=METHODS[(index+repeat)%3:]+METHODS[:(index+repeat)%3]
   for method in order:
    path=RUN/'efficiency_checkpoints'/(s['question_id']+'__'+str(repeat)+'__'+method+'.json')
    if path.exists():
     r=read(path)
     assert r['protocol_sha256']==ph and r['source_manifest_sha256']==sh and r['input_tensor_binding']==tensor_binding
    else:
     assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==GPU
     row,_,_=p.generate(model,tok,ready,method,fixed=True,timed=True)
     assert row['generated_tokens']==32
     r=dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],repeat=repeat,
      order=order.index(method),protocol_sha256=ph,source_manifest_sha256=sh,input_tensor_binding=tensor_binding,
      input_binding= binding,**row)
     save(path,r)
    total+=1
  save(RUN/'progress.json',dict(stage='efficiency',inputs=index+1,formal_requests=total,target=450))
  print('EFFICIENCY',index+1,'/30',total,'/450',flush=True)
 save(RUN/'efficiency_complete.json',dict(inputs=30,repeats=5,formal_requests=total))
def main():
 model=None;monitor=None
 protocol=read(RUN/'protocol.json');manifest=read(RUN/'sample_manifest.json');ph=sha(RUN/'protocol.json')
 assert sha(RUN/'sample_manifest.json')==protocol['sample_manifest_sha256']
 assert read(RUN/'cpu_tests.json')['status']=='PASS'
 assert torch.__version__==protocol['torch'] and transformers.__version__==protocol['transformers']
 assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==GPU
 assert not command('nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader')
 sources=read(RUN/'source_manifest.json')
 for name,h in sources.items():assert sha(name)==h,name
 sh=sha(RUN/'source_manifest.json')
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 save(RUN/'active.json',dict(pid=os.getpid(),stage='GPU_SMOKE',protocol_sha256=ph))
 env=dict(python=platform.python_version(),python_executable=sys.executable,torch=torch.__version__,
  torch_import=torch.__file__,transformers=transformers.__version__,transformers_import=transformers.__file__,
  CUDA=torch.version.cuda,GPU_UUID=GPU,GPU=command('nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader'),
  mounts=command('findmnt','-T',str(RUN)),persistent_mount=command('findmnt','-T',str(OUT)),
  model_revision=REV,source_manifest_sha256=sh,source_commit=command('git','rev-parse','HEAD'),
  no_system_or_power_changes=True)
 save(RUN/'environment.json',env)
 try:
  log=(RUN/'gpu_telemetry.csv').open('w')
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv','-l','2'],stdout=log,stderr=subprocess.STDOUT)
  model=load();tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,trust_remote_code=True,local_files_only=True)
  loader=b.official_preprocessor(MODEL);versions=[(x._version,x.data_ptr()) for x in model.parameters()]
  if not smoke(model,tok,loader,manifest,protocol):
   save(RUN/'execution_status.json',dict(status='FAILED',reason='Predeclared numerical gate failed',quality='NOT_RUN',efficiency='NOT_RUN'));return
  quality(model,tok,loader,manifest,ph,sh)
  efficiency(model,tok,loader,manifest,ph,sh)
  assert versions==[(x._version,x.data_ptr()) for x in model.parameters()]
  save(RUN/'execution_status.json',dict(status='PASS',quality_requests=360,efficiency_requests=450))
 except BaseException as exc:
  save(RUN/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),protocol_sha256=ph))
  save(RUN/'execution_status.json',dict(status='FAILED',reason=repr(exc),quality='PARTIAL' if (RUN/'quality_checkpoints').exists() else 'NOT_RUN',efficiency='PARTIAL' if (RUN/'efficiency_checkpoints').exists() else 'NOT_RUN'))
  raise
 finally:
  model=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  if monitor is not None:monitor.terminate();monitor.wait(timeout=10);log.close()
  save(RUN/'gpu_release.json',dict(worker_pid=os.getpid(),model_deleted=True,allocated_MiB=torch.cuda.memory_allocated()/1048576 if torch.cuda.is_initialized() else 0,context_released_on_process_exit=True))
if __name__=='__main__':main()
