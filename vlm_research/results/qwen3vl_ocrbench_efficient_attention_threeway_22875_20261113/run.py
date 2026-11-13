"""Single-load SVH-R-only OCRBench1000. No completed Native/SVH rerun."""
import sys,json,gc,time,datetime,traceback,shutil,gzip,os,collections
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-svhr-efficient-attention-22875-20261113');B=O.parent/'qwen3vl-ocrbench-memory-efficient-attention-22875-20261113';L=O.parent/'ocrbench-ate-frozen-completion-22875-20261113';T=Path('/home/featurize/work/ATE/vlm_research/results/qwen3vl_ocrbench_svhr_efficient_attention')
sys.path[:0]=[str(O),str(O/'sources'),str(L)]
import torch,transformers,numpy as np
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from run_common import atomic,sha,score,RAW,gpu_idle
from qwen_frozen_benchmark_inputs import prepare,digest
from svhr_adapter import EfficientSVHR
from runtime import generate
from vlm_research.qwen3vl_svhr_search.engine import Functional
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((L/'results/qwen/rows.json').read_text());by={r['question_id']:r for r in rows};model=None;current=None;start=time.time();records={};pending=[];new_count=0;uuids=set()
assert len(rows)==1000 and transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
for path,h in json.loads((O/'protected_sources.json').read_text()).items():assert sha(path)==h
for f,h in P['new_source_sha256'].items():assert sha(O/f)==h
assert json.loads((O/'cpu_tests.json').read_text())['status']=='PASS'
for i,s in enumerate(rows):assert s['question_id']==f'{i:04d}' and s['question']==RAW[i]['question'] and s['answer']==RAW[i]['answers']

def status(state,**kw):atomic(O/'progress.json',dict(status=state,current=current,completed=len(records),expected=1000,new_generated=new_count,elapsed_seconds=time.time()-start,**kw))
def backup():
 for p in pending:
  dest=T/'checkpoint'/p.relative_to(O);dest=dest.with_suffix(dest.suffix+'.gz');dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp')
  with p.open('rb') as src,gzip.open(tmp,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
  os.replace(tmp,dest)
 pending.clear()
 for f in ['protocol.json','progress.json','run_metadata.json','gate.json','failure.json','cpu_tests.json','environment.json','svhr_adapter.py','runtime.py','attention_backend.py','run.py']:
  if (O/f).exists():shutil.copyfile(O/f,T/f)
def persist(s,l,g,origin):
 global new_count
 q=s['question_id'];assert q not in records and g['request_cache_uuid'] not in uuids
 assert g['condition']=='SVH-R' and g['finite_logits'] and g['independent_cache'] and g['natural_eos']
 a=g['physical_audit'];assert a['read_layers']==list(range(19,36)) and not a['physical_fallback']
 r=dict(question_id=q,image_sha256=s['image_sha256'],request_sha256=s['request_SHA256'],protocol_sha256=pb,base_protocol_sha256=P['base_protocol_sha256'],input_layout_sha256=digest(l),input_tensor_hashes=l['input_tensor_hashes'],source_sha256=P['new_source_sha256'],origin=origin,generation=g,**score(q,g['text']))
 p=O/'predictions'/(q+'.json');atomic(p,r);records[q]=r;uuids.add(g['request_cache_uuid']);pending.append(p);new_count+=1
 print('SVHR_DONE',q,'n',l['input_length'],'image',l['image_tokens'],'tokens',g['generated_tokens'],'EOS',g['eos'],'score',r['official_score'],'coverage',len(records),'peak_GiB',round(g['memory']['peak_allocated']/2**30,3),'text',repr(g['text'])[:120],flush=True)

try:
 gpu=gpu_idle();status('LOADING_SVHR_ONLY',gpu=gpu)
 M='/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct';proc=AutoProcessor.from_pretrained(M,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation={'text_config':'sdpa','vision_config':'sdpa'},device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
 versions=[(p.data_ptr(),p._version) for p in model.parameters()]
 atomic(O/'environment.json',dict(gpu=gpu,torch=torch.__version__,transformers=transformers.__version__,model=P['model'],revision=P['revision'],dtype='BF16',backend='SDPA_EFFICIENT_ATTENTION_ONLY',model_loads=1,base_model_reused_from_local=True,processor=proc.image_processor.to_dict()))
 metadata=dict(status='RUNNING',protocol_sha256=pb,condition='SVH-R',expected=1000,read_layers=list(range(19,36)),Native_and_Original_reused_without_generation=True,new_stage_model_loads=1,previous_base_model_loads=1,user_authorized_additional_condition=True,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=P['new_source_sha256'])
 atomic(O/'run_metadata.json',metadata)
 for f in sorted((O/'predictions').glob('*.json')):
  r=json.loads(f.read_text());q=r['question_id'];assert r['protocol_sha256']==pb and r['input_layout_sha256']==P['all_layout_hashes'][q];assert score(q,r['generation']['text'])=={k:r[k] for k in ['official_score','independent_score']}
  records[q]=r;assert r['generation']['request_cache_uuid'] not in uuids;uuids.add(r['generation']['request_cache_uuid'])
 if not (O/'gate.json').exists():
  gates=[];short_vectors=None;short_tokens=None
  for q in P['smoke_ids']:
   current=dict(phase='SVHR_PHYSICAL_GATE',question_id=q);status('SMOKE_RUNNING');s=by[q];x,l=prepare(proc,s);assert digest(l)==P['all_layout_hashes'][q]
   execution=EfficientSVHR(model,l,max_tokens=128,audit=True)
   g,vectors=generate(model,proc,x,l,'SVH-R',profile=True,capture=q==P['smoke_ids'][0],execution=execution)
   assert versions==[(p.data_ptr(),p._version) for p in model.parameters()]
   a=g['physical_audit'];assert len(a['layers'])==36*g['generated_tokens']
   for r in a['layers']:
    expected=l['input_length']-(l['image_tokens'] if r['layer']>=22 else 0) if r['step']==0 else 1
    assert r['hidden_rows']==expected
   assert all(k['live_tokens']==0 for k in g['kv']['layers'] if k['layer'] in range(1,7))
   assert a['kv']['persistent_image_bytes']==17*l['image_tokens']*4096
   gates.append(dict(question_id=q,input_length=l['input_length'],image_tokens=l['image_tokens'],text=g['text'],token_ids=g['token_ids'],score=score(q,g['text']),memory=g['memory'],builder_layers=a['builder_layers'],all_structural_checks=True,read_layers=a['read_layers'],persistent_visual_KV_bytes=a['kv']['persistent_image_bytes']))
   if q==P['smoke_ids'][0]:short_vectors=vectors;short_tokens=g['token_ids'];(O/'gate_logits').mkdir(exist_ok=True);np.savez_compressed(O/'gate_logits'/('SVHR_'+q+'.npz'),logits=torch.stack(vectors).numpy())
   persist(s,l,g,'validated_physical_smoke_reused');status('SMOKE_RUNNING');backup()
   del execution,g,vectors,x,l;gc.collect();torch.cuda.empty_cache()
  q=P['smoke_ids'][0];current=dict(phase='SHORT_DENSE_R_NUMERICAL_REFERENCE',question_id=q);status('SMOKE_NUMERICAL_REFERENCE')
  x,l=prepare(proc,by[q]);reference=Functional(model,l,'F',128,audit=False,layers=tuple(range(19,36)));reference.check_attention=lambda *args:None
  rg,rv=generate(model,proc,x,l,'Dense-R',profile=False,capture=True,execution=reference,forced=short_tokens)
  comparisons=[]
  for j,(a,b) in enumerate(zip(short_vectors,rv)):
   d=(a-b).abs();comparisons.append(dict(step=j,max_abs=float(d.max()),mean_abs=float(d.mean()),RMSE=float(((a-b)**2).mean().sqrt()),physical_argmax=int(a.argmax()),dense_argmax=int(b.argmax()),generated=short_tokens[j]))
  maxabs=max(r['max_abs'] for r in comparisons);maxmean=max(r['mean_abs'] for r in comparisons);strict=maxabs<=.5 and maxmean<=.02;greedy=all(r['physical_argmax']==r['dense_argmax']==r['generated'] for r in comparisons)
  atomic(O/'numerical_comparison.json',dict(condition='SVH-R vs frozen Functional-R',question_id=q,reference_teacher_forced_same_prefix=True,strict_thresholds=dict(max_abs=.5,mean_abs=.02),strict_gate='PASS' if strict else 'FAIL',all_short_greedy_tokens_identical=greedy,steps=comparisons))
  assert greedy,'PHYSICAL_FUNCTIONAL_R_SHORT_GREEDY_MISMATCH_STOP'
  decision='PASS' if strict else 'PASS_STRUCTURAL_AND_GREEDY_WITH_NUMERICAL_WARNING'
  atomic(O/'gate.json',dict(status=decision,formal_allowed=True,structural_PASS=True,finite_PASS=True,actual_efficient_cuda_kernels=True,strict_numerical_gate='PASS' if strict else 'FAIL',strict_thresholds_unchanged=True,short_greedy_identical=True,cases=gates,smoke_cases=3,numerical_reference_cases=1,Native_or_Original_new_generation=0,physical_implementation=True,frozen_physical_forward_unmodified=True,read_layers=list(range(19,36))))
  del x,l,reference,rg,rv,short_vectors;gc.collect();torch.cuda.empty_cache();backup();print('SVHR_PHYSICAL_GATE',decision,'maxabs',maxabs,'maxmean',maxmean,flush=True)
 assert json.loads((O/'gate.json').read_text())['formal_allowed']
 for s in rows:
  q=s['question_id']
  if q in records:continue
  current=dict(phase='SVHR1000',question_id=q);status('FORMAL_RUNNING');x,l=prepare(proc,s);assert digest(l)==P['all_layout_hashes'][q]
  assert torch.cuda.mem_get_info()[0]>4*2**30,'GPU_RESOURCE_CHANGED'
  execution=EfficientSVHR(model,l,max_tokens=128,audit=False);g,_=generate(model,proc,x,l,'SVH-R',execution=execution)
  assert versions==[(p.data_ptr(),p._version) for p in model.parameters()];persist(s,l,g,'SVHR1000_new_only')
  del execution,g,x,l;gc.collect();torch.cuda.empty_cache();status('FORMAL_RUNNING')
  if new_count%25==0:backup()
 assert len(records)==1000
 metadata.update(status='GENERATION_COMPLETE_AWAIT_CPU_AUDIT',completed=1000,new_generated=new_count,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(O/'run_metadata.json',metadata);status('GENERATION_COMPLETE_AWAIT_CPU_AUDIT');backup();print('SVHR1000_GENERATION_COMPLETE',flush=True)
except BaseException as error:
 atomic(O/'failure.json',dict(status='BLOCKED',current=current,error_type=type(error).__name__,error=str(error),request_partial=getattr(error,'request_partial',None),traceback=traceback.format_exc()));status('BLOCKED',error=str(error));backup();traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('SVHR_GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
