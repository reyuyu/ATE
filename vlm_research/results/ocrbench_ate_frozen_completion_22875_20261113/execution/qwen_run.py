import sys,json,gzip,os,time,gc,traceback
from run_common import *
sys.path[:0]=[str(L/'sources/qwen'),str(L)]
import torch,transformers
from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
from qwen_frozen_benchmark_inputs import prepare,digest as frozen_digest
from qwen_frozen_runtime import generate
O=L/'results/qwen';P=json.loads((O/'protocol.json').read_text());binding=sha(O/'protocol.json');rows=json.loads((O/'rows.json').read_text());by={r['question_id']:r for r in rows};reuse=json.loads((O/'verified_reuse.json').read_text());current=None;model=None;start=time.time()
records={m:load_records(O/'predictions'/m) for m in P['conditions']}
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def progress(status,**kw):atomic(O/'progress.json',dict(status=status,complete={m:len(records[m])+len(reuse[m]) for m in records},expected_each=1000,elapsed_s=time.time()-start,**kw))
try:
 assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
 for p,h in P['source_SHA256'].items():assert sha(L/'sources/qwen'/p)==h,p
 gpu=gpu_idle();progress('LOADING_MODEL',gpu=gpu)
 M='/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct';proc=AutoProcessor.from_pretrained(M,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
 assert model.config.text_config._attn_implementation==model.config.vision_config._attn_implementation=='eager'
 atomic(O/'environment.json',dict(gpu=gpu,torch=torch.__version__,transformers=transformers.__version__,model_loads=1,all_weights_GPU_bf16=True,eager=True,image_processor=proc.image_processor.to_dict()))
 def infer(q,m,audit,x=None,l=None):
  global current
  current=dict(question_id=q,method=m)
  if q in records[m]:return
  assert q not in reuse[m]
  if x is None:x,l=prepare(proc,by[q])
  assert frozen_digest(l)==P['all_layout_hashes'][q]
  progress('SMOKE' if audit else 'RUNNING',current=current,input_length=l['input_length'],visual_tokens=l['image_tokens'])
  g,_=generate(model,proc,x,l,m,audit=audit,capture=False,max_tokens=128)
  assert g['finite_checked'] and not g['fixed_length'] and g['max_new_tokens']==128
  if audit:
   a=g['audit'];assert not a['physical_fallback']
   if m=='Physical-Original':
    assert a['condition']=='C' and a['deepstack']==[0,1,2]
    n=l['input_length'];text=set(range(n))-set(l['image_positions'])
    for v in a['layers']:
     positions=(set(range(n)) if v['layer']<22 else text) if v['step']==0 else {n+v['step']-1}
     assert set(v['logical_positions'])==positions and v['hidden_rows']==len(positions)
    for v in a['probability']:assert v['forbidden_probability']==0
    for layer in range(1,7):
     assert any(v['layer']==layer and v['step']==0 and v.get('computed_rows')==n-1 for v in a['attention_events'])
     assert all(v['residual_exact'] and v['mlp_and_norm_executed'] for v in a['attention_events'] if v['layer']==layer and 'residual_exact' in v)
    for v in a['kv']['layers']:assert v['live_tokens']==(0 if v['layer'] in range(1,7) else len(text)+g['generated_tokens']-1)
   else:
    assert a['condition']=='N' and all(v['live_tokens']==l['input_length']+g['generated_tokens']-1 for v in a['kv']['layers'])
  r=dict(model='Qwen3-VL-8B',method=m,question_id=q,image_sha256=by[q]['image_sha256'],input_layout=l,protocol_sha256=binding,generation=g,**score(q,g['text']),origin='NEW_GPU_MISSING_ONLY')
  atomic(O/'predictions'/m/(q+'.json'),r);records[m][q]=r
  if audit or len(records[m])%25==0:print('QWEN_PROGRESS',m,len(records[m])+len(reuse[m]),'/1000',q,flush=True)
 for q in P['smoke_ids']:
  x,l=prepare(proc,by[q])
  for m in P['conditions']:
   if q not in reuse[m]:infer(q,m,True,x,l)
  del x
 atomic(O/'smoke.json',dict(status='PASS_STRUCTURAL_INHERITED_NUMERICAL_STATUS',ids=P['smoke_ids'],conditions=P['conditions'],same_input_tensors=True,independent_KV_per_request=True,reused_as_formal=True,frozen_Original_numerical_status='Historical numerical gate and exploratory label retained, not relaxed or redefined'))
 for s in rows:
  q=s['question_id'];x,l=prepare(proc,s)
  for m in P['conditions']:
   if q not in reuse[m]:infer(q,m,False,x,l)
  del x
 assert all(len(records[m])+len(reuse[m])==1000 for m in records);progress('COMPLETE_PENDING_INDEPENDENT_AUDIT')
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',current=current,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));progress('BLOCKED',error=str(e),failed=current);traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('QWEN_GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
