"""Native memory gate, then keep the same loaded model for explicitly dispatched next phases."""
import os,sys,json,time,gc,traceback,runpy,hashlib,collections
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path[:0]=[str(O),str(L),str(L/'sources/qwen')]
from run_common import sha,atomic,gpu_idle,score,RAW
from qwen_frozen_benchmark_inputs import prepare,digest
import torch,transformers,numpy as np
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from attention_backend import efficient_only,sdpa_compute,align_mask
from runtime import generate
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((L/'results/qwen/rows.json').read_text());by={r['question_id']:r for r in rows};model=None;current=None
assert len(rows)==1000 and len({r['image_sha256'] for r in rows})==930
for i,r in enumerate(rows):assert r['question_id']==f'{i:04d}' and r['question']==RAW[i]['question'] and r['answer']==RAW[i]['answers'] and r['image_id']==RAW[i]['image_path']
def status(s,**kw):atomic(O/'progress.json',dict(status=s,current=current,**kw))
try:
 assert json.loads((O/'cpu_tests.json').read_text())['status']=='PASS';gpu=gpu_idle()
 source={f:sha(O/f) for f in ['attention_backend.py','runtime.py','native_gate_run.py']};atomic(O/'new_native_source_binding.json',dict(protocol_sha256=pb,source_sha256=source,frozen_before_GPU=True))
 status('GPU_BACKEND_COMPATIBILITY',gpu=gpu)
 from types import SimpleNamespace
 from unittest.mock import patch
 from transformers.integrations import sdpa_attention as hf_sdpa
 probe=[]
 for kind,heads,kvheads,dim,causal,masked in [('Vision',16,16,72,False,False),('Language',32,8,128,True,False),('Custom_mask',32,8,128,False,True)]:
  q=torch.randn(1,heads,63,dim,device='cuda',dtype=torch.bfloat16);k=torch.randn(1,kvheads,63,dim,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);bias=torch.zeros(1,1,63,63,device='cuda',dtype=torch.bfloat16) if masked else None
  if bias is not None:bias[...,:3]=-torch.inf
  with torch.inference_mode(),efficient_only(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as pr:
   module=SimpleNamespace(num_key_value_groups=heads//kvheads,is_causal=causal)
   out,_=sdpa_compute(module,q,k,v,bias,dim**-.5)
   with patch.object(hf_sdpa,'use_gqa_in_sdpa',lambda mask,key:False):reference,_=hf_sdpa.sdpa_attention_forward(module,q,k,v,align_mask(bias),scaling=dim**-.5)
   assert torch.equal(out,reference),'ADAPTER_OFFICIAL_SDPA_NOOP_MISMATCH'
  assert bool(torch.isfinite(out).all())
  ops=collections.Counter(e.name for e in pr.events() if 'scaled_dot_product' in e.name);kernels=collections.Counter(e.name for e in pr.events() if e.device_type==torch.autograd.DeviceType.CUDA)
  assert ops.get('aten::_scaled_dot_product_efficient_attention',0)>0 and not any('math' in n or 'flash' in n or 'cudnn' in n for n in ops)
  probe.append(dict(kind=kind,query=list(q.shape),key=list(k.shape),finite=True,adapter_equals_official_hf_expanded_sdpa_torch_equal=True,attention_operators=dict(ops),CUDA_kernel_counts=dict(kernels)));del q,k,v,out,bias,reference
 atomic(O/'gpu_backend_probe.json',dict(status='PASS',cases=probe,math_disabled=True,flash_disabled=True,cudnn_disabled=True));torch.cuda.empty_cache()
 status('LOADING_MODEL');M='/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct';proc=AutoProcessor.from_pretrained(M,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation={'text_config':'sdpa','vision_config':'sdpa'},device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
 assert model.config.vision_config._attn_implementation==model.config.text_config._attn_implementation=='sdpa'
 atomic(O/'environment.json',dict(gpu=gpu,torch=torch.__version__,transformers=transformers.__version__,revision=P['revision'],model_loads=1,all_weights_GPU_BF16=True,backend='SDPA_EFFICIENT_ATTENTION_ONLY',input_processor=proc.image_processor.to_dict(),source_SHA256=source))
 gates=[]
 for q in P['smoke_ids']:
  current=dict(phase='Native_GATE',question_id=q);status('RUNNING_NATIVE_GATE');x,l=prepare(proc,by[q]);assert digest(l)==P['all_layout_hashes'][q]
  free,total=torch.cuda.mem_get_info();assert free>4*2**30,('NATIVE_MEMORY_PRECHECK',free)
  g,logits=generate(model,proc,x,l,'Native',profile=True,capture=True);rec=dict(question_id=q,image_sha256=by[q]['image_sha256'],request_sha256=by[q]['request_SHA256'],protocol_sha256=pb,input_layout_sha256=digest(l),input_tensor_hashes=l['input_tensor_hashes'],generation=g,**score(q,g['text']))
  atomic(O/'predictions/Native'/(q+'.json'),rec);(O/'gate_logits').mkdir(exist_ok=True);np.savez_compressed(O/'gate_logits'/('Native_'+q+'.npz'),logits=torch.stack(logits).numpy())
  gates.append(dict(question_id=q,input_length=l['input_length'],image_tokens=l['image_tokens'],success=True,text=g['text'],token_ids=g['token_ids'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],memory=g['memory'],finite=True))
  atomic(O/'native_memory_gate.json',dict(status='PARTIAL',completed=len(gates),expected=3,cases=gates,protocol_sha256=pb));print('NATIVE_MEMORY_GATE_CASE_PASS',q,'tokens',g['generated_tokens'],'peak_GiB',g['memory']['peak_allocated']/2**30,flush=True)
  del x,g,logits;gc.collect();torch.cuda.empty_cache()
 assert len(gates)==3 and gates[-1]['question_id']=='0738'
 atomic(O/'native_memory_gate.json',dict(status='PASS',complete_native_0738=True,completed=3,expected=3,cases=gates,protocol_sha256=pb,actual_backend='SDPA_EFFICIENT_ATTENTION_ONLY',no_math_fallback=True,unchanged_inputs=True))
 status('NATIVE_GATE_PASS_WAIT_NEXT_PHASE');print('NATIVE_0738_FULL_GENERATION_PASS_SAME_MODEL_WAITING',flush=True)
 done=set()
 while True:
  commands=sorted((O/'control').glob('command_*.json')) if (O/'control').exists() else []
  pending=[f for f in commands if f.name not in done and not f.name.endswith('_ack.json')]
  if not pending:time.sleep(5);continue
  f=pending[0];cmd=json.loads(f.read_text());done.add(f.name)
  if cmd['action']=='STOP':break
  assert cmd['action'] in ['SVH_GATES','FORMAL'];script=O/cmd['script'];assert sha(script)==cmd['sha256']
  current=dict(phase=cmd['action']);status('RUNNING_'+cmd['action']);runpy.run_path(str(script),init_globals=dict(MODEL=model,PROCESSOR=proc,PROTOCOL=P,PROTOCOL_SHA256=pb),run_name='__qwen_stage__')
  atomic(O/'control'/(f.stem+'_ack.json'),dict(status='COMPLETE',action=cmd['action']))
  if cmd['action']=='FORMAL':break
  status('SVH_GATE_READY_WAIT_FORMAL');print('SVH_STAGE_RETURNED_SAME_MODEL_WAITING',flush=True)
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',current=current,error_type=type(e).__name__,error=str(e),request_partial=getattr(e,'request_partial',None),traceback=traceback.format_exc()));status('BLOCKED',error=str(e));traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
