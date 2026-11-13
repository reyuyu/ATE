import sys,json,hashlib,shutil,datetime
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-svhr-efficient-attention-22875-20261113');B=O.parent/'qwen3vl-ocrbench-memory-efficient-attention-22875-20261113';L=O.parent/'ocrbench-ate-frozen-completion-22875-20261113'
sys.path[:0]=[str(O),str(O/'sources'),str(L)]
import torch
from run_common import atomic,sha
from svhr_adapter import EfficientSVHR,Physical,READ
from vlm_research.qwen3vl_svhr_search.physical import SelectedKV
assert not torch.cuda.is_initialized()
assert EfficientSVHR.forward is Physical.forward and EfficientSVHR.context is Physical.context
checks=[];n=71;images=list(range(7,39));text=torch.tensor([i for i in range(n) if i not in images]);v=32
late={i:(torch.randn(1,8,v,128),torch.randn(1,8,v,128)) for i in range(22,36)}
cache=SelectedKV(n,text,127,READ,late)
for i in range(36):
 if i in range(1,7):continue
 k=torch.randn(1,8,n if i<22 else len(text),128);value=torch.randn_like(k)
 before=late.get(i);kk,vv=cache.update(k,value,i)
 if i>=22 and i in READ:
  assert torch.equal(kk[:,:,text],k) and torch.equal(vv[:,:,text],value)
  assert torch.equal(kk[:,:,images],before[0]) and torch.equal(vv[:,:,images],before[1])
 expected=n if i in READ else len(text);assert cache.used[i]==expected
 for p,t in zip(cache.entries[i],(kk,vv)):
  wanted=t if i in READ or i>=22 else t.index_select(-2,text)
  assert torch.equal(p[...,:expected,:],wanted)
 checks.append(dict(layer=i,prefill_KV_exact=True,visual_tokens=v if i in READ else 0,capacity=cache.capacities[i]))
assert not late
cache.prefill=False
for step in range(1,4):
 for i in range(36):
  if i in range(1,7):continue
  k=torch.randn(1,8,1,128);value=torch.randn_like(k);kk,vv=cache.update(k,value,i)
  assert torch.equal(kk[:,:,-1:],k) and torch.equal(vv[:,:,-1:],value)
kv=cache.audit();assert kv['persistent_image_bytes']==17*v*2*8*128*2
base=json.loads((B/'protocol.json').read_text());rows=json.loads((L/'results/qwen/rows.json').read_text())
assert len(rows)==1000 and len({r['image_sha256'] for r in rows})==930
protected={str(p):sha(p) for p in [B/'protocol.json',B/'final_independent_audit.json',B/'gate_decision.json',B/'final_report_zh.md']}
for p in (O/'sources').rglob('*.py'):protected[str(p)]=sha(p)
atomic(O/'protected_sources.json',protected)
sources={f:sha(O/f) for f in ['attention_backend.py','runtime.py','svhr_adapter.py','cpu_prepare.py','run.py']}
P=dict(status='FROZEN_BEFORE_GPU',model=base['model'],revision=base['revision'],base_protocol_sha256=sha(B/'protocol.json'),all_layout_hashes=base['all_layout_hashes'],rows_sha256=sha(L/'results/qwen/rows.json'),read_layers=list(READ),main_hidden_exit=22,skip_layers=list(range(1,7)),auxiliary_visual_prefix_layers=list(range(22,36)),auxiliary_full_blocks=list(range(22,35)),auxiliary_L35_QKV_only=True,max_new_tokens=128,greedy=True,natural_eos=True,batch_size=1,dtype='bfloat16',attention='SDPA_EFFICIENT_ATTENTION_ONLY',torch='2.6.0+cu124',transformers='4.57.6',source_commit='f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760',new_source_sha256=sources,expected=1000,smoke_ids=base['smoke_ids'],inputs_unchanged=True,Gold_passed_to_model=False,source_adaptation='Only bridge frozen eager KV-capture hook to existing SDPA dispatcher; inherit Physical.forward/SelectedKV/Original.context unchanged',user_authorization='Add frozen SVH-R1000, reuse completed Native/Original, publish independent GitHub branch',historical_strict_gate_unchanged=base['legacy_dense_physical_numerical_gate'],model_loads_planned=1)
atomic(O/'protocol.json',P)
atomic(O/'cpu_tests.json',dict(status='PASS',SelectedKV_prefill_layers=checks,decode_append_steps=3,frozen_forward_identity=True,frozen_context_identity=True,CUDA_initialized=torch.cuda.is_initialized(),model_loaded=False,unique_original_images=930,expected_questions=1000,source_SHA256=sources))
T=Path('/home/featurize/work/ATE/vlm_research/results/qwen3vl_ocrbench_svhr_efficient_attention');T.mkdir(exist_ok=True)
for f in ['protocol.json','cpu_tests.json','setup_bindings.json','protected_sources.json','github_source_verification.json']:shutil.copyfile(O/f,T/f)
print('CPU_SVHR_PASS',len(checks),'layers;1000 input hashes inherited;930 original images;CUDA not initialized')
