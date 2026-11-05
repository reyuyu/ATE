"""One fixed Lt policy, extended only to measured next late-layer divergences."""
import os,sys,gc,traceback,datetime,csv,subprocess
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,hook_counts
from ..qwen3vl_c3v2_numerical_debug.common import sample
from ..qwen3vl_c3v2_numerical_debug.run import logits_pair
from ..qwen3vl_c3v2_numerical_repair2.repair import Repair2Execution
from ..qwen3vl_c3v2_numerical_repair2.probe import Probe,align,metrics,ORDER
from ..qwen3vl_c3v2_physical import engine
from .common import *
from .candidates import LtPlan

class FixedLtExecution(Repair2Execution):
 def __init__(self,*a,repair_layers=(),**kw):
  super().__init__(*a,variant='decode-zero-key',**kw);self.repair_layers=repair_layers;self.lt_records=[];self.plans={};self.workspace=None
 @contextmanager
 def context(self):
  try:
   with ExitStack() as st:
    for i in self.repair_layers:
     for name in ['k_proj','v_proj','gate_proj','up_proj']:
      mod=getattr(self.d.layers[i].self_attn if name in ['k_proj','v_proj'] else self.d.layers[i].mlp,name);fn=mod.forward
      def projected(x,mod=mod,fn=fn,i=i,name=name):
       if not(self.physical and self.step==0):return fn(x)
       assert x.dtype==torch.bfloat16 and x.shape[0]==1 and x.shape[1]==len(self.text) and mod.bias is None
       a=x.reshape(-1,x.shape[-1]).contiguous();m,k=a.shape;n=mod.weight.shape[0];key=(m,n,k)
       if self.workspace is None:self.workspace=torch.empty(8*2**20,device=x.device,dtype=torch.uint8)
       if key not in self.plans:
        dense=LtPlan(self.n,n,k,self.workspace);physical=LtPlan(m,n,k,self.workspace);assert physical.select(dense,0)==0
        self.plans[key]=(dense,physical)
       dense,physical=self.plans[key];y=physical.run(a,mod.weight).reshape(1,m,n)
       self.lt_records.append(dict(layer=i,operator=name,input_shape=list(x.shape),weight_shape=list(mod.weight.shape),output_shape=list(y.shape),config=physical.config(),actual_gemm_rows=m,dense_descriptor_only=self.n,dense_projection_executed=False,weight_conversion_bytes=0,workspace_bytes=self.workspace.numel(),matrix_flops=2*m*n*k))
       return y
      st.enter_context(patch.object(mod,'forward',projected))
    st.enter_context(super().context());yield self
  finally:
   for pair in self.plans.values():
    for p in pair:p.close()
   self.plans.clear();self.workspace=None
 def audit(self):
  a=super().audit();a['fixed_lt_projections']=self.lt_records;a['repair_layers']=list(self.repair_layers);return a

def generate(model,processor,inputs,layout,c,layers,**kw):
 class Bound(FixedLtExecution):
  def __init__(self,*a,**k):super().__init__(*a,repair_layers=layers,**k)
 with patch.object(engine,'Execution',Bound):return engine.generate(model,processor,inputs,layout,c,True,True,**kw)

def save_run(folder,key,r,v):
 # Built-in one-pass timers are diagnostic implementation bookkeeping, not a benchmark.
 for k in ['vision_ms','prefill_ms','ttft_ms','decode_ms','decode_ms_per_token','e2e_ms']:r.pop(k,None)
 path=PRIVATE/folder/(key+'_logits.pt');path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(logits=v,token_ids=r['token_ids']),path)
 write(OUT/folder/(key+'.json'),dict(result=r,raw_logits=dict(path=str(path),sha256=sha(path),external_to_git=True)))

def compare(folder,key,base,r,v):
 assert len(base['logits'])==len(v),'Generated step count changed'
 rows=logits_pair(base['logits'],v)
 for x in rows:x.update(comparison=key,complete_token_equal=r['token_ids']==base['token_ids'],gate_pass=x['max_abs']<=.5 and x['mean_abs']<=.02 and x['top1_equal'] and r['token_ids']==base['token_ids'])
 return rows

def differences(ref,probe):
 rows=[];first={}
 for step in sorted(set(int(k.split('/')[0][1:]) for k in probe.records)):
  for layer in range(38):
   for stage in ORDER:
    key=f'S{step}/L{layer:02d}/{stage}'
    if key not in ref or key not in probe.records:continue
    a=ref[key];a['retained_positions']=probe.text if step==0 else [probe.n+step-1];x,y,pos=align(a,probe.records[key],'text')
    if x is None:continue
    row=dict(step=step,layer=layer,stage=stage,**metrics(x,y));rows.append(row)
    if not row['equal'] and str(step) not in first:first[str(step)]=row
 return rows,first

def main(rank=0,mode='original'):
 torch.set_num_threads(1);check();freeze=read(OUT/'source_manifest_full_model.json')
 for path,h in freeze['files'].items():assert sha(ROOT/path)==h,path
 folder=f'full_v2_rank{rank}';write(OUT/folder/'environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],source_commit=freeze['commit'],mode=mode))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();hc=hook_counts(model)
 key='GQA__06905471';inputs,layout=prepare(processor,sample(key));assert layout==read(ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_debug/sample_manifest.json')['input_bindings'][key]
 allrows=[];scope=[22,23] if mode=='original' else read(OUT/'full_v2_rank0/completion.json')['final_repair_layers'];attempts=[]
 if mode=='original':
  prior=read(OUT/'full_rank0/C_L22_components.json');lookup={(r['step'],r['layer'],r['stage']):r for r in prior}
  assert lookup[0,22,'layer_output']['equal'] and lookup[0,23,'input_norm']['equal'] and not lookup[0,23,'k_proj']['equal']
 for c in ['N','Noop','F','W22-functional','B']:
  r,v=generate(model,processor,inputs,layout,'N' if c=='Noop' else c,[],noop=c=='Noop');assert hook_counts(model)==hc
  if c=='Noop':assert r['token_ids']==nr['token_ids'] and len(v)==len(nv) and all(torch.equal(a,b) for a,b in zip(v,nv))
  if c=='N':nr,nv=r,v
  old=torch.load(RAW_PARENT/'phase1_rank0'/f'{c}_logits.pt',weights_only=True) if c!='B' else torch.load(RAW_PARENT/'phase2_rank0/decode-zero-key__B_logits.pt',weights_only=True)
  assert old['token_ids']==r['token_ids'] and len(v)==len(old['logits']) and all(torch.equal(a,b) for a,b in zip(v,old['logits'])),c
  save_run(folder,c,r,v)
  if c=='F':ref=dict(logits=v,token_ids=r['token_ids'])
  if c=='W22-functional':wref=dict(logits=v,token_ids=r['token_ids'])
  if c=='B':allrows+=compare(folder,'F__B',ref,r,v)
 tr=torch.load(RAW_PARENT/'phase1_rank0/F_probes.pt',weights_only=True)['records']
 for attempt in range(14):
  check();probe=Probe(model,layout,'C')
  with probe.context():r,v=generate(model,processor,inputs,layout,'C',scope)
  assert hook_counts(model)==hc;name='C_L'+str(scope[-1]);save_run(folder,name,r,v);rows=compare(folder,'F__'+name,ref,r,v);allrows+=rows;dr,first=differences(tr,probe)
  write(OUT/folder/(name+'_first_divergence.json'),first);write(OUT/folder/(name+'_components.json'),dr)
  rec=dict(repair_layers=scope.copy(),comparison=rows,first_divergences=first);attempts.append(rec);print('FULL_ATTEMPT',rank,rec,flush=True)
  passed=all(x['gate_pass'] for x in rows)
  if passed or mode!='original':break
  next_layer=scope[-1]+1;lookup={(x['step'],x['layer'],x['stage']):x for x in dr}
  # A transient subnormal AV difference can vanish before the layer boundary.
  # Extend only when the NEXT layer input is strictly equal and its own K/V projection is not.
  if next_layer>35 or not lookup.get((0,next_layer,'input_norm'),{}).get('equal',False) or not any(not lookup.get((0,next_layer,s),{'equal':True})['equal'] for s in ['k_proj','v_proj']):break
  # Extension is triggered only by measured first divergence, same fixed policy; no layer selection search.
  scope.append(next_layer);del probe;gc.collect()
 # Permission matched visual-exit pair and uninstrumented repeat.
 r2,v2=generate(model,processor,inputs,layout,'C',scope);assert r2['token_ids']==r['token_ids'] and len(v)==len(v2) and all(torch.equal(a,b) for a,b in zip(v,v2))
 save_run(folder,'C_final',r2,v2);rp,vp=generate(model,processor,inputs,layout,'P',scope);save_run(folder,'P_final',rp,vp);allrows+=compare(folder,'W22-functional__P_final',wref,rp,vp)
 write(OUT/folder/'logits_comparison.json',allrows);write(OUT/folder/'completion.json',dict(status='ORIGINAL_PASSED' if passed else 'PARTIAL_REPAIR',original_full_C_gate=passed,final_repair_layers=scope,attempts=attempts,native_noop_strict=True,native_F_W_B_historical_strict=True,probe_readonly_strict=True,hooks_clean=hook_counts(model)==hc,all_parameters_bf16=all(p.dtype==torch.bfloat16 for p in model.parameters()),source_commit=freeze['commit'],at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0,sys.argv[2] if len(sys.argv)>2 else 'original')
 except BaseException as e:write(OUT/'full_model_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
