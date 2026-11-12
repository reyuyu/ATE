import os,sys,gc,csv,datetime,traceback,pathlib
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,MODEL,hook_counts
from ..qwen3vl_c3v2_numerical_debug.common import sample
from ..qwen3vl_c3v2_numerical_debug.run import logits_pair
from ..qwen3vl_c3v2_numerical_debug.probe import metrics,align,ORDER
from .common import *
from .phase1 import table
from .probe import Probe
from .repair import generate

def main(rank=0,verify_only=False):
 torch.set_num_threads(1);check();source=read(OUT/'source_manifest_phase2.json')
 for path,h in source['files'].items():assert sha(ROOT/path)==h,path
 folder=f'phase2_rank{rank}';write(OUT/folder/'environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],source_commit=source['commit']))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);inputs,layout=prepare(processor,sample(ORIGINAL));assert layout==read(PREVIOUS/'sample_manifest.json')['input_bindings'][ORIGINAL]
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();hc=hook_counts(model);index=[]
 for c,no in [('N',False),('Noop',True),('F',False)]:
  r,v=generate(model,processor,inputs,layout,c if c!='Noop' else 'N',True,True,noop=no)
  old=torch.load(PRIVATE/'phase1_rank0'/f'{c}_logits.pt',weights_only=True);assert old['token_ids']==r['token_ids'] and all(torch.equal(a,b) for a,b in zip(old['logits'],v));index.append(save(PRIVATE/folder/f'{c}_logits.pt',dict(logits=v,token_ids=r['token_ids'])));write(OUT/folder/f'{c}.json',dict(result=r,source_commit=source['commit']))
 ref=torch.load(PRIVATE/'phase1_rank0/F_logits.pt',weights_only=True);wref=torch.load(PRIVATE/'phase1_rank0/W22-functional_logits.pt',weights_only=True)
 # Verify later-layer same-input projection pattern BEFORE expanding repair scope.
 later=[]
 if not verify_only:
  tr=torch.load(PRIVATE/'phase1_rank0/F_probes.pt',weights_only=True)['records'];ix=torch.tensor([i for i in range(layout['input_length']) if i not in layout['image_positions']],device='cuda:0')
  with torch.inference_mode():
   for i in range(24,36):
    for name in ['k_proj','v_proj','gate_proj','up_proj']:
     key=f'S0/L{i:02d}/'+('input_norm' if name in ['k_proj','v_proj'] else 'post_attention_norm');x=tr[key]['value'].cuda();mod=getattr(model.model.language_model.layers[i].self_attn if name in ['k_proj','v_proj'] else model.model.language_model.layers[i].mlp,name);a=mod(x).index_select(1,ix);b=mod(x.index_select(1,ix));later.append(dict(layer=i,operator=name,variant='BF16_native_same_input',full_rows=x.shape[1],physical_rows=len(ix),**metrics(a,b)))
  table('later_layer_operator_differences.csv',later);del tr
 allrows=[];candidates=['decode-zero-key'] if verify_only else ['decode-zero-key','FP32-L22+decode','FP32-late+decode'];proberows=[];first={}
 for variant in candidates:
  if variant=='FP32-late+decode':assert any(not r['equal'] for r in later if r['layer']>=24),'No measured evidence to expand'
  for c in (['B','C'] if verify_only else ['B','P','C'] if variant=='decode-zero-key' else ['P','C']):
   check();r,v=generate(model,processor,inputs,layout,c,True,True,variant=variant);assert hook_counts(model)==hc
   key=variant+'__'+c;index.append(save(PRIVATE/folder/f'{key}_logits.pt',dict(logits=v,token_ids=r['token_ids'])));write(OUT/folder/f'{key}.json',dict(result=r,source_commit=source['commit'],input_hashes=layout['input_tensor_hashes']))
   base=wref if c=='P' else ref;reference='W22-functional' if c=='P' else 'F'
   for x in logits_pair(base['logits'],v):
    x.update(variant=variant,reference=reference,candidate=c,complete_token_equal=r['token_ids']==base['token_ids']);x['gate_pass']=x['max_abs']<=.5 and x['mean_abs']<=.02 and x['top1_equal'] and x['complete_token_equal'];allrows.append(x)
   probe=Probe(model,layout,c)
   with probe.context():rr,vv=generate(model,processor,inputs,layout,c,True,True,variant=variant)
   assert rr['token_ids']==r['token_ids'] and len(vv)==len(v) and all(torch.equal(a,b) for a,b in zip(vv,v)) and hook_counts(model)==hc
   index.append(save(PRIVATE/folder/f'{key}_probes.pt',dict(records=probe.records,cache_snapshots=probe.cache_snapshots,position_records=probe.position_records,deepstack=probe.deepstack,layout=layout,source_commit=source['commit'],readonly_logits_strict=True)))
   tr=torch.load(PRIVATE/f'phase1_rank0/{reference}_probes.pt',weights_only=True)['records']
   for step in [0,1]:
    for i in range(38):
     for stage in ORDER:
      k=f'S{step}/L{i:02d}/{stage}'
      if k not in tr or k not in probe.records:continue
      a=tr[k];b=probe.records[k];a['retained_positions']=probe.text if step==0 else [layout['input_length']];x,y,pos=align(a,b,'text')
      if x is None:continue
      row=dict(variant=variant,comparison=reference+'__'+c,step=step,layer=i,stage=stage,**metrics(x,y));proberows.append(row)
      if not row['equal'] and f'{key}/step{step}' not in first:first[f'{key}/step{step}']=row
   print('CANDIDATE',rank,variant,c,[(x['step'],x['max_abs'],x['mean_abs'],x['gate_pass']) for x in allrows if x['variant']==variant and x['candidate']==c],flush=True);del tr,probe;gc.collect()
 write(OUT/folder/'logits_comparison.json',allrows);write(OUT/folder/'first_divergences.json',first);write(OUT/folder/'tensor_index.json',index)
 table(folder.replace('/','_')+'_component_differences.csv',proberows)
 write(OUT/folder/'completion.json',dict(status='TARGETED_CANDIDATES_MEASURED',source_commit=source['commit'],native_noop_and_F_strict_to_phase1=True,all_probes_readonly=True,all_parameters_bf16=all(p.dtype==torch.bfloat16 for p in model.parameters()),at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0,len(sys.argv)>2)
 except BaseException as e:write(OUT/'phase2_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
