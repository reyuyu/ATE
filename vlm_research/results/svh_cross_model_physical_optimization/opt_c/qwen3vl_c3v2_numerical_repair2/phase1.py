"""New task: original-case reproduction, Decode localization and operator shadows."""
import os,sys,gc,csv,datetime,traceback,time
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,MODEL,hook_counts
from ..qwen3vl_c3v2_numerical_debug.common import sample
from ..qwen3vl_c3v2_numerical_debug.repair import generate
from ..qwen3vl_c3v2_numerical_debug.run import logits_pair
from ..qwen3vl_c3v2_numerical_debug.probe import metrics,align,ORDER
from .common import *
from .probe import Probe

def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(rank=0):
 torch.set_num_threads(1);check();source=read(OUT/'source_manifest.json')
 for path,h in source['files'].items():assert sha(ROOT/path)==h,path
 write(OUT/f'phase1_rank{rank}/environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],source_commit=source['commit'],torch=torch.__version__,transformers=__import__('transformers').__version__,bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);s=sample(ORIGINAL);inputs,layout=prepare(processor,s);assert layout==read(PREVIOUS/'sample_manifest.json')['input_bindings'][ORIGINAL]
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();hc=hook_counts(model);index=[];out={};vec={};probes={}
 for c in ['N','Noop']+CONDITIONS:
  check();r,v=generate(model,processor,inputs,layout,'N' if c=='Noop' else c,True,True,noop=c=='Noop');assert hook_counts(model)==hc
  out[c]=r;vec[c]=v;index.append(save(PRIVATE/f'phase1_rank{rank}/{c}_logits.pt',dict(logits=v,token_ids=r['token_ids'])));write(OUT/f'phase1_rank{rank}/{c}.json',dict(result=r,source_commit=source['commit'],input_hashes=layout['input_tensor_hashes']))
 assert all(torch.equal(a,b) for a,b in zip(vec['N'],vec['Noop'])) and out['N']['token_ids']==out['Noop']['token_ids']
 for c in ['F','B','C']:
  old=torch.load(pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_debug/repair_shard0')/f'{c}_logits.pt',weights_only=True);assert old['token_ids']==out[c]['token_ids'] and all(torch.equal(a,b) for a,b in zip(old['logits'],vec[c])),('Previous early repair not reproduced',c)
 with torch.inference_mode():hf=model.generate(**{k:v.to('cuda:0') for k,v in inputs.items()},do_sample=False,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True)
 assert hf.sequences[0,layout['input_length']:].tolist()==out['N']['token_ids'] and all(torch.equal(a,b[0].float().cpu()) for a,b in zip(vec['N'],hf.scores));del hf
 rows=[]
 for a,b in PAIRS:
  for r in logits_pair(vec[a],vec[b]):r.update(reference=a,candidate=b,complete_token_equal=out[a]['token_ids']==out[b]['token_ids']);rows.append(r)
 write(OUT/f'phase1_rank{rank}/logits_comparison.json',rows)
 for c in CONDITIONS:
  probe=Probe(model,layout,c)
  with probe.context():r,v=generate(model,processor,inputs,layout,c,True,True)
  assert hook_counts(model)==hc and r['token_ids']==out[c]['token_ids'] and len(v)==len(vec[c]) and all(torch.equal(a,b) for a,b in zip(v,vec[c]))
  probes[c]=probe;index.append(save(PRIVATE/f'phase1_rank{rank}/{c}_probes.pt',dict(records=probe.records,cache_snapshots=probe.cache_snapshots,position_records=probe.position_records,deepstack=probe.deepstack,layout=layout,source_commit=source['commit'],readonly_logits_strict=True)))
  print('READONLY PROBE',c,'PASS',flush=True)
 diffs=[];first={}
 for a,b in PAIRS:
  for step in [0,1]:
   for i in range(38):
    for st in ORDER:
     key=f'S{step}/L{i:02d}/{st}'
     if key not in probes[a].records or key not in probes[b].records:continue
     x=probes[a].records[key];y=probes[b].records[key];x['retained_positions']=probes[a].text if step==0 else [layout['input_length']]
     u,v,pos=align(x,y,'text')
     if u is None:continue
     row=dict(comparison=a+'__'+b,step=step,layer=i,stage=st,reference_shape=str(x['shape']),candidate_shape=str(y['shape']),**metrics(u,v));diffs.append(row)
     if step==1 and not row['equal'] and a+'__'+b not in first:first[a+'__'+b]=row
 table('phase1_component_differences.csv',diffs);write(OUT/'decode_first_divergence.json',first)
 # Same-input late-layer shadows. No padding, no previously failed layouts.
 ops=[];raw={}
 with torch.inference_mode():
  for key,x_cpu in probes['F'].shadow_inputs.items():
   step,layer,name=key.split('/');layer=int(layer)
   if int(step)!=0 or layer not in [22,23] or name not in ['k_proj','v_proj','gate_proj','up_proj','down_proj']:continue
   block=model.model.language_model.layers[layer];mod=getattr(block.self_attn if name in ['k_proj','v_proj'] else block.mlp,name);x=x_cpu.cuda();ix=torch.tensor(probes['F'].text,device=x.device);short=x.index_select(1,ix);ref=mod(x).index_select(1,ix)
   variants={'BF16_native_26':lambda:mod(short),'FP32_local_round_BF16':lambda:torch.nn.functional.linear(short.float(),mod.weight.float(),None if mod.bias is None else mod.bias.float()).to(short.dtype),'BF16_batched_GEMM':lambda:torch.bmm(short,mod.weight.T.unsqueeze(0))+(0 if mod.bias is None else mod.bias)}
   for name2,fn in variants.items():
    ts=time.perf_counter();torch.cuda.synchronize();a=fn();torch.cuda.synchronize();ms=(time.perf_counter()-ts)*1000
    ops.append(dict(layer=layer,operator=name,variant=name2,full_rows=x.shape[1],physical_rows=short.shape[1],diagnostic_single_call_wall_ms=ms,**metrics(ref,a)));raw[key+'/'+name2]=a.cpu()
   flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
   try:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;a=mod(short)
   finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
   ops.append(dict(layer=layer,operator=name,variant='BF16_no_reduced_accumulation',full_rows=x.shape[1],physical_rows=short.shape[1],diagnostic_single_call_wall_ms=None,**metrics(ref,a)));raw[key+'/BF16_no_reduced_accumulation']=a.cpu();raw[key+'/reference']=ref.cpu()
 table('operator_differences.csv',ops);index.append(save(PRIVATE/f'phase1_rank{rank}/operator_shadows.pt',raw))
 # Preserve exact Decode attention inputs for targeted repair design after B1.
 index.append(save(PRIVATE/f'phase1_rank{rank}/F_attention_inputs.pt',probes['F'].attention_shadow))
 write(OUT/f'phase1_rank{rank}/tensor_index.json',index);write(OUT/f'phase1_rank{rank}/completion.json',dict(status='PHASE1_LOCALIZED',native_noop_strict=True,native_hf_strict=True,previous_repair_vectors_strict=True,probes_readonly=True,source_commit=source['commit'],at=datetime.datetime.now(datetime.timezone.utc).isoformat(),first_decode=first))
 print('FIRST DECODE',first,flush=True);print('OPERATOR SUMMARY',[(x['layer'],x['operator'],x['variant'],x['mean_abs']) for x in ops],flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
 except BaseException as e:write(OUT/'phase1_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
