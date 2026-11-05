"""Fixed numerical diagnostics only. No accuracy or benchmark entry point."""
import os,sys,gc,time,datetime,traceback,csv,subprocess
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,MODEL,hook_counts
from ..qwen3vl_c3v2_physical.engine import generate
from .common import *
from .probe import Probe,metrics

def check_time():
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(read(OUT/'time_budget.json')['GPU_cutoff_UTC'])
 assert __import__('shutil').disk_usage('/data').free>20*2**30
def tensor_save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)
 return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,external_to_git=True)
def logits_pair(a,b):
 rows=[]
 for step,(x,y) in enumerate(zip(a,b)):
  m=metrics(x,y);m.update(step=step,top1_equal=int(x.argmax())==int(y.argmax()),reference_top1=int(x.argmax()),candidate_top1=int(y.argmax()))
  indices=sorted(set(x.topk(20).indices.tolist()+y.topk(20).indices.tolist()))
  m['top20_union']=[dict(token_id=i,reference=float(x[i]),candidate=float(y[i]),signed_difference=float(y[i]-x[i])) for i in indices];rows.append(m)
 return rows
def shadow(model,probe,rank):
 rows=[];raw={}
 for key,x_cpu in probe.shadow_inputs.items():
  check_time();layer,name=key.split('/');layer=int(layer)
  if layer<22 and name not in ['q_proj','k_proj','v_proj','o_proj']:continue
  x=x_cpu.to('cuda:0');indices=list(range(x.shape[1]-1)) if layer<22 else probe.text
  idx=torch.tensor(indices,device=x.device);short=x.index_select(1,idx)
  block=model.model.language_model.layers[layer]
  module=getattr(block.self_attn if name in ['q_proj','k_proj','v_proj','o_proj'] else block.mlp,name)
  with torch.inference_mode():
   full=module(x);small=module(short)
   fp_full=torch.nn.functional.linear(x.float(),module.weight.float(),None if module.bias is None else module.bias.float())
   fp_short=torch.nn.functional.linear(short.float(),module.weight.float(),None if module.bias is None else module.bias.float())
  bf=metrics(full.index_select(1,idx),small);fp=metrics(fp_full.index_select(1,idx),fp_short)
  # Same shared BF16 input rows, independently executed full vs shorter GEMMs.
  rows.append(dict(rank=rank,layer=layer,op=name,full_rows=x.shape[1],physical_rows=len(indices),input_shared_equal=bool(torch.equal(x.index_select(1,idx),short)),bf16_max_abs=bf['max_abs'],bf16_mean_abs=bf['mean_abs'],bf16_rmse=bf['rmse'],bf16_equal=bf['equal'],fp32_max_abs=fp['max_abs'],fp32_mean_abs=fp['mean_abs'],fp32_equal=fp['equal']))
  raw[key]=dict(full_input_shape=list(x.shape),physical_input_shape=list(short.shape),positions=indices,
       bf16_reference=full.index_select(1,idx).cpu(),bf16_physical=small.cpu(),fp32_reference=fp_full.index_select(1,idx).cpu(),fp32_physical=fp_short.cpu())
  del x,short,full,small,fp_full,fp_short
 if rows:
  with (OUT/f'shard{rank}/bf16_shape_sensitivity.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return rows,raw
def main(rank,mode):
 torch.set_num_threads(1);check_time();frozen=read(OUT/'source_manifest.json')
 for name,h in frozen['files'].items():assert sha(ROOT/name)==h,name
 s=sample(ORIGINAL);expected=read(OUT/'sample_manifest.json')['input_bindings'][ORIGINAL]
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);inputs,layout=prepare(processor,s);assert layout==expected
 write(OUT/f'shard{rank}/environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,transformers=__import__('transformers').__version__,allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,allow_tf32=torch.backends.cuda.matmul.allow_tf32,source_commit=frozen['commit']))
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();initial=hook_counts(model)
 index=[];results={};vectors={}
 for c,noop in [('N',False),('Noop',True)]:
  r,v=generate(model,processor,inputs,layout,'N',True,True,noop=noop);results[c]=r;vectors[c]=v
  index.append(tensor_save(PRIVATE/f'shard{rank}/{c}_logits.pt',dict(logits=v,token_ids=r['token_ids'])))
 assert results['N']['token_ids']==results['Noop']['token_ids'] and len(vectors['N'])==len(vectors['Noop']) and all(torch.equal(a,b) for a,b in zip(vectors['N'],vectors['Noop']))
 with torch.inference_mode():
  hf=model.generate(**{k:v.to('cuda:0') for k,v in inputs.items()},do_sample=False,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True)
 assert hf.sequences[0,layout['input_length']:].tolist()==results['N']['token_ids']
 assert len(hf.scores)==len(vectors['N']) and all(torch.equal(a,b[0].float().cpu()) for a,b in zip(vectors['N'],hf.scores))
 del hf
 write(OUT/f'shard{rank}/native_validation.json',dict(native_noop_equal=True,native_hf_logits_equal=True,native_hf_tokens_equal=True,tensor_archives=index,token_ids=results['N']['token_ids']))
 for c in CONDITIONS:
  check_time();r,v=generate(model,processor,inputs,layout,c,True,True);assert hook_counts(model)==initial
  results[c]=r;vectors[c]=v
  index.append(tensor_save(PRIVATE/f'shard{rank}/{c}_logits.pt',dict(logits=v,token_ids=r['token_ids'])))
  write(OUT/f'shard{rank}/baseline/{c}.json',dict(sample=s,result=r,source_commit=frozen['commit'],input_hashes=layout['input_tensor_hashes']))
 comparison=logits_pair(vectors['F'],vectors['C'])
 assert comparison[0]['max_abs']==.4453125 and comparison[0]['mean_abs']==.08348099887371063,comparison
 assert results['F']['token_ids']==results['C']['token_ids']==[2152,151645]
 write(OUT/f'shard{rank}/numerical_baseline_reproduction.json',dict(status='EXACT_FAILURE_REPRODUCED',old_max_abs=.4453125,old_mean_abs=.08348099887371063,comparisons=comparison,original_thresholds_unchanged=True,tokens_F=results['F']['token_ids'],tokens_C=results['C']['token_ids']))
 factors=[]
 for a,b in PAIRS:
  for row in logits_pair(vectors[a],vectors[b]):
   row.update(reference=a,candidate=b,rank=rank,complete_token_equal=results[a]['token_ids']==results[b]['token_ids']);factors.append(row)
 write(OUT/f'shard{rank}/factor_isolation.json',factors)
 traces={};Fprobe=None
 if mode=='baseline':
  write(OUT/f'shard{rank}/tensor_index.json',index);return
 for c in CONDITIONS:
  check_time();probe=Probe(model,layout,c)
  with probe.context():r,v=generate(model,processor,inputs,layout,c,True,True)
  assert hook_counts(model)==initial
  assert r['token_ids']==results[c]['token_ids'] and len(v)==len(vectors[c]) and all(torch.equal(a,b) for a,b in zip(v,vectors[c])),('Probe changed logits',c)
  target=PRIVATE/f'shard{rank}/{c}_prefill_probes.pt'
  index.append(tensor_save(target,dict(records=probe.records,position_records=probe.position_records,deepstack=probe.deepstack,layout=layout,condition=c,source_commit=frozen['commit'],probe_logits_equal=True)))
  write(OUT/f'shard{rank}/probe_{c}_validation.json',dict(status='PASS',probe_logits_equal=True,records=len(probe.records),archive=index[-1],tensor_shapes={k:dict(shape=x['shape'],dtype=x['dtype'],positions=x['positions'],key_positions=x['key_positions']) for k,x in probe.records.items()}))
  if c=='F':Fprobe=probe;probe.records={}
  else:del probe
  gc.collect();print('PROBE',rank,c,'PASS',flush=True)
 if Fprobe:
  shadow_rows,raw=shadow(model,Fprobe,rank);index.append(tensor_save(PRIVATE/f'shard{rank}/bf16_shape_shadow.pt',raw));del Fprobe,raw
 write(OUT/f'shard{rank}/tensor_index.json',index)
 write(OUT/f'shard{rank}/diagnostic_completion.json',dict(status='FIXED_DIAGNOSTICS_COLLECTED',probe_readonly=True,all_matched_factor_comparisons_completed=True,no_quality_or_benchmark=True,source_commit=frozen['commit'],at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 rank=int(sys.argv[1]);mode=sys.argv[2] if len(sys.argv)>2 else 'full'
 try:main(rank,mode)
 except BaseException as e:
  write(OUT/f'shard{rank}/diagnostic_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),at=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
