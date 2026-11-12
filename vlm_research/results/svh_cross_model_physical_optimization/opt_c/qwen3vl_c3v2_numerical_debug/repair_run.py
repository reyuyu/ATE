import sys,os,datetime,gc,traceback
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,MODEL,hook_counts
from .common import *
from .run import check_time,tensor_save,logits_pair
from .probe import Probe
from .repair import generate

def main(rank=0):
 torch.set_num_threads(1);check_time();src=read(OUT/'source_manifest.json')
 for p,h in src['files'].items():assert sha(ROOT/p)==h,p
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);s=sample(ORIGINAL);inputs,layout=prepare(processor,s);assert layout==read(OUT/'sample_manifest.json')['input_bindings'][ORIGINAL]
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();initial=hook_counts(model);index=[];outputs={};logits={}
 for c,no in [('N',False),('Noop',True),('F',False),('B',False),('C',False)]:
  check_time();r,v=generate(model,processor,inputs,layout,'N' if c=='Noop' else c,True,True,noop=no)
  assert hook_counts(model)==initial
  outputs[c]=r;logits[c]=v;index.append(tensor_save(PRIVATE/f'repair_shard{rank}/{c}_logits.pt',dict(logits=v,token_ids=r['token_ids'])))
  write(OUT/f'repair_shard{rank}/{c}.json',dict(sample=s,result=r,source_commit=src['commit'],input_hashes=layout['input_tensor_hashes']))
 assert outputs['N']['token_ids']==outputs['Noop']['token_ids'] and all(torch.equal(a,b) for a,b in zip(logits['N'],logits['Noop']))
 with torch.inference_mode():hf=model.generate(**{k:v.to('cuda:0') for k,v in inputs.items()},do_sample=False,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True)
 assert hf.sequences[0,layout['input_length']:].tolist()==outputs['N']['token_ids'] and len(hf.scores)==len(logits['N']) and all(torch.equal(a,b[0].float().cpu()) for a,b in zip(logits['N'],hf.scores));del hf
 probe=Probe(model,layout,'C')
 with probe.context():r,v=generate(model,processor,inputs,layout,'C',True,True)
 assert len(v)==len(logits['C']) and all(torch.equal(a,b) for a,b in zip(v,logits['C'])) and r['token_ids']==outputs['C']['token_ids']
 index.append(tensor_save(PRIVATE/f'repair_shard{rank}/C_prefill_probes.pt',dict(records=probe.records,position_records=probe.position_records,deepstack=probe.deepstack,cache_snapshots=probe.cache_snapshots,layout=layout,condition='C-one-dead-key',source_commit=src['commit'],probe_logits_equal=True)))
 rows=[]
 for a,b in [('F','B'),('F','C')]:
  for x in logits_pair(logits[a],logits[b]):
   x.update(reference=a,candidate=b,complete_token_equal=outputs[a]['token_ids']==outputs[b]['token_ids']);rows.append(x)
 gate=all(x['max_abs']<=.5 and x['mean_abs']<=.02 and x['top1_equal'] and x['complete_token_equal'] for x in rows if x['candidate']=='C')
 assert hook_counts(model)==initial
 write(OUT/f'repair_shard{rank}/repair_validation.json',dict(status='ORIGINAL_CASE_PASSED' if gate else 'ORIGINAL_CASE_FAILED_STOP_EXPANSION',comparisons=rows,native_noop_strict=True,native_hf_tokens_and_logits_strict=True,probe_logits_strict=True,all_parameters_bf16=all(p.dtype==torch.bfloat16 for p in model.parameters()),all_layers_unchanged=True,tolerances_unchanged=True,source_commit=src['commit'],extra_samples_run=0,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 write(OUT/f'repair_shard{rank}/tensor_index.json',index)
 print('REPAIR ORIGINAL GATE',gate,[(r['reference'],r['candidate'],r['step'],r['max_abs'],r['mean_abs']) for r in rows],flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
 except BaseException as e:write(OUT/'repair_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
