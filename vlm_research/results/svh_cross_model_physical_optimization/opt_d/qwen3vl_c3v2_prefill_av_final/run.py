"""Fixed three-smoke numerical verification; original gates every later request."""
import os,sys,gc,datetime,traceback,csv
from types import SimpleNamespace
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from ..qwen3vl_early_windows.experiment import prepare,hook_counts
from ..qwen3vl_c3v2_numerical_debug.common import sample
from ..qwen3vl_c3v2_numerical_debug.run import logits_pair
from ..qwen3vl_c3v2_numerical_repair2.probe import Probe,align,metrics,ORDER
from .common import *
from .engine import generate

OLD2=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_repair2')
def table(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows([{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def pack_probe(p):return dict(records=p.records,position_records=p.position_records,deepstack=p.deepstack,cache_snapshots=p.cache_snapshots)
def trim(records,layout):
 out={};images=set(layout['image_positions'])
 for key,r in records.items():
  if r['layer']<22 or r['layer']>35:continue
  positions=r['positions'];keep=[i for i,p in enumerate(positions) if p not in images];x=r['value'].index_select(r['axis'],torch.tensor(keep));out[key]={**r,'positions':[positions[i] for i in keep],'value':x,'shape':list(x.shape)}
 return out
def layer_diffs(ref,candidate,layout,tag):
 rows=[];text=[i for i in range(layout['input_length']) if i not in layout['image_positions']]
 for key,b in candidate.items():
  if key not in ref:continue
  a=dict(ref[key]);step=int(key.split('/')[0][1:]);a['retained_positions']=text if step==0 else [layout['input_length']+step-1];x,y,pos=align(a,b,'text')
  if x is None:continue
  # QK values of dead image columns are intentionally different; compare only live nonimage keys.
  visible='all_aligned'
  if a.get('key_positions') is not None:
   keys=sorted(set(a['key_positions'])&set(b['key_positions']));keep=[i for i,p in enumerate(keys) if p not in layout['image_positions'] and p>=0];x=x.index_select(-1,torch.tensor(keep));y=y.index_select(-1,torch.tensor(keep));visible='nonimage_aligned_keys'
  for scope in ['retained_text','p_minus_1'] if step==0 else ['decode']:
   xx,yy=x,y
   if scope=='p_minus_1':
    at=pos.index(layout['p_minus_1']);ix=torch.tensor([at]);xx=x.index_select(a['axis'],ix);yy=y.index_select(b['axis'],ix)
   r=dict(comparison=tag,step=step,layer=a['layer'],stage=a['stage'],scope=scope,key_scope=visible,**metrics(xx,yy))
   bad=(xx!=yy).nonzero();r['different_elements']=len(bad);r['first_coordinates']=[dict(index=ix,reference=float(xx[tuple(ix)]),candidate=float(yy[tuple(ix)])) for ix in bad[:4].tolist()];rows.append(r)
 return rows
def semantics(ref,cp,layout):
 # mRoPE is compared on actual surviving absolute positions, without renumbering.
 for a,b in zip(ref['position_records'],cp.position_records):
  pa=a['cache_position'];pb=b['cache_position'];ix=torch.tensor([pa.index(p) for p in pb]);assert torch.equal(a['cos'].index_select(1,ix),b['cos']) and torch.equal(a['sin'].index_select(1,ix),b['sin']);assert torch.equal(torch.tensor(a['position_ids']).index_select(-1,ix),torch.tensor(b['position_ids']))
 assert len(ref['deepstack'])==len(cp.deepstack)==3
 for a,b in zip(ref['deepstack'],cp.deepstack):assert a['layer']==b['layer'] and torch.equal(a['features'],b['features']) and torch.equal(a['visual_mask'],b['visual_mask'])
 caches=[]
 for key,b in cp.cache_snapshots.items():
  if b['layer'] in range(1,7):assert b['keys'] is None;continue
  a=ref['cache_snapshots'][key];ix=torch.tensor([a['positions'].index(p) for p in b['positions']]);caches.append(dict(layer=b['layer'],step=b['step'],positions=b['positions'],keys=metrics(a['keys'].index_select(-2,ix),b['keys']),values=metrics(a['values'].index_select(-2,ix),b['values'])))
 return dict(mrope_strict=True,deepstack_strict=True,compact_cache_all_positions_original=True,cache_comparisons=caches)
def save_run(folder,name,r,v):
 for k in ['vision_ms','prefill_ms','ttft_ms','decode_ms','decode_ms_per_token','e2e_ms']:r.pop(k,None)
 p=PRIVATE/folder/(name+'_logits.pt');p.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(logits=v,token_ids=r['token_ids']),p);write(OUT/folder/(name+'.json'),dict(result=r,raw_logits=dict(path=str(p),sha256=sha(p),external_to_git=True)))
 return dict(logits=v,token_ids=r['token_ids'])
def comparisons(ref,rv,r,v,tag):
 rows=logits_pair(rv,v);length_equal=len(rv)==len(v)
 for row in rows:row.update(comparison=tag,generated_steps_equal=length_equal,complete_token_equal=ref['token_ids']==r['token_ids'],gate_pass=length_equal and row['max_abs']<=.5 and row['mean_abs']<=.02 and row['top1_equal'] and ref['token_ids']==r['token_ids'])
 return rows
def main(rank=0):
 check();torch.set_num_threads(1);source=read(OUT/'source_manifest.json')
 for path,h in source['files'].items():assert sha(ROOT/path)==h,path
 assert torch.__version__=='2.6.0+cu124' and __import__('transformers').__version__=='4.57.6'
 bindings=read(OUT/'sample_manifest.json');cases=read(OUT/'protocol.json')['smoke_order'];allrows=[];done=[]
 if rank==1:
  previous=read(OUT/'rank0/completion.json');cases=previous['samples_completed'];assert cases and cases[0]==ORIGINAL
 write(OUT/f'rank{rank}/environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],source_commit=source['commit'],mode='single_card_test' if rank==0 else 'independent_fixed_replication'))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval();hc=hook_counts(model)
 for sid in cases:
  check();folder=f'rank{rank}/{sid}';inputs,layout=prepare(processor,sample(sid));assert layout==bindings['input_bindings'][sid];write(OUT/folder/'input_binding.json',layout)
  # Native/No-op and HF reference are fully independent cache requests.
  nr,nv=generate(model,processor,inputs,layout,'N');xr,xv=generate(model,processor,inputs,layout,'N',noop=True);assert nr['token_ids']==xr['token_ids'] and len(nv)==len(xv) and all(torch.equal(a,b) for a,b in zip(nv,xv))
  with torch.inference_mode():hf=model.generate(**{k:v.cuda() for k,v in inputs.items()},do_sample=False,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True,logits_to_keep=1)
  assert hf.sequences[0,layout['input_length']:].tolist()==nr['token_ids'] and len(hf.scores)==len(nv) and all(torch.equal(a,b[0].float().cpu()) for a,b in zip(nv,hf.scores));del hf
  save_run(folder,'Native',nr,nv);save_run(folder,'Noop',xr,xv)
  if sid==ORIGINAL:
   fr,fv=generate(model,processor,inputs,layout,'F');refprobe=torch.load(OLD2/'phase1_rank0/F_probes.pt',weights_only=True)
   old=torch.load(OLD_RAW/'full_final_rank0/F_logits.pt',weights_only=True);assert fr['token_ids']==old['token_ids'] and all(torch.equal(a,b) for a,b in zip(fv,old['logits']))
  else:
   fp=Probe(model,layout,'F')
   with fp.context():fr,fv=generate(model,processor,inputs,layout,'F')
   refprobe=pack_probe(fp);del fp
  save_run(folder,'F',fr,fv)
  br,bv=generate(model,processor,inputs,layout,'B');save_run(folder,'B',br,bv);allrows+= [dict(sample=sid,**r) for r in comparisons(fr,fv,br,bv,'F__B')]
  before,vbefore=generate(model,processor,inputs,layout,'C',prefill_zero=False);save_run(folder,'C_before',before,vbefore);allrows += [dict(sample=sid,**r) for r in comparisons(fr,fv,before,vbefore,'F__C_before')]
  if sid==ORIGINAL:
   old=torch.load(OLD_RAW/'full_final_rank0/C_final_logits.pt',weights_only=True);assert before['token_ids']==old['token_ids'] and len(vbefore)==len(old['logits']) and all(torch.equal(a,b) for a,b in zip(vbefore,old['logits']))
  cp=Probe(model,layout,'C')
  with cp.context():cr,cv=generate(model,processor,inputs,layout,'C')
  assert hook_counts(model)==hc;dr=layer_diffs(refprobe['records'],cp.records,layout,'F__C_after');sem=semantics(refprobe,cp,layout);write(OUT/folder/'semantic_validation.json',sem);table(OUT/folder/'layerwise_errors.csv',dr)
  # Repeat without observer to prove readonly instrumentation and collect local costs.
  clean,cleanv=generate(model,processor,inputs,layout,'C');assert cr['token_ids']==clean['token_ids'] and len(cv)==len(cleanv) and all(torch.equal(a,b) for a,b in zip(cv,cleanv));save_run(folder,'C_after',clean,cleanv)
  rr=comparisons(fr,fv,clean,cleanv,'F__C_after');allrows += [dict(sample=sid,**r) for r in rr]
  wp,wv=generate(model,processor,inputs,layout,'W22-functional');pr,pv=generate(model,processor,inputs,layout,'P');save_run(folder,'W22-functional',wp,wv);save_run(folder,'P_after',pr,pv);allrows += [dict(sample=sid,**r) for r in comparisons(wp,wv,pr,pv,'W22-functional__P_after')]
  costs=[]
  for label,r in [('before',before),('after',clean)]:
   for rec in r['audit']['prefill_av_alignment']:costs.append(dict(sample=sid,condition=label,**rec))
  table(OUT/folder/'prefill_costs.csv',costs)
  saved=dict(F=trim(refprobe['records'],layout),C=trim(cp.records,layout),layout=layout,source_commit=source['commit']);p=PRIVATE/folder/'late_prefill_component_evidence.pt';torch.save(saved,p);write(OUT/folder/'component_evidence_index.json',dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,external_to_git=True))
  audit=clean['audit'];assert len(audit['fixed_lt_projections'])==56 and len(audit['prefill_av_alignment'])==14 and all(x['persistent_image_KV_added']==0 and x['non_target_mask_strict'] for x in audit['prefill_av_alignment']);assert all(r['hidden_rows']==len(cp.text) for r in audit['layers'] if r['step']==0 and r['layer']>=22)
  valid=all(x['gate_pass'] for x in rr) and all(r['gate_pass'] for r in allrows if r['sample']==sid and r['comparison'] in ['F__B','W22-functional__P_after']);assert hook_counts(model)==hc
  result=dict(sample=sid,full_C_gate=all(x['gate_pass'] for x in rr),all_pair_gates=valid,native_noop_hf_strict=True,probe_readonly_strict=True,hooks_clean=True,mrope_deepstack_strict=True,input_unchanged=True,finite_logits=True,complete_token_ids=clean['token_ids'],eos=clean['eos'],empty=clean['empty'],truncated=clean['truncated'],source_commit=source['commit']);write(OUT/folder/'completion.json',result);done.append(sid);table(OUT/f'rank{rank}/logits_comparison.csv',allrows)
  print('SMOKE',rank,sid,result,[(r['step'],r['max_abs'],r['mean_abs'],r['gate_pass']) for r in rr],flush=True)
  del cp,refprobe,saved;gc.collect()
  if not valid:break
 passed=len(done)==len(read(OUT/'protocol.json')['smoke_order']) and all(read(OUT/f'rank{rank}/{sid}/completion.json')['all_pair_gates'] for sid in done)
 write(OUT/f'rank{rank}/completion.json',dict(status='ALL_SMOKES_PASSED' if passed else 'STOPPED_AT_FAILED_SMOKE',samples_completed=done,all_smokes_passed=passed,all_parameters_bf16=all(p.dtype==torch.bfloat16 for p in model.parameters()),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source['commit']))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
 except BaseException as e:write(OUT/'worker_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
