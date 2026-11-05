"""CPU consolidation, semantic and raw evidence checks; no new GPU inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,csv,subprocess,pathlib,gc,collections,shutil,json
import torch,numpy as np
from vlm_research.qwen3vl_c3v2_numerical_repair2.common import *
from vlm_research.qwen3vl_c3v2_numerical_debug.probe import metrics,align
torch.set_num_threads(1)
def table(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def archive(run,name):return torch.load(PRIVATE/run/(name+'_probes.pt'),map_location='cpu',weights_only=True)
def vectors(run,name):return torch.load(PRIVATE/run/(name+'_logits.pt'),map_location='cpu',weights_only=True)
source1=read(OUT/'source_manifest.json');source2=read(OUT/'source_manifest_phase2.json');layout=read(PREVIOUS/'sample_manifest.json')['input_bindings'][ORIGINAL];n=layout['input_length'];images=set(layout['image_positions']);text=[i for i in range(n) if i not in images]
# Assemble all metrics, including negative candidates and independent replicate.
rows=[]
for run in ['phase1_rank0','phase2_rank0','phase2_rank1']:
 for r in read(OUT/run/'logits_comparison.json'):
  r=dict(r);r['run']=run;r.setdefault('variant','previous-prefill-zero-key');r.setdefault('gate_pass',r['max_abs']<=.5 and r['mean_abs']<=.02 and r['top1_equal'] and r['complete_token_equal']);rows.append(r)
table('logits_comparison.csv',[{k:v for k,v in r.items() if k!='top20_union'} for r in rows]);write(OUT/'logits_comparison_full.json',rows)
candidates=[]
for run in ['phase2_rank0','phase2_rank1']:
 groups=collections.defaultdict(list)
 for r in rows:
  if r['run']==run:groups[(r['variant'],r['candidate'])].append(r)
 for (variant,c),rs in groups.items():
  audit=read(OUT/run/(variant+'__'+c+'.json'))['result']['audit'];high=audit['local_fp32'];key=variant+'__'+c
  first=read(OUT/run/'first_divergences.json');r0=next(r for r in rs if r['step']==0);r1=next(r for r in rs if r['step']==1)
  candidates.append(dict(run=run,variant=variant,comparison=rs[0]['reference']+'__'+c,first_prefill_stage=first.get(key+'/step0',{}).get('stage','NONE'),first_prefill_layer=first.get(key+'/step0',{}).get('layer','NONE'),first_decode_stage=first.get(key+'/step1',{}).get('stage','NONE'),first_decode_layer=first.get(key+'/step1',{}).get('layer','NONE'),step0_mean=r0['mean_abs'],step0_max=r0['max_abs'],step1_mean=r1['mean_abs'],step1_max=r1['max_abs'],all_effective_steps_pass=all(r['gate_pass'] for r in rs),physical_hidden_rows=26 if c in ['P','C'] else 122,fp32_call_count=len(high),fp32_single_run_total_gpu_ms=sum(x['single_call_gpu_ms'] for x in high),fp32_weight_cast_bytes_total=sum(x['fp32_weight_conversion_bytes'] for x in high),decode_zero_columns_per_active_layer=96,persistent_dummy_keys=0))
table('repair_candidates.csv',candidates);table('failed_candidate_steps.csv',[{k:v for k,v in r.items() if k!='top20_union'} for r in rows if not r['gate_pass']])
# Raw logits recomputation, input/token binding, Native/Noop, two-card result equality.
logits={};ldesc=[];index=[]
for run in ['phase1_rank0','phase2_rank0','phase2_rank1']:
 for item in read(OUT/run/'tensor_index.json'):
  assert sha(item['path'])==item['sha256'];index.append(dict(run=run,**item))
  if item['path'].endswith('_logits.pt'):
   name=pathlib.Path(item['path']).name[:-len('_logits.pt')];x=vectors(run,name);key=run+'__'+name;logits[key]=torch.stack(x['logits']).numpy();ldesc.append(dict(key=key,shape=list(logits[key].shape),dtype='float32 from effective BF16 logits',token_ids=x['token_ids'],array_bytes_sha256=__import__('hashlib').sha256(logits[key].tobytes()).hexdigest()))
for r in rows:
 run=r['run'];ref=vectors('phase1_rank0',r['reference']);name=r['candidate'] if run=='phase1_rank0' else r['variant']+'__'+r['candidate'];cur=vectors(run,name);a=ref['logits'][r['step']];b=cur['logits'][r['step']];m=metrics(a,b)
 for k in ['max_abs','mean_abs','rmse','relative_l2']:assert m[k]==r[k],(run,k)
 assert ref['token_ids']==cur['token_ids']==[2152,151645]
for run in ['phase1_rank0','phase2_rank0','phase2_rank1']:assert np.array_equal(logits[run+'__N'],logits[run+'__Noop'])
for c in ['N','Noop','F','decode-zero-key__B','decode-zero-key__C']:assert np.array_equal(logits['phase2_rank0__'+c],logits['phase2_rank1__'+c])
(OUT/'arrays').mkdir(exist_ok=True);np.savez_compressed(OUT/'arrays/effective_logits.npz',**logits);write(OUT/'arrays/effective_logits.json',dict(sha256=sha(OUT/'arrays/effective_logits.npz'),arrays=ldesc))
# Cache contents/prefix/absolute position and all layer execution invariants.
f=archive('phase1_rank0','F');checks=[];physical=[];archives=[]
for c in CONDITIONS:archives.append(('phase1_rank0',c,c))
for run in ['phase2_rank0','phase2_rank1']:
 for c in candidates:
  if c['run']==run:archives.append((run,c['variant']+'__'+c['comparison'].split('__')[-1],c['comparison'].split('__')[-1]))
first_arrays={};descs=[];coordinates=[]
def extract(label,a,b,keys):
 for key in keys:
  x=a['records'][key];y=b['records'][key];x['retained_positions']=text if key.startswith('S0/') else [n];u,v,pos=align(x,y,'text')
  for side,t in [('reference',u),('candidate',v)]:
   k=label+'__'+key.replace('/','_')+'__'+side;t=t.contiguous();dt=str(t.dtype);first_arrays[k]=t.view(torch.int16).numpy() if t.dtype==torch.bfloat16 else t.numpy();descs.append(dict(key=k,dtype=dt,shape=list(t.shape),positions=pos,array_bytes_sha256=__import__('hashlib').sha256(first_arrays[k].tobytes()).hexdigest()))
b=archive('phase1_rank0','B');extract('before_decode_F_B',f,b,['S1/L12/'+s for s in ['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k','scores_before_mask','scores_after_mask','attention_probabilities','attention_AV_output','o_proj_raw','layer_output']]);u,v,pos=align(f['records']['S1/L12/attention_AV_output'],b['records']['S1/L12/attention_AV_output']);coordinates=(u!=v).nonzero().tolist();write(OUT/'decode_first_difference_coordinates.json',dict(layer=12,stage='attention_AV_output',changed_count=len(coordinates),coordinates=[dict(index=ix,reference=float(u[tuple(ix)]),candidate=float(v[tuple(ix)])) for ix in coordinates]));del b
cache_count=0;canonical=[]
for run,name,c in archives:
 a=archive(run,name);r=read(OUT/run/(name+'.json'))['result'];au=r['audit'];skip=c in ['B','C'];isphysical=c in ['P','C'];compact=c in ['P','B','C'];normal=[i for i in range(36) if not(skip and i in range(1,7))]
 assert r['eos'] and not r['truncated'] and not r['empty'] and r['token_ids']==[2152,151645] and not r['physical_fallback']
 assert au['deepstack']==[0,1,2]
 mlpevents=[x for x in au['attention_events'] if x.get('mlp_and_norm_executed')];assert len(mlpevents)==72 and all(x['residual_exact'] for x in mlpevents)
 for x in au['layers']:
  if x['step']==0:assert x['hidden_rows']==(26 if isphysical and x['layer']>=22 else 122)
  else:assert x['hidden_rows']==1 and x['logical_positions']==[122]
 for key,s in a['cache_snapshots'].items():
  i=s['layer'];step=s['step'];old=a['cache_snapshots'][f'0/{i}'];cache_count+=1
  if skip and i in range(1,7):assert s['keys'] is None and s['values'] is None and not s['positions'];continue
  expected=(text+list(range(n,n+step))) if compact else list(range(n+step));assert s['positions']==expected
  if step==1:
   count=len(old['positions']);assert torch.equal(s['keys'][...,:count,:],old['keys']) and torch.equal(s['values'][...,:count,:],old['values'])
  if i>=22 and step==0:
   rk=a['records'][f'S0/L{i:02d}/rope_k'];vp=a['records'][f'S0/L{i:02d}/v_proj'];ix=torch.tensor([rk['positions'].index(p) for p in expected]);assert torch.equal(rk['value'].index_select(2,ix),s['keys']);vv=vp['value'].reshape(1,len(vp['positions']),8,128).transpose(1,2).index_select(2,ix);assert torch.equal(vv,s['values'])
  if run.startswith('phase2') and name=='decode-zero-key__B':
   ref=f['cache_snapshots'][key];ix=torch.tensor([ref['positions'].index(p) for p in expected]);assert torch.equal(ref['keys'].index_select(2,ix),s['keys']) and torch.equal(ref['values'].index_select(2,ix),s['values'])
 if run.startswith('phase2'):
  assert len(au['decode_temporary_zero_keys'])==len(normal)
  for pad in au['decode_temporary_zero_keys']:assert pad['dummy_probability']==0 and pad['persistent_dummy_KV']==0 and pad['actual_attention_key_rows']==123 and pad['original_KV_rows']==27
  if skip:assert all(au['kv']['layers'][i]['capacity_bytes']==0 for i in range(1,7));assert not any(x['step']>0 and x['layer'] in range(1,7) and x['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV'] for x in au['projection_shapes'])
 # Prefill mRoPE and three DeepStack features/masks remain identical at common positions.
 for x,y in zip(f['position_records'],a['position_records']):
  xp=x['cache_position'];yp=y['cache_position'];common=sorted(set(xp)&set(yp));xi=torch.tensor([xp.index(p) for p in common]);yi=torch.tensor([yp.index(p) for p in common]);assert torch.equal(x['cos'].index_select(1,xi),y['cos'].index_select(1,yi)) and torch.equal(x['sin'].index_select(1,xi),y['sin'].index_select(1,yi))
 for x,y in zip(f['deepstack'],a['deepstack']):assert torch.equal(x['features'],y['features']) and torch.equal(x['visual_mask'],y['visual_mask'])
 for shape in au['projection_shapes']:
  item=dict(shape);early=next((p for p in au.get('padding_records',[]) if p['step']==shape['step'] and p['layer']==shape['layer']),None);late=next((p for p in au.get('decode_temporary_zero_keys',[]) if p['step']==shape['step'] and p['layer']==shape['layer']),None)
  if shape['op'] in ['QK','AV']:
   item['key']=list(shape['key']);item['value']=list(shape['value']);count=(early or late or {}).get('actual_attention_key_rows',item['key'][-2]);item['key'][-2]=count;item['value'][-2]=count;item['flops']=2*item['query'][0]*item['query'][1]*item['query'][-2]*count*item['query'][-1]
  high=next((h for h in au.get('local_fp32',[]) if h['step']==shape['step'] and h['layer']==shape['layer'] and h['operator']==shape['op']),None)
  canonical.append(dict(run=run,name=name,condition=c,compute_dtype='float32 then BF16' if high else 'bfloat16 (Softmax accumulation FP32)',**{k:(str(v) if isinstance(v,list) else v) for k,v in item.items()}))
 physical.append(dict(run=run,name=name,condition=c,hidden_rows_by_layer=au['layers'],kv=au['kv'],engine_shape_records_before_zero_padding=au['projection_shapes'],canonical_actual_shapes_file='actual_matrix_shapes.csv',early_padding=au.get('padding_records',[]),decode_padding=au.get('decode_temporary_zero_keys',[]),local_fp32=au.get('local_fp32',[]),language_matrix_flops=au.get('actual_language_matrix_flops'),physical_fallback=False))
 if run=='phase2_rank0' and name=='decode-zero-key__B':extract('after_decode_F_B',f,a,['S1/L12/'+s for s in ['layer_input','q_proj','k_proj','v_proj','scores_after_mask','attention_probabilities','attention_AV_output','layer_output']])
 if run=='phase2_rank0' and name=='decode-zero-key__C':extract('remaining_L22_F_C',f,a,['S0/L22/'+s for s in ['layer_input','input_norm','q_proj','k_proj','v_proj','layer_output']])
 checks.append(dict(run=run,name=name,cache_original_positions=True,decode_append_prefix_unchanged=True,late_kv_matches_actual_projections=True,all_36_mlp_norm_residual=True,mrope_common_positions=True,deepstack_unchanged=True,real_exit_and_skip_preserved=True));del a;gc.collect()
np.savez_compressed(OUT/'arrays/first_difference_tensors.npz',**first_arrays);write(OUT/'arrays/first_difference_tensors.json',dict(sha256=sha(OUT/'arrays/first_difference_tensors.npz'),arrays=descs,encoding='BF16 bit patterns stored as int16; all other arrays native dtype'))
table('actual_matrix_shapes.csv',canonical)
write(OUT/'physical_execution_audit.json',dict(status='REAL_SKIP_AND_EXIT_PRESERVED_NUMERIC_GATE_FAILED',checks=checks,executions=physical,cache_snapshot_checks=cache_count,shape_record_note='Engine observers run before zero-column wrappers; actual_matrix_shapes.csv incorporates observed temporary early/Decode padding dimensions and corresponding QK/AV FLOPs. Original engine records retained separately.',decode_key_note='Transient 96 zero key columns restore full Key reduction dimension; stored KV remains27 rather than123; six Decode attention layers remain absent. This restores QK/AV work in active layers and must not be presented as free or as demonstrated acceleration.'))
write(OUT/'raw_tensor_index.json',dict(files=index,external_large_tensors_not_in_git=True,small_extracts_in_git=['arrays/effective_logits.npz','arrays/first_difference_tensors.npz']))
# Source, inherited results and raw files immutable; all new data parses.
requests=[]
for mf in [source1,source2]:
 for path,digest in mf['files'].items():requests.append((mf['commit']+':'+path,digest,path,mf is source2))
raw=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(r[0] for r in requests)+'\n').encode(),cwd=ROOT);pos=0
for request,digest,path,current in requests:
 end=raw.index(b'\n',pos);size=int(raw[pos:end].decode().split()[2]);data=raw[end+1:end+1+size];pos=end+2+size;assert __import__('hashlib').sha256(data).hexdigest()==digest
 if current:assert sha(ROOT/path)==digest
for item in read(PREVIOUS/'artifact_index.json')['files']:assert sha(PREVIOUS/item['path'])==item['sha256']
for fpath in OUT.rglob('*.json'):read(fpath)
assert subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
cleanup=read(OUT/'cpu_exception_cleanup.json');assert cleanup['normal_exit_restored'] and cleanup['injected_exception_exit_restored']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
for u in ['GPU-96bc853f-ceea-2404-c704-c48a3d1ca185','GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6']:assert u not in processes
write(OUT/'gpu_release_audit.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,processes=processes,both_assigned_devices_released=True,others_not_terminated=True))
times=[]
for run in ['phase1_rank0','phase2_rank0','phase2_rank1']:
 env=read(OUT/run/'environment.json');end=read(OUT/run/'completion.json');seconds=(datetime.datetime.fromisoformat(end['at'])-datetime.datetime.fromisoformat(env['at'])).total_seconds();times.append(dict(run=run,uuid=env['uuid'],start_UTC=env['at'],end_UTC=end['at'],worker_residency_seconds=seconds))
write(OUT/'gpu_time_audit.json',dict(runs=times,total_worker_residency_seconds=sum(r['worker_residency_seconds'] for r in times),not_kernel_or_e2e_benchmark=True))
write(OUT/'validation.json',dict(status='PARTIAL_REPAIR',decode_B_original_case_strict_on_two_gpus=True,L22_repair_passed=False,full_C_original_passed=False,all_effective_logits_recomputed=True,all_important_tensors_sha_verified=True,native_noop_hf_strict=True,probes_readonly=True,source_versions_git_verified=True,old_results_unchanged=True,cache_snapshots_checked=cache_count,physical_invariants=checks,cpu_cleanup=True,remaining_smokes_run=False,extra_input_count=0,accuracy_or_benchmark_run=False,main_unchanged=True,limitations=['One original input only','FP32 projections cannot be treated as matching frozen dense BF16','Temporary Decode key restoration adds QK/AV work despite compressed persistent KV','GPU single-call timings lack controlled benchmark repetition']))
print('CPU consolidation PASS',len(rows),'paired steps',cache_count,'cache snapshots','GPU seconds',sum(r['worker_residency_seconds'] for r in times),flush=True)
