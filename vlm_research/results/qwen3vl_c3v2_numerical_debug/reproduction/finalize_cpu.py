"""CPU-only evidence consolidation after original repair gate failed."""
import pathlib,json,csv,collections,datetime,subprocess,gc,os
import torch,numpy as np
from vlm_research.qwen3vl_c3v2_numerical_debug.common import *
from vlm_research.qwen3vl_c3v2_numerical_debug.probe import align,metrics,ORDER
torch.set_num_threads(1)
def table(name,rows,fields=None):
 with (OUT/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
def load(c,repair=False):
 return torch.load(PRIVATE/('repair_shard0' if repair else 'shard1')/f'{c}_prefill_probes.pt',weights_only=True)
layout=read(OUT/'sample_manifest.json')['input_bindings'][ORIGINAL];text=[i for i in range(layout['input_length']) if i not in layout['image_positions']]
cache=[];physical=[];archives={}
for c in CONDITIONS:
 a=load(c);archives[c]=a
 for key,snapshot in a['cache_snapshots'].items():
  i=snapshot['layer'];step=snapshot['step'];pair=(snapshot['keys'],snapshot['values']);skip=c in ['B','C'] and i in range(1,7)
  if skip:
   assert pair[0] is None and pair[1] is None and not snapshot['positions'];cache.append(dict(condition=c,layer=i,step=step,used=0,correct_original_positions=True,contents_match_native_projection=True,intentionally_absent=True,append_prefix_unchanged=True));continue
  expect=list(range(layout['input_length']+step)) if c in ['F','W22-functional'] else text+list(range(layout['input_length'],layout['input_length']+step))
  assert snapshot['positions']==expect
  exact=True;prefix=True
  if step==0:
   rk=a['records'][f'L{i:02d}/rope_k'];vp=a['records'][f'L{i:02d}/v_proj'];idx=torch.tensor([rk['positions'].index(p) for p in expect]);ki=rk['value'].index_select(2,idx)
   v=vp['value'].reshape(1,len(vp['positions']),8,128).transpose(1,2);vi=v.index_select(2,idx)
   exact=bool(torch.equal(ki,pair[0]) and torch.equal(vi,pair[1]));assert exact
  else:
   old=a['cache_snapshots'][f'0/{i}'];n=len(old['positions']);prefix=bool(torch.equal(old['keys'],pair[0][...,:n,:]) and torch.equal(old['values'],pair[1][...,:n,:]));assert prefix
  cache.append(dict(condition=c,layer=i,step=step,used=len(expect),correct_original_positions=True,contents_match_native_projection=exact,intentionally_absent=False,append_prefix_unchanged=prefix))
 r=read(OUT/f'shard1/baseline/{c}.json')['result'];au=r['audit'];physical.append(dict(version='original',condition=c,hidden_rows_by_layer=[x for x in au['layers'] if x['step']==0],kv=au['kv'],projection_shapes=au['projection_shapes'],physical_fallback=r['physical_fallback']))
write(OUT/'kv_cache_audit.json',dict(status='PASS_ON_FIXED_ORIGINAL_INPUT',rows=cache,all_prefill_stored_keys_values_equal_recorded_projection_gather=True,all_decode_prefixes_unchanged=True,all_original_positions_correct=True,note='These checks rule out observed cache content/index corruption. Shape-sensitive floating-point reductions after compacting legal keys remain numerical effects, not excluded by semantic equality.'))
repaired=load('C',True);left=archives['F'];repair_rows=[];dummy_alignment=[]
# Raw probe length mapping treated the appended zero column like original P-1.
# It is not an original key. Preserve raw archives and align that column under
# a distinct dummy logical ID, so an intentionally unused score cannot be
# mistaken for a numerical or semantic divergence of an effective connection.
for layer in range(1,7):
 for stage in ['scores_before_mask','effective_mask','scores_after_mask','attention_probabilities']:
  key=f'L{layer:02d}/{stage}';record=repaired['records'][key]
  assert record['key_positions']==list(range(layout['input_length']))
  record['key_positions']=list(range(layout['input_length']-1))+[-1]
  dummy_alignment.append(dict(record=key,dummy_column=layout['input_length']-1,raw_length_based_label=layout['input_length']-1,correct_semantic_label='DUMMY_ZERO_KEY (-1); never original P-1 KV',reason='RepairExecution padding_records and masked probability0 verify this extra column has no original key'))
write(OUT/'repair_dummy_key_alignment.json',dict(raw_archives_unchanged=True,analysis_only_alignment=dummy_alignment))
for layer in range(38):
 for stage in ORDER:
  key=f'L{layer:02d}/{stage}'
  if key not in left['records'] or key not in repaired['records']:continue
  x=left['records'][key];y=repaired['records'][key];x['retained_positions']=text
  for scope in ['text','p_minus_1','all_common']:
   u,v,positions=align(x,y,scope)
   if u is None:continue
   repair_rows.append(dict(comparison='F__C-repaired',layer=layer,stage=stage,scope=scope,common_queries=len(positions),reference_physical_shape=str(x['shape']),candidate_physical_shape=str(y['shape']),**metrics(u,v)))
table('repair_component_differences.csv',repair_rows)
table('repair_layerwise_hidden_differences.csv',[r for r in repair_rows if r['stage'] in ['layer_input','attention_residual','layer_output','final_norm','lm_head_input']])
first=next(r for r in repair_rows if r['scope']=='text' and not r['equal']);write(OUT/'repair_first_divergence.json',first)
before=read(OUT/'shard1/factor_isolation.json');after=read(OUT/'repair_shard0/repair_validation.json');repair_compare=[]
for x in before:
 if (x['reference'],x['candidate']) not in [('F','B'),('F','C')]:continue
 y=next(r for r in after['comparisons'] if r['reference']==x['reference'] and r['candidate']==x['candidate'] and r['step']==x['step'])
 repair_compare.append(dict(comparison=x['reference']+'__'+x['candidate'],step=x['step'],before_max_abs=x['max_abs'],after_max_abs=y['max_abs'],before_mean_abs=x['mean_abs'],after_mean_abs=y['mean_abs'],before_rmse=x['rmse'],after_rmse=y['rmse'],top1_equal=y['top1_equal'],complete_tokens_equal=y['complete_token_equal'],after_original_gate_pass=bool(y['max_abs']<=.5 and y['mean_abs']<=.02 and y['top1_equal'] and y['complete_token_equal'])))
table('repair_comparison.csv',repair_compare)
for c in ['B','C']:
 r=read(OUT/f'repair_shard0/{c}.json')['result'];a=r['audit'];padding=a['padding_records'];assert len(padding)==6
 assert all(p['computed_query_rows']==121 and p['computed_KV_projection_rows']==121 and p['actual_attention_key_rows']==122 and p['persistent_dummy_KV_rows']==0 and p['dummy_probability']==0 for p in padding)
 assert all(a['kv']['layers'][i]['capacity_bytes']==0 for i in range(1,7))
 assert all(not(x['layer'] in range(1,7) and x['step']>0 and x['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV']) for x in a['projection_shapes'])
 physical.append(dict(version='minimal_repair',condition=c,hidden_rows_by_layer=[x for x in a['layers'] if x['step']==0],kv=a['kv'],projection_shapes=a['projection_shapes'],physical_fallback=r['physical_fallback'],padding=padding,actual_language_matrix_flops=a['actual_language_matrix_flops']))
write(OUT/'physical_execution_audit.json',dict(status='PHYSICAL_PATH_OBSERVED_NUMERICAL_ACCEPTANCE_FAILED',executions=physical,flops_scope='Language GEMMs + QK/AV only; repaired estimate adds actual one-zero-key overhead. No latency or full-model speed/FLOPs claim.'))
# Persist compact first-divergence tensors in Git for genuine offline checking.
arrays={};descriptors=[]
for label,a,b,layer,stages in [
 ('L1_F_B',archives['F'],archives['B'],1,['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k','scores_before_mask','effective_mask','scores_after_mask','attention_probabilities','attention_AV_output','o_proj_raw','attention_effective_output','layer_output']),
 ('L22_W_P',archives['W22-functional'],archives['P'],22,['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k','mlp_output','layer_output']),
 ('repaired_L22_F_C',archives['F'],repaired,22,['layer_input','input_norm','q_proj','k_proj','v_proj','layer_output'])]:
 for stage in stages:
  x=a['records'][f'L{layer:02d}/{stage}'];y=b['records'][f'L{layer:02d}/{stage}'];x['retained_positions']=text;u,v,pos=align(x,y,'text')
  for side,t in [('reference',u),('candidate',v)]:
   name=label+'__'+stage+'__'+side;t=t.contiguous();native=t.dtype
   arrays[name]=t.view(torch.int16).numpy() if native==torch.bfloat16 else t.numpy()
   descriptors.append(dict(key=name,dtype=str(native),stored_dtype=str(arrays[name].dtype),shape=list(t.shape),positions=pos,encoding='BF16 bits reinterpreted int16' if native==torch.bfloat16 else 'native numpy'))
small=OUT/'arrays/first_divergence_evidence.npz';small.parent.mkdir(exist_ok=True);np.savez_compressed(small,**arrays)
write(OUT/'arrays/first_divergence_evidence.json',dict(path=str(small.relative_to(OUT)),sha256=sha(small),arrays=descriptors,reconstruct_BF16='torch.from_numpy(array.copy()).view(torch.bfloat16); shape is preserved',note='Aligned retained text rows; at L1 P-1 is intentionally absent from physical raw attention. Raw attention heads are all32 at L1/L22, not only a selected subset.'))
vectors={};logdesc=[]
for folder,conditions in [('shard1',['N','Noop']+CONDITIONS),('repair_shard0',['N','Noop','F','B','C'])]:
 for c in conditions:
  x=torch.load(PRIVATE/folder/f'{c}_logits.pt',weights_only=True);name=folder+'__'+c;vectors[name]=torch.stack(x['logits']).numpy();logdesc.append(dict(key=name,token_ids=x['token_ids'],shape=list(vectors[name].shape),dtype='float32 from original BF16 effective logits'))
np.savez_compressed(OUT/'arrays/effective_logits.npz',**vectors);write(OUT/'arrays/effective_logits.json',dict(sha256=sha(OUT/'arrays/effective_logits.npz'),arrays=logdesc))
# Locate changed coordinates using the exact first divergent AV tensor.
x=archives['F']['records']['L01/attention_AV_output'];y=archives['B']['records']['L01/attention_AV_output'];x['retained_positions']=text;u,v,pos=align(x,y,'text');coords=(u!=v).nonzero().tolist()
write(OUT/'first_difference_coordinates.json',dict(layer=1,stage='attention_AV_output',retained_text_difference_count=len(coords),coordinates=[dict(original_query_position=pos[c[1]],head=c[2],channel=c[3],reference=float(u[tuple(c)]),candidate=float(v[tuple(c)])) for c in coords[:32]],repair_first_divergence=first))
write(OUT/'numerical_baseline_reproduction.json',dict(status='EXACT_OLD_FAILURE_REPRODUCED_BOTH_GPUS',shards={str(i):read(OUT/f'shard{i}/numerical_baseline_reproduction.json') for i in range(2)},old_failure_preserved=True))
write(OUT/'gpu_smoke_validation.json',dict(status='ROOT_CAUSE_IDENTIFIED_REPAIR_FAILED',original_smoke_all_passed=False,fixed_smoke_cases=read(OUT/'protocol.json')['smoke_cases'],cases_run=[ORIGINAL],remaining_smokes_run=False,reason='Repaired original C still fails unchanged mean absolute error gate on steps0 and1; no other sample may be added.',acceptance=read(OUT/'protocol.json')['tolerances'],repair_validation=after,source_commit=read(OUT/'source_manifest.json')['commit'],no_quality_or_e2e_benchmark=True))
all_indices=[]
for folder in ['shard0','shard1','repair_shard0']:
 for item in read(OUT/folder/'tensor_index.json'):
  assert sha(item['path'])==item['sha256'];all_indices.append(dict(run=folder,**item))
write(OUT/'raw_tensor_index.json',dict(files=all_indices,external_arrays_not_in_git=True,small_offline_extracts_in_git=['arrays/first_divergence_evidence.npz','arrays/effective_logits.npz']))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
assert 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185' not in procs and 'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6' not in procs
write(OUT/'gpu_release_audit.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,processes=procs,both_assigned_devices_released=True,other_processes_killed=False,no_gpu_work_after_failed_repair=True))
timings=[]
for folder,startfile,endfile in [('shard0','shard0/environment.json','shard0/diagnostic_completion.json'),('shard1','shard1/environment.json','shard1/diagnostic_completion.json'),('repair_shard0','gpu_preflight_repair.json','repair_shard0/repair_validation.json')]:
 a=read(OUT/startfile)['at'];b=read(OUT/endfile)['at'];seconds=(datetime.datetime.fromisoformat(b)-datetime.datetime.fromisoformat(a)).total_seconds();timings.append(dict(run=folder,start_UTC=a,end_UTC=b,worker_residency_wall_seconds=seconds))
write(OUT/'gpu_time_audit.json',dict(runs=timings,total_worker_residency_seconds=sum(x['worker_residency_wall_seconds'] for x in timings),measurement='Wall time of model worker including loading, CPU probe copies and file writes. Not GPU kernel time, formal latency or E2E benchmark.'))
print('CPU evidence finalized. First repaired divergence:',first,'KV checks',len(cache),'small evidence bytes',small.stat().st_size,flush=True)
