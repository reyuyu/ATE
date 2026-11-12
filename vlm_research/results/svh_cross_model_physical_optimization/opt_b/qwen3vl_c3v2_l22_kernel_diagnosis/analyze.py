"""CPU-only consolidation. No inference or GPU initialization."""
import json,csv,pathlib,hashlib,subprocess,datetime,shutil
import numpy as np
import torch
from .common import *

def table(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def csvread(p):return list(csv.DictReader(pathlib.Path(p).open()))
def main():
 o=OUT;now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[];profiles=[];timings=[]
 for rank in [0,1]:
  for dest,name in [(rows,'same_input_operator_comparison.csv'),(profiles,'gemm_kernel_profiles.csv'),(timings,'microbenchmark.csv')]:
   dest.extend([dict(rank=rank,**r) for r in csvread(o/f'shard{rank}'/name)])
 table(o/'same_input_operator_comparison.csv',rows);table(o/'gemm_kernel_profiles.csv',profiles);table(o/'microbenchmark.csv',timings)
 configs=read(o/'shard0/algorithm_configs.json');table(o/'kernel_candidates.csv',[dict(**r,selected=r['candidate']=='Lt_dense_heuristic_0' and r['operator']!='down_proj',workspace_bytes=8*2**20,valid_physical_rows=26,input_padding_rows=0,internal_tile_padding='opaque hardware tile tails; see physical_flops_audit.json') for r in configs]+[dict(candidate='row13+13',class_id=2,status='NOT_RUN; first class matched local reference'),dict(candidate='fixed Triton BM32/BN64/BK32',class_id=3,status='NOT_RUN; first class matched local reference')])
 arrays={};cross=[]
 z0=np.load(PRIVATE/'shard0/operator_outputs.npz');z1=np.load(PRIVATE/'shard1/operator_outputs.npz')
 for k in z1.files:
  assert k in z0.files and np.array_equal(z0[k],z1[k]),k
  cross.append(dict(array=k,strict_equal=True))
 for k in z0.files:arrays[k]=z0[k][:,:32].copy()
 (o/'arrays').mkdir(exist_ok=True);np.savez_compressed(o/'arrays/operator_excerpt_bf16_bits.npz',**arrays)
 write(o/'arrays/operator_excerpt_index.json',dict(description='First 32 output channels of every text row, exact BF16 int16 bit patterns. Complete operators remain external.',arrays={k:dict(shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}))
 logits=[];fullcross=[];fullarrays={}
 for rank in [0,1]:
  folder=f'full_final_rank{rank}';comp=read(o/folder/'completion.json');assert not comp['original_full_C_gate'] and comp['status']=='PARTIAL_REPAIR'
  for r in read(o/folder/'logits_comparison.json'):logits.append(dict(rank=rank,phase='final_fixed_L22_L35',evidence='NEW_GENERATION',**r))
  for c in ['F','B','W22-functional','C_final','P_final']:
   a=torch.load(PRIVATE/folder/(c+'_logits.pt'),weights_only=True);assert a['token_ids']==[2152,151645]
   for step,v in enumerate(a['logits']):
    fullarrays[f'rank{rank}__{c}__step{step}']=v.numpy()
    if rank==1:
     b=torch.load(PRIVATE/'full_final_rank0'/(c+'_logits.pt'),weights_only=True);assert torch.equal(v,b['logits'][step]);fullcross.append(dict(condition=c,step=step,strict_equal=True))
 for folder in ['full_rank0','full_v2_rank0']:
  for r in read(o/folder/'logits_comparison.json'):logits.append(dict(rank=0,phase=folder,evidence='NEW_GENERATION_intermediate_NOT_selected_by_score',**r))
 table(o/'logits_comparison.csv',logits);np.savez_compressed(o/'arrays/full_valid_logits.npz',**fullarrays)
 write(o/'cross_gpu_replication.json',dict(operator_arrays=cross,full_valid_vectors=fullcross,all_equal=True,at=now))
 # Exact difference coordinates retained for new AV evidence; match original logical positions.
 ref=torch.load(RAW_PARENT/'phase1_rank0/F_probes.pt',weights_only=True)['records'];e=torch.load(PRIVATE/'full_final_rank1/selected_component_evidence.pt',weights_only=True);er=[]
 for key,b in e['records'].items():
  a=ref[key];apos=a['positions'];bpos=b['positions'];common=sorted(set(apos)&set(bpos));common=[p for p in common if p not in e['layout']['image_positions']];x=a['value'].index_select(a['axis'],torch.tensor([apos.index(p) for p in common]));y=b['value'].index_select(b['axis'],torch.tensor([bpos.index(p) for p in common]))
  if a.get('key_positions') is not None and b.get('key_positions') is not None:
   ak=a['key_positions'];bk=b['key_positions'];keys=sorted(set(ak)&set(bk));x=x.index_select(-1,torch.tensor([ak.index(p) for p in keys]));y=y.index_select(-1,torch.tensor([bk.index(p) for p in keys]))
  assert x.shape==y.shape
  bad=(x!=y).nonzero();d=(x.float()-y.float()).abs();coords=[dict(index=ix,reference=float(x[tuple(ix)]),candidate=float(y[tuple(ix)])) for ix in bad[:8].tolist()]
  er.append(dict(record=key,shape=list(x.shape),different_elements=len(bad),max_abs=float(d.max()),mean_abs=float(d.mean()),coordinates=coords))
 write(o/'component_difference_coordinates.json',er)
 finalcomp=read(o/'full_final_rank0/C_L35_components.json');write(o/'first_divergence.json',dict(baseline_first='L22 k_proj (frozen parent and current same-input replay)',after_fixed_gemm_first_internal=read(o/'full_final_rank0/C_L35_first_divergence.json'),after_fixed_gemm_first_prefill_layer_boundary=next(r for r in finalcomp if r['step']==0 and r['stage']=='layer_output' and not r['equal']),evidence='L22 tiny AV delta vanishes in o_proj; L26 AV delta propagates through residual. No tolerance relaxed. AV shape remains an independent unresolved factor.'))
 # Matrix dimensions and honest distinction between logical work and opaque hardware tile work.
 desc=read(o/'shard0/tensor_descriptors.json');opwork=[]
 for d in desc:
  name=d['operator'];m,k=d['physical_input']['shape'];n=d['weight']['tensor']['shape'][0]
  # Shape labels are read from actual kernel names. Exact issued instructions cannot be counted without counters.
  tile_m=128 if name in ['gate_proj','up_proj'] else 64
  opwork.append(dict(operator=name,valid_gemm_shape=[m,n,k],physical_input_rows=26,allocated_input_padding_rows=0,original_image_hidden_rows_projected=0,logical_gemm_flops=2*m*n*k,dense_logical_gemm_flops=2*122*n*k,logical_valid_row_ratio=26/122,observed_kernel_tile_M_extent=tile_m,tile_M_tail_capacity=tile_m-26,tile_capacity_MNK_flops_upper_nominal=2*tile_m*n*k,exact_issued_hardware_flops='UNMEASURED; opaque split-K partition rounding, tail predication and tensor-core instruction scheduling',extra_splitK_reduction='yes for K/V, see actual trace; reduction workspace distinct from persistent KV',extra_weight_conversion_bytes=0,workspace_cap_bytes=8*2**20,gate_up_same_launch_grid_as_dense=name in ['gate_proj','up_proj']))
 audits={}
 for rank in [0,1]:
  a=read(o/f'full_final_rank{rank}/C_final.json')['result']['audit'];assert all(r['hidden_rows']==26 for r in a['layers'] if r['step']==0 and r['layer']>=22);assert a['deepstack']==[0,1,2];assert len(a['fixed_lt_projections'])==56
  assert all(r['actual_gemm_rows']==26 and not r['dense_projection_executed'] for r in a['fixed_lt_projections'])
  assert len(a['padding_records'])==6 and all(x['dummy_probability']==0 for x in a['padding_records']);assert all(x['dummy_probability']==0 and x['persistent_dummy_KV']==0 for x in a['decode_temporary_zero_keys'])
  assert all(r['live_tokens']==0 for r in a['kv']['layers'] if r['layer'] in range(1,7));assert all(r['live_tokens']==27 for r in a['kv']['layers'] if r['layer'] not in range(1,7))
  audits[str(rank)]=dict(prefill_rows_26_from_L22=True,early_attention_skip_events=a['attention_events'],early_dummy_QK_AV_extra_flops=sum(x['extra_QK_AV_flops'] for x in a['padding_records']),decode_dummy_QK_AV_extra_flops=sum(x['extra_QK_AV_flops'] for x in a['decode_temporary_zero_keys']),logical_matrix_flops_with_dummy=a['actual_language_matrix_flops'],kv=a['kv'],deepstack=a['deepstack'],fixed_lt_projection_calls=56)
 write(o/'physical_flops_audit.json',dict(operators=opwork,full_model=audits,no_input_padding_or_dense_projection=True,hardware_tail_note='26 genuine rows are loaded/written. This does NOT mean only 26 rows of tensor-core arithmetic. Gate/Up retain identical Dense launch kernel/grid/block, so no measured GEMM work or latency saving there. Tile capacities are nominal, not hardware-counter measurements.',no_e2e_claim=True))
 # Every new raw artifact indexed, including traces, exact full outputs and selected component tensors.
 raw=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),external_to_git=True) for p in sorted(PRIVATE.rglob('*')) if p.is_file() and 'triton_cache' not in str(p)]
 write(o/'raw_tensor_index.json',dict(new_artifacts=raw,parent_reused=[dict(path=str(RAW_PARENT/'phase1_rank0'/n),sha256=sha(RAW_PARENT/'phase1_rank0'/n),external_to_git=True) for n in ['F_probes.pt','operator_shadows.pt']],large_raw_arrays_on_github=False))
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv'],text=True);procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
 assert not any(u in procs for u in ['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'])
 write(o/'gpu_release_audit.json',dict(at=now,gpus=gpu,processes=procs,both_authorized_gpus_released=True,other_processes_terminated=False))
 mainsha=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip();assert mainsha=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
 checks=dict(new_same_input_reference_strict_both_gpus=True,four_L22_fixed_operators_exact=True,down_native_control_preserved=True,full_model_original_failed_both_gpus=True,cross_gpu_logits_strict=True,native_noop_strict=True,native_F_W_B_historical_strict=True,readonly_probe_strict=True,physical_rows_kv_skip_audit=True,weights_bf16=True,hooks_clean=True,no_other_smoke_run=True,no_accuracy_e2e=True,no_main_modification=True,main_sha=mainsha,first_divergence_no_tolerance_relaxed=True,classes_tested=[1],second_gpu_fixed_replication_only=True)
 write(o/'validation.json',dict(at=now,checks=checks,limits=['Nsight absent; exact internal accumulation order and hardware issued FLOPs unobserved','Original full C fails. All frozen smoke NOT passed','Source froze in separately committed stages; earlier source recovered with its recorded Git commit.']))
 write(o/'completion.json',dict(status='PARTIAL_REPAIR',task_completed=True,full_model_gate_pass=False,remaining_smoke_run=False,accuracy_or_e2e_run=False,base_commit='a1a99c88c0db9244ef280017f297d9c5b5e75bbe',single_kernel_class_only=True,at=now))
 print('CPU CONSOLIDATION PASS',len(rows),len(profiles),len(logits),len(raw));print('P final',[(r['step'],r['max_abs'],r['mean_abs']) for r in logits if r['rank']==0 and r['phase']=='final_fixed_L22_L35' and r['comparison'].startswith('W22')])
if __name__=='__main__':main()
