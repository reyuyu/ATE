"""One frozen L22 operator task. No model generation or dataset evaluation."""
import os,sys,time,gc,csv,datetime,traceback,pathlib,statistics,json,hashlib
import torch,numpy as np
from safetensors import safe_open
from torch.profiler import profile,ProfilerActivity,record_function
from ..qwen3vl_c3v2_numerical_debug.probe import metrics
from .common import *
from .candidates import LtPlan,row13,fixed_triton

def table(path,rows):
 path=pathlib.Path(path);keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def tensor_sha(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def desc(t):return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),contiguous=t.is_contiguous(),storage_offset=t.storage_offset(),sha256=tensor_sha(t))
def details(ref,out):
 m=metrics(ref,out);ix=(ref!=out).nonzero();m['different_elements']=len(ix);m['different_fraction']=len(ix)/ref.numel();m['coordinates']=[dict(index=r,reference=float(ref[tuple(r)]),candidate=float(out[tuple(r)])) for r in ix[:8].tolist()];return m

def profiled(fn,label,run,profiles,performance):
 check()
 for _ in range(5):a=fn()
 torch.cuda.synchronize();before=a.clone();trace=PRIVATE/run/(label+'.json');trace.parent.mkdir(parents=True,exist_ok=True)
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as p:
  with record_function(label):a=fn();torch.cuda.synchronize()
 p.export_chrome_trace(str(trace));events=read(trace).get('traceEvents',[]);kernels=[e for e in events if e.get('cat')=='kernel']
 for i,e in enumerate(kernels):
  args=e.get('args',{});profiles.append(dict(label=label,event=i,kernel_name=e['name'],duration_us=e.get('dur'),grid=str(args.get('grid')),block=str(args.get('block')),registers_per_thread=args.get('registers per thread'),shared_memory=args.get('shared memory'),stream=args.get('stream'),args_json=json.dumps(args,sort_keys=True),trace_path=str(trace),trace_sha256=sha(trace),actual_profiler_event=True))
 if not kernels:profiles.append(dict(label=label,event=-1,kernel_name='UNAVAILABLE: CUDA kernel events absent',actual_profiler_event=False,trace_path=str(trace),trace_sha256=sha(trace)))
 assert torch.equal(before,a),'Profiler changed operator output'
 events=[]
 for _ in range(20):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();a=fn();end.record();events.append((start,end))
 torch.cuda.synchronize();times=[s.elapsed_time(e) for s,e in events];assert torch.equal(before,a),'Repeated operator is not deterministic'
 performance.append(dict(label=label,warmup=5,repeats=20,gpu_event_ms_median=statistics.median(times),gpu_event_ms_min=min(times),gpu_event_ms_max=max(times),measurement='Local GPU event only; allocations/host overhead not complete E2E',profiler_output_strict=True,repeated_output_strict=True))
 return a

def main(rank=0,verify_only=False):
 torch.set_num_threads(1);check();run=f'shard{rank}';(PRIVATE/run).mkdir(parents=True,exist_ok=True);(OUT/run).mkdir(parents=True,exist_ok=True);source=read(OUT/'source_manifest.json')
 for path,h in source['files'].items():assert sha(ROOT/path)==h,path
 bridge=read(OUT/'bridge_build.json');assert sha(PRIVATE/'build/liblt_bridge.so')==bridge['library_sha256']
 write(OUT/run/'environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,triton=__import__('triton').__version__,source_commit=source['commit'],device_name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,allow_tf32=torch.backends.cuda.matmul.allow_tf32))
 fixed_policy=read(OUT/'shard0/candidate_policy.json') if verify_only else None
 raw_input=RAW_PARENT/'phase1_rank0/F_probes.pt';bound=next(r for r in read(PARENT/'raw_tensor_index.json')['files'] if r['path']==str(raw_input));assert sha(raw_input)==bound['sha256'];old=torch.load(raw_input,map_location='cpu',weights_only=True);records=old['records'];layout=old['layout'];text=[i for i in range(122) if i not in layout['image_positions']];assert len(text)==26
 ix=torch.tensor(text,device='cuda:0');weightmap=read(MODEL/'model.safetensors.index.json')['weight_map'];weights={};weight_meta={}
 for name in NAMES:
  sub='self_attn' if name in ['k_proj','v_proj'] else 'mlp';key=f'model.language_model.layers.22.{sub}.{name}.weight';path=MODEL/weightmap[key]
  with safe_open(path,framework='pt',device='cpu') as sf:w=sf.get_tensor(key)
  assert w.dtype==torch.bfloat16;weight_meta[name]=dict(key=key,source=str(path),tensor=desc(w));weights[name]=w.cuda()
 previous=torch.load(pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_repair2/phase1_rank0/operator_shadows.pt'),map_location='cpu',weights_only=True)
 workspace=torch.empty(8*1024*1024,device='cuda:0',dtype=torch.uint8)
 # Actual wrapper/custom-kernel correctness test on exactly representable integers.
 with torch.inference_mode():
  testx=(torch.arange(26*32,device='cuda:0').reshape(26,32)%7-3).bfloat16();testw=(torch.arange(64*32,device='cuda:0').reshape(64,32)%5-2).bfloat16();exact=torch.nn.functional.linear(testx.float(),testw.float()).bfloat16();tp=LtPlan(26,64,32,workspace)
  assert tp.count()>0;assert torch.equal(tp.run(testx,testw),exact);tp.close()
 smoke=dict(Lt_pointer_and_layout_correct=True,exact_integer_reference=True,no_weight_or_prompt_changes=True)
 profiles=[];performance=[];comparison=[];configs=[];descriptors=[];arrays={};asmindex=[]
 with torch.inference_mode():
  full_norm=records['S0/L22/input_norm']['value'].to('cuda:0').reshape(122,-1).contiguous();full_post=records['S0/L22/post_attention_norm']['value'].to('cuda:0').reshape(122,-1).contiguous()
  full_down=torch.nn.functional.silu(torch.nn.functional.linear(full_post,weights['gate_proj']))*torch.nn.functional.linear(full_post,weights['up_proj'])
  for name in NAMES:
   x=full_norm if name in ['k_proj','v_proj'] else full_down if name=='down_proj' else full_post;w=weights[name];small=x.index_select(0,ix);m,k=small.shape;n=w.shape[0]
   native_full=torch.nn.functional.linear(x,w);reference=native_full.index_select(0,ix);saved=previous[f'0/22/{name}/reference'].reshape_as(reference);assert torch.equal(reference.cpu(),saved),'Frozen dense reference did not reproduce: '+name
   descriptors.append(dict(operator=name,full_input=desc(x),physical_input=desc(small),weight=weight_meta[name],full_output=desc(native_full),gathered_reference=desc(reference),image_rows_computed_only_in_diagnostic_reference=96))
   native=profiled(lambda:torch.nn.functional.linear(small,w),name+'__torch_physical26',run,profiles,performance);dense=profiled(lambda:torch.nn.functional.linear(x,w),name+'__torch_dense122',run,profiles,performance);assert torch.equal(dense.index_select(0,ix),reference)
   base=details(reference,native);comparison.append(dict(operator=name,candidate='torch_physical26',class_id=0,physical_rows=26,padded_rows=0,matrix_flops=2*m*n*k,temporary_output_bytes=m*n*2,weight_conversion_bytes=0,**base));arrays[name+'__reference']=reference.cpu().view(torch.int16).numpy();arrays[name+'__torch_physical26']=native.cpu().view(torch.int16).numpy()
   pd=LtPlan(122,n,k,workspace);ps=LtPlan(26,n,k,workspace)
   fixed=fixed_policy['selected'][name] if verify_only else None
   choices=([int(fixed.rsplit('_',1)[1])] if fixed.startswith('Lt_dense_heuristic_') else []) if verify_only else range(min(2,pd.count()))
   for idx in choices:
    code=ps.select(pd,idx);dcode=pd.select(pd,idx);cfg=dict(operator=name,candidate=f'Lt_dense_heuristic_{idx}',class_id=1,physical_compatibility_status=code,dense_compatibility_status=dcode,config=ps.config() if code==0 else None,workspace_limit=workspace.numel(),library_version=LtPlan.lib.lt_version());configs.append(cfg)
    if code or dcode:continue
    label=f'{name}__Lt_fixed_{idx}';y=profiled(lambda:ps.run(small,w),label+'__physical26',run,profiles,performance);yd=profiled(lambda:pd.run(x,w),label+'__dense122_shadow',run,profiles,performance);cfg['dense_shadow_vs_frozen']=details(reference,yd.index_select(0,ix))
    comparison.append(dict(operator=name,candidate=f'Lt_dense_heuristic_{idx}',class_id=1,physical_rows=26,padded_rows=0,matrix_flops=2*m*n*k,temporary_output_bytes=m*n*2+workspace.numel(),weight_conversion_bytes=0,**details(reference,y)));arrays[name+f'__Lt_dense_heuristic_{idx}']=y.cpu().view(torch.int16).numpy()
   ps.close();pd.close()
   sufficient=lambda:any(r['operator']==name and r['class_id']>0 and r['mean_abs']<=.5*base['mean_abs'] and r['max_abs']<=base['max_abs'] for r in comparison)
   if (verify_only and fixed=='row13') or (not verify_only and not sufficient()):
    if 'row13_correct' not in smoke:assert torch.equal(row13(testx,testw),exact);smoke['row13_correct']=True
    y=profiled(lambda:row13(small,w),name+'__row13',run,profiles,performance);comparison.append(dict(operator=name,candidate='row13',class_id=2,physical_rows=26,padded_rows=0,row_chunks='13+13',matrix_flops=2*m*n*k,temporary_output_bytes=2*m*n*2,weight_conversion_bytes=0,**details(reference,y)));arrays[name+'__row13']=y.cpu().view(torch.int16).numpy()
   if (verify_only and fixed=='Triton_32x64x32') or (not verify_only and not sufficient()):
    if 'Triton_mask_and_index_correct' not in smoke:ty,_=fixed_triton(testx,testw);assert torch.equal(ty,exact);smoke['Triton_mask_and_index_correct']=True
    y,compiled=fixed_triton(small,w);y=profiled(lambda:fixed_triton(small,w)[0],name+'__Triton_32x64x32',run,profiles,performance);comparison.append(dict(operator=name,candidate='Triton_32x64x32',class_id=3,physical_rows=26,padded_rows=6,tile_M=32,tile_N=64,tile_K=32,num_warps=4,num_stages=3,matrix_flops=2*32*__import__('math').ceil(n/64)*64*k,valid_math_flops=2*m*n*k,temporary_output_bytes=m*n*2,weight_conversion_bytes=0,**details(reference,y)));arrays[name+'__Triton_32x64x32']=y.cpu().view(torch.int16).numpy()
    for ext in ['ptx','ttir','ttgir']:
     if ext in compiled.asm:
      path=PRIVATE/run/f'{name}_triton.{ext}';path.write_text(compiled.asm[ext]);asmindex.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
   # The only allowed final policy: choose per-op lowest mean, then max, class/order.
   print('OPERATOR',name,[(r['candidate'],r['mean_abs'],r['different_elements']) for r in comparison if r['operator']==name],flush=True)
 torch.cuda.synchronize();selected={};bases={};bestrows={}
 for name in NAMES[:4]:
  rs=[r for r in comparison if r['operator']==name];base=next(r for r in rs if r['candidate']=='torch_physical26');best=min([r for r in rs if r['max_abs']<=base['max_abs']],key=lambda r:(r['mean_abs'],r['max_abs'],r['class_id'],r['candidate']));selected[name]=best['candidate'];bases[name]=base;bestrows[name]=best
 selected['down_proj']='torch_physical26'
 local=all(bestrows[name]['mean_abs']<=bases[name]['mean_abs'] for name in NAMES[:4]) and (any(bestrows[name]['mean_abs']<=.5*bases[name]['mean_abs'] for name in ['k_proj','v_proj']) or sum(r['mean_abs'] for r in bestrows.values())<=.75*sum(r['mean_abs'] for r in bases.values()))
 policy=dict(selected=selected,local_gate_pass=bool(local),rule='All four means/max no worse; at least one K/V mean50% lower OR summed four mean25% lower. Select per-op lowest mean among no-worse-max candidates, then max/class/order. Down stays native. Engineering triage only; full logits thresholds unchanged. Never select by answer.',full_model_allowed=bool(local))
 table(OUT/run/'gemm_kernel_profiles.csv',profiles);table(OUT/run/'same_input_operator_comparison.csv',[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in comparison]);table(OUT/run/'microbenchmark.csv',performance);write(OUT/run/'algorithm_configs.json',configs);write(OUT/run/'tensor_descriptors.json',descriptors);write(OUT/run/'candidate_policy.json',policy);write(OUT/run/'triton_assembly_index.json',asmindex)
 np.savez_compressed(PRIVATE/run/'operator_outputs.npz',**arrays);write(OUT/run/'raw_arrays_index.json',dict(path=str(PRIVATE/run/'operator_outputs.npz'),sha256=sha(PRIVATE/run/'operator_outputs.npz'),bytes=(PRIVATE/run/'operator_outputs.npz').stat().st_size,encoding='Exact BF16 bits as int16',arrays=[dict(key=k,shape=list(a.shape),stored_dtype=str(a.dtype),original_dtype='torch.bfloat16',sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in arrays.items()]))
 write(OUT/run/'kernel_smoke.json',smoke)
 write(OUT/run/'completion.json',dict(status='LOCAL_CANDIDATE_FOUND' if local else 'STRICT_BF16_REFERENCE_MATCH_NOT_ACHIEVED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_frozen_dense_operators_reproduced=True,classes_evaluated=sorted(set(r['class_id'] for r in comparison if r['class_id']>0)),later_classes_skipped_when_earlier_local_gate_passed=True,verification_only=verify_only,source_commit=source['commit'],policy=policy,profiler_kernel_events=sum(bool(r['actual_profiler_event']) for r in profiles),no_model_generation=True))
 print('POLICY',policy,flush=True);del weights;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main(int(sys.argv[1]) if len(sys.argv)>1 else 0,len(sys.argv)>2)
 except BaseException as e:write(OUT/'worker_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),at=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
