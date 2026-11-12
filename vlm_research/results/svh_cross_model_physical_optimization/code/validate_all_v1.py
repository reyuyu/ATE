from pathlib import Path
import json,hashlib,collections,subprocess,sys,csv
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert read(A/'coordinator_status.json')['status']=='COMPLETE_ALL_PHASES'
assert read(A/'refinement_coordinator_status.json')['status']=='COMPLETE'
for script in ['analyze_results_v1.py','analyze_profiles_v1.py','analyze_monitors_v1.py']:
 subprocess.run([sys.executable,'-B',str(A/'code'/script)],check=True)
checks=[]
for name in ['frozen_source_manifest.json','additional_frozen_sources.json']:
 items=read(A/name);fail=[]
 for x in items:
  p=Path(x['path']);snap=A/'frozen_sources'/str(p).lstrip('/')
  if not p.exists() or sha(p)!=x['sha256']:fail.append(dict(path=str(p),problem='ORIGINAL_CHANGED_OR_MISSING'))
  if not snap.exists() or sha(snap)!=x['sha256']:fail.append(dict(path=str(snap),problem='SNAPSHOT_MISSING_OR_CHANGED'))
 assert not fail,(name,fail[:8])
 checks.append(dict(check=name,status='PASS',count=len(items)))
phase_counts=[];total_expected=0;numeric_counts=[];audit_checks=[]
for kind in ['Qwen','InternVL','LLaVA']:
 G=A/kind/'screen';screen=read(G/'completion.json');assert screen['status']=='COMPLETE_SCREEN'
 assert read(G/'native_noop.json')['status']=='PASS'
 selected=read(A/kind/'selected_stages.json');eligible=read(G/'candidate_eligibility.json')['eligible']
 numeric=read(G/'numerical_validation.json')
 for method,entry in selected.items():
  for stage,ok in eligible[method].items():
   if not ok:continue
   rows=[x for x in numeric if x['method']==method and x['stage']==stage]
   assert len(rows)==30 and len({x['key'] for x in rows})==30 and all(x['status']=='PASS' for x in rows)
   if stage in ['B','C']:assert all(x['logits_torch_equal'] and x['tokens_exact'] and x['hidden_relative_L2_max']==0 for x in rows)
   numeric_counts.append(dict(kind=kind,method=method,stage=stage,n=30,status='PASS'))
  if entry['stage']:assert eligible[method][entry['stage']]
 audit=read(A/kind/'audit/audit_mode_validation.json');assert audit['status']=='PASS';audit_checks.append(dict(kind=kind,**audit))
 n=sum(bool(v['stage']) for v in selected.values())
 expected=30*(1+2*n) if n else 0
 for phase in ['session1','session2']:
  if not expected:continue
  done=read(A/kind/phase/'completion.json')
  assert done['status'].startswith('COMPLETE') and done['formal_completed']==expected==done['formal_expected']
  total_expected+=expected;phase_counts.append(dict(kind=kind,phase=phase,formal=expected,status=done['status']))
 assert read(A/kind/'profile/completion.json')['status']=='COMPLETE_PROFILE'
for kind in ['Qwen','InternVL']:
 root=A/kind;g=root/'refine_screen';assert read(g/'completion.json')['status']=='COMPLETE_SCREEN'
 assert read(g/'native_noop.json')['status']=='PASS'
 eligible=read(g/'candidate_eligibility.json')['eligible']['SVH']['E']
 if eligible:
  rows=[x for x in read(g/'numerical_validation.json') if x['stage']=='E']
  assert len(rows)==30 and all(x['status']=='PASS' and x['logits_torch_equal'] and x['tokens_exact'] and x['hidden_relative_L2_max']==0 for x in rows)
  numeric_counts.append(dict(kind=kind,method='SVH',stage='E',n=30,status='PASS'))
 audit=read(root/'refine_audit/audit_mode_validation.json');assert audit['status']=='PASS';audit_checks.append(dict(kind=kind,campaign='refinement_E',**audit))
 entry=read(root/'refine_selected_stages.json')['SVH']
 if entry['stage']:
  assert eligible
  for phase in ['refine_session1','refine_session2']:
   done=read(root/phase/'completion.json');assert done['status'].startswith('COMPLETE') and done['formal_completed']==90==done['formal_expected']
   total_expected+=90;phase_counts.append(dict(kind=kind,phase=phase,formal=90,status=done['status']))
 assert read(root/'refine_profile/completion.json')['status']=='COMPLETE_PROFILE'

for row in read(A/'runtime_dependency_manifest.json'):
 assert sha(row['path'])==row['sha256'] and sha(A/row['snapshot'])==row['sha256']
 if row.get('actual_dynamic_cache'):assert sha(row['actual_dynamic_cache'])==row['cache_sha256']==row['sha256']
for row in read(A/'frozen_reference_evidence_manifest.json'):
 assert sha(row['original'])==row['sha256'] and sha(A/row['snapshot'])==row['sha256']
checks.append(dict(check='Runtime_and_model_sources_and_reference_evidence',status='PASS'))
checks.append(dict(check='Native_NoOp_all_backbones',status='PASS',backbones=3))
checks.append(dict(check='All30_full32_logits_tokens',status='PASS',conditions=numeric_counts))
checks.append(dict(check='Actual_Audit_Mode',status='PASS',results=audit_checks))
# Actual C-buffer lifetime: no reallocation at0/1/8/31, incrementally changing valid lengths only.
lifecycle=[]
for kind in ['Qwen','InternVL']:
 snaps=read(A/kind/'screen/physical_audit.json');groups=collections.defaultdict(list)
 eligibility=read(A/kind/'screen/candidate_eligibility.json')['eligible']
 for r in snaps:
  if r['stage']!='C' or not eligibility[r['method']].get('C',False):continue
  for l in r['layers']:
   if l['present']:groups[r['key'],r['method'],l['layer']].append((r['step'],l))
 for (key,method,layer),rs in groups.items():
  rs=sorted(rs);assert [r[0] for r in rs]==[0,1,8,31]
  assert len({tuple(r[1]['storage_pointers']) for r in rs})==1
  assert len({tuple(r[1]['storage_bytes']) for r in rs})==1
  assert all(l['used']==rs[0][1]['used']+s for s,l in rs)
  lifecycle.append(dict(kind=kind,key=key,method=method,layer=layer,steps=[0,1,8,31],stable_storage=True,correct_used_growth=True))
for kind in ['Qwen','InternVL']:
 g=A/kind/'refine_screen'
 if not read(g/'candidate_eligibility.json')['eligible']['SVH']['E']:continue
 groups=collections.defaultdict(list)
 for r in read(g/'physical_audit.json'):
  if r['stage']!='E':continue
  for l in r['layers']:
   if l['present']:groups[r['key'],r['method'],l['layer']].append((r['step'],l))
 for (key,method,layer),rs in groups.items():
  rs=sorted(rs);assert [r[0] for r in rs]==[0,1,8,31]
  assert len({tuple(r[1]['storage_pointers']) for r in rs})==1
  assert len({tuple(r[1]['storage_bytes']) for r in rs})==1
  assert all(l['used']==rs[0][1]['used']+step for step,l in rs)
  lifecycle.append(dict(kind=kind,campaign='refinement_E',key=key,method=method,layer=layer,steps=[0,1,8,31],stable_storage=True,correct_used_growth=True))

save(A/'kv_lifecycle_validation.json',dict(status='PASS',layer_request_cases=len(lifecycle),cases=lifecycle))
checks.append(dict(check='C_actual_KV_growth_pointer_storage',status='PASS',cases=len(lifecycle)))
incremental=read(A/'validation_incremental.json');assert incremental['formal_requests']==total_expected
assert read(A/'process_conflict_validation.json')['status']=='PASS'
# Verify no lingering compute process; do not touch any process.
pidlist=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,power.limit','--format=csv,noheader'],text=True).strip()
release=dict(status='RELEASED' if not pidlist else 'OTHER_COMPUTE_PRESENT_NOT_TOUCHED',compute_processes=pidlist,gpu=gpu,our_inference_tasks_complete=True,server_not_stopped=True)
save(A/'resource_release.json',release);assert not pidlist,release
validation=dict(status='PASS_WITH_REJECTED_CANDIDATES',scope='Completed all allowed optimization stages; failed numerical stages excluded, negatives preserved',formal_requests=total_expected,phases=phase_counts,checks=checks,independent_cpu_recomputation=True,
 input_and_KV_and_checkpoint_checks=incremental,actual_GPU_UUIDs=incremental['GPU_UUIDs'],no_quality_benchmark=True,no_git_writes=True,no_original_source_or_history_overwritten=True,
 quality_limitation='B/C matched original30 full32 logits and output regression; no blanket full-Benchmark Accuracy claim; failed D never inherits historical quality',
 Audit_order_deviation='Qwen actual Audit Mode followed Session1, after full30 strict numerical screen and before Session2; disclosed, all passed, no selective retry',
 GPU_release=release)
save(A/'validation.json',validation)
subprocess.run([sys.executable,'-B',str(A/'code/build_reports_v1.py')],check=True)
with (A/'final_report_zh.md').open('a') as f:f.write('\n最终独立验收：PASS_WITH_REJECTED_CANDIDATES；正式请求 '+str(total_expected)+' 个。GPU 推理进程已退出，资源释放检查通过。失败数值候选未进入正式表。持久盘文件完整性以结果目录外的 svh_cross_model_physical_optimization_delivery.json 回执为准。\n')
print(json.dumps(dict(status=validation['status'],formal_requests=total_expected,GPU_release=release['status'])))
