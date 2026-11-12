"""CPU preflight; bind the original frozen implementation without rewriting it."""
from pathlib import Path
import os,json,gzip,hashlib,subprocess,shutil,datetime
R=Path('/home/featurize/work/ATE');U=R/'vlm_research/results/unified_e2e32_4090';D=U.parent/'unified_e2e32_eight_methods_3090_session2';L=Path('/home/featurize/ate-local-experiments/unified_e2e32_eight_methods_3090_session2');S=U.parent/'featurize_23386_environment_restore_20261110';M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf');UUID='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def save(root,name,v):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def prepare():
 assert not (D/'protocol.json').exists() and not L.exists(),'Existing attempt must be preserved; no implicit restart'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==UUID
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 old=read(U/'protocol.json');native=read(U.parent/'native_e2e32_3090_recheck/protocol.json');sources=dict(native['sources'])
 for p,h in sources.items():assert sha(p)==h,p
 refs={}
 for name in ['code/e2e32_run.py','code/e2e32_runtime.py','code/e2e32_prepare.py','code/e2e32_analyze.py','protocol.json','input_verification.json','input_manifest.jsonl.gz','efficiency_summary.csv','drift_summary.json']:
  b=subprocess.check_output(['git','-C',str(R),'show','5ee2a0dbe78df336229b55fed14a4a6d0d880bed:'+str((U/name).relative_to(R))]);h=hashlib.sha256(b).hexdigest();assert h==sha(U/name),name;refs[name]=h
 mapping=read(S/'frozen30_image_path_mapping.json');rows=lines(U/'input_manifest.jsonl.gz');assert len(rows)==30 and [r['key'] for r in rows]==old['keys']
 for r in rows:
  assert sha(mapping[r['key']])==r['image_sha256'] and hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
 assert read(S/'input_verification_30.json')['all_three_input_tensor_SHA256_exact']
 receipt=read(M/'.ate_revision.json');meta=read(S/'hf_revision_metadata.json');assert receipt['revision']==old['revision'] and receipt['all_LFS_SHA256_verified']
 for e in meta['siblings']:
  item=receipt['files'][e['rfilename']];assert (M/e['rfilename']).stat().st_size==item['bytes']==e['size'];assert item['hash']==e.get('lfs',{}).get('sha256',e['blobId'])
 D.mkdir(exist_ok=True);L.mkdir(parents=True);(L/'code').mkdir();(D/'code').mkdir(exist_ok=True)
 for f in Path(__file__).parent.glob('*.py'):
  if f.name in ['prepare_3090.py','run_3090.py','monitor_3090.py','history_policy.py']:shutil.copyfile(f,L/'code'/f.name)
 protected=read(S/'historical_protected_SHA256.json')
 previous=D.parent/'unified_e2e32_eight_methods_3090'
 protected.update({str(f):sha(f) for f in previous.rglob('*') if f.is_file()})
 for p,h in protected.items():assert sha(p)==h,p
 p=dict(old);p.update(task='One RTX3090 eight-method physical efficiency session, gated before any formal request',reference_commit='5ee2a0dbe78df336229b55fed14a4a6d0d880bed',GPU_UUID=UUID,reference_GPU_UUID=old['GPU_UUID'],output=str(D),local_spool=str(L),formal_sessions=1,requested_formal_requests=240,model_receipt_SHA256=sha(M/'.ate_revision.json'),input_manifest_SHA256=sha(U/'input_manifest.jsonl.gz'),input_mapping_SHA256=digest(mapping),reference_files_SHA256=refs,entry_source_SHA256={f.name:sha(f) for f in (L/'code').glob('*.py')},created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality_requests=0,GitHub_writes=False,Profiler_used=False)
 p['native_stability']=dict(minimum_requests=60,minimum_summed_E2E_s=180,maximum_requests=120,window_requests=20,probe_order=old['smoke_keys'],evaluate_after=[60,80,100,120],required_latest_nonoverlapping_windows=3,metrics=['E2E_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'],per_input=True,robust_CV_limit=.01,adjacent_window_mean_change_limit=.01,three_window_mean_range_limit=.02,single_request_deviation_from_window_median_limit=.03,native_early5_vs_late26_decode_median_change_limit=.05,unexplained_request_gap_limit_s=1,telemetry_max_gap_s=1,scope='This bounded Native stability warmup excludes separately labelled correctness Smoke, equal-method warmup and independent post-admission baseline. No gate baseline is chosen by speed.')
 p['session_label']='session2';p['session_index']=2
 p['history_output_policy']=dict(Native='strict_full32',FastV='strict_prefix_through_first_EOS_inclusive; if neither has EOS require full32; retain every full32 difference',other_methods='unchanged historical Smoke policy',EOS_token_id=2,policy_audit_SHA256=sha(Path(__file__).parent/'patch_audit.json'))
 p['extra_policy_smoke_key']='GQA:201759069'
 p['equal_warmup_per_method']=2;p['smoke_requests_planned']=21;p['independent_baseline_requests']=20
 p['drift']=dict(keys=old['smoke_keys'],baseline_rounds=10,after_each_inputs=5,probe_repeats_per_input=3,threshold_fraction=.02,metrics=['E2E_ms','TPOT_ms'],per_input_and_pooled=True,consecutive_breaches_to_pause=2,immediate_probe_change_limit=.05,within_request_native_step_jump_limit=.05,automatic_rebaseline=False)
 p['schedule']='30 frozen inputs; all eight methods per input, method list rotated by input index modulo8. One model load. Equal warmup2 per method, then bounded Native admission, independent baseline, formal240 and six calibration windows.'
 p['resume']='Immutable local SQLite+SHA and attempts log. Never automatically reload and continue a partially measured session; preserve PARTIAL/BLOCKED and all data.'
 p['telemetry']=dict(GPU_Hz=5,CPU_Hz=1,process_inventory_period_s=2,independent_collector_processes=True,local_spool=True,no_UUID_subprocess_per_request=True)
 p['stop_on']='GPU conflict, UUID/source/input/model mismatch, nonfinite/invalid KV/RoPE/layout/no-op, failed admission, persistent 2% drift or a >5% abrupt Native decode/probe change. No automatic retry of failed formal requests.'
 for root in [D,L]:
  save(root,'protocol.json',p);save(root,'source_manifest.json',dict(frozen_sources=sources,reference_commit_files=refs,new_entry_sources=p['entry_source_SHA256']));save(root,'historical_protected_SHA256.json',protected);save(root,'input_path_mapping.json',mapping)
  for n in ['input_manifest.jsonl.gz','question_positions.json','historical_token_references.json']:shutil.copyfile(U/n,root/n)
 for n in ['patch_audit.json','session2_entry.diff','policy_cpu_test.py']:
  shutil.copyfile(Path(__file__).parent/n,D/n)
 save(D,'preflight.json',dict(status='PASS',GPU_UUID=UUID,images=30,CPU_tensor_acceptance_from_recovery=sha(S/'input_verification_30.json'),model_files=len(receipt['files']),model_receipt_SHA256=p['model_receipt_SHA256'],GPU_before=subprocess.check_output(['nvidia-smi','-q'],text=True),CPU_info=Path('/proc/cpuinfo').read_text(),mounts=Path('/proc/mounts').read_text(),CPU_affinity=sorted(os.sched_getaffinity(0)),free_disk_bytes=shutil.disk_usage('/home/featurize').free))
 print(json.dumps(dict(status='CPU_PREFLIGHT_FROZEN',output=str(D),local=str(L),source_files=len(sources),formal_max=240,stability_Native_max=120)))
if __name__=='__main__':prepare()
