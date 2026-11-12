"""CPU preflight; bind the original frozen implementation without rewriting it."""
from pathlib import Path
import os,json,gzip,hashlib,subprocess,shutil,datetime
R=Path('/home/featurize/work/ATE');U=R/'vlm_research/results/unified_e2e32_4090';B=U.parent/'unified_e2e32_eight_methods_3090_repeatability';SESSION=int(os.environ.get('ATE_REPEAT_SESSION','1'));assert SESSION in (1,2);D=B/('session'+str(SESSION));L=Path('/home/featurize/ate-local-experiments/unified_e2e32_eight_methods_3090_repeatability')/('session'+str(SESSION));S=U.parent/'featurize_23386_environment_restore_20261110';M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf');UUID='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27'
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
 D.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True);(L/'code').mkdir();(D/'code').mkdir(exist_ok=True)
 for f in Path(__file__).parent.glob('*.py'):
  if f.name in ['prepare_3090.py','run_3090.py','monitor_3090.py','history_policy.py']:shutil.copyfile(f,L/'code'/f.name)
 protected=read(S/'historical_protected_SHA256.json')
 previous=B.parent/'unified_e2e32_eight_methods_3090'
 protected.update({str(f):sha(f) for f in previous.rglob('*') if f.is_file()})
 previous2=D.parent.parent/'unified_e2e32_eight_methods_3090_session2'
 protected.update({str(f):sha(f) for f in previous2.rglob('*') if f.is_file()})
 for p,h in protected.items():assert sha(p)==h,p
 p=dict(old);p.update(task='Two independent RTX3090 sessions; fixed warmup; performance fluctuations recorded as WARN, correctness/safety remain hard gates',reference_commit='5ee2a0dbe78df336229b55fed14a4a6d0d880bed',GPU_UUID=UUID,reference_GPU_UUID=old['GPU_UUID'],output=str(D),local_spool=str(L),formal_sessions=1,requested_formal_requests=240,model_receipt_SHA256=sha(M/'.ate_revision.json'),input_manifest_SHA256=sha(U/'input_manifest.jsonl.gz'),input_mapping_SHA256=digest(mapping),reference_files_SHA256=refs,entry_source_SHA256={f.name:sha(f) for f in (L/'code').glob('*.py')},created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality_requests=0,GitHub_writes=False,Profiler_used=False)
 p['session_label']='repeatability_session'+str(SESSION);p['session_index']=SESSION
 p['warmup']=dict(Native_requests=16,other_method_requests_each=2,probe_order=old['smoke_keys'],Native_first=True,full32=True,extra_policy_smoke='GQA:201759069',fixed_counts=True,no_performance_admission=True,performance_statistics_warning_only=True)
 p['history_output_policy']=dict(Native='strict_full32',FastV='first_EOS_inclusive; neither EOS means full32',other_methods='record_all_historical_differences_without_output-only_hard_failure',EOS_token_id=2,policy_audit_SHA256=sha(Path(__file__).parent/'patch_audit.json'))
 p['extra_policy_smoke_key']='GQA:201759069'
 p['other_method_warmup_each']=2;p['smoke_requests_planned']=21;p['independent_baseline_requests']=20
 p['drift']=dict(keys=old['smoke_keys'],baseline_rounds=10,after_each_inputs=5,probe_repeats_per_input=3,threshold_fraction=.02,metrics=['E2E_ms','TPOT_ms'],per_input_and_pooled=True,performance_breaches_stop=False,within_request_decode_warning_limit=.05,automatic_rebaseline=False)
 p['schedule']='Fixed Native16, other7x2, Smoke21, independent Native20, interleaved formal240, six Native dual-probe calibration windows6 each. Same sequence in both sessions.'
 p['explicit_formal_order']=[dict(input_index=i+1,key=k,methods=p['requested_methods'][i%8:]+p['requested_methods'][:i%8]) for i,k in enumerate(p['keys'])]
 p['schedule_SHA256']=digest(dict(formal=p['explicit_formal_order'],warmup=p['warmup'],smoke=p['smoke_keys'],baseline_requests=20,calibration=p['drift']))
 p['resume']='Immutable local SQLite+SHA and attempts log. Never automatically reload and continue a partially measured session; preserve PARTIAL/BLOCKED and all data.'
 p['telemetry']=dict(GPU_Hz=5,CPU_Hz=1,process_inventory_period_s=2,independent_collector_processes=True,local_spool=True,no_UUID_subprocess_per_request=True)
 p['stop_on']='Only input/source/model/UUID/correctness/no-op/physical-layout/nonfinite/OOM, GPU conflict, hardware safety or unavailable safety monitoring. All pure performance thresholds WARN and continue. No retries/replacements.'
 for root in [D,L]:
  save(root,'protocol.json',p);save(root,'source_manifest.json',dict(frozen_sources=sources,reference_commit_files=refs,new_entry_sources=p['entry_source_SHA256']));save(root,'historical_protected_SHA256.json',protected);save(root,'input_path_mapping.json',mapping)
  for n in ['input_manifest.jsonl.gz','question_positions.json','historical_token_references.json']:shutil.copyfile(U/n,root/n)
 for n in ['patch_audit.json','repeatability_entry.diff','policy_cpu_test.py']:
  shutil.copyfile(Path(__file__).parent/n,D/n)
 save(D,'preflight.json',dict(status='PASS',GPU_UUID=UUID,images=30,CPU_tensor_acceptance_from_recovery=sha(S/'input_verification_30.json'),model_files=len(receipt['files']),model_receipt_SHA256=p['model_receipt_SHA256'],GPU_before=subprocess.check_output(['nvidia-smi','-q'],text=True),CPU_info=Path('/proc/cpuinfo').read_text(),mounts=Path('/proc/mounts').read_text(),CPU_affinity=sorted(os.sched_getaffinity(0)),free_disk_bytes=shutil.disk_usage('/home/featurize').free))
 print(json.dumps(dict(status='CPU_PREFLIGHT_FROZEN',output=str(D),local=str(L),source_files=len(sources),formal_max=240,Native_fixed_warmup=16)))
if __name__=='__main__':prepare()
