set -eu
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=''
python3 -u - <<'PY'
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os,subprocess
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110');W=L/'worktree';O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';D=L/'data';M=L/'model'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
copies=read(L/'copy_progress.json');assert copies['status']=='COPIED_PENDING_INTEGRITY_AND_BIND_MOUNTS'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for n,h in read(O/'source_manifest.json')['files'].items():assert sha(W/n)==h,('Frozen source',n)
for manifest in ['analysis_manifest.json','partial_diagnostics_manifest.json']:
 if (O/manifest).exists():
  for n,h in read(O/manifest)['files'].items():assert sha(O/n)==h,(manifest,n)
timestamp_restored=[]
for n,s in read(O/'model_manifest.json')['files'].items():
 p=M/n;assert p.is_file() and p.stat().st_size==s['bytes'],('Model size',n)
 assert sha(p)==s['SHA256'],('Model SHA256',n)
 if p.stat().st_mtime_ns!=s['mtime_ns']:
  os.utime(p,ns=(p.stat().st_atime_ns,s['mtime_ns']));timestamp_restored.append(n)
 assert p.stat().st_mtime_ns==s['mtime_ns']
 print('LOCAL_MODEL_FILE_SHA256_VERIFIED',n,flush=True)
base={k+'_SHA256':sha(O/(k+'.json')) for k in ['protocol','source_manifest','model_manifest']};base['GPU_UUID']=read(O/'protocol.json')['gpu_UUID']
manifest=read(O/'input_manifest.json');manifest_sha=sha(O/'input_manifest.json');historical=read(O/'historical_inputs.json')
assert manifest_sha=='4f95043737ca7be4c6e62442064406b9f81d91ca5974e7873c02f40d7780fb3f'
assert read(O/'historical_completion.json')['binding']==base and read(O/'validation.json')['binding']==base
for key,name in [('protocol_SHA256','protocol.json'),('source_manifest_SHA256','source_manifest.json'),('model_manifest_SHA256','model_manifest.json')]:assert sha(L/'migration_checkpoint'/name)==base[key]
image_paths={s['image_path'] for name in [historical,manifest] for s in name['rows']}
for old in image_paths:
 rel=Path(old).relative_to('/home/featurize/work/ate-qwen3090-data');p=D/rel
 assert p.is_file() and p.stat().st_size>0,('Missing copied input image',str(rel))
peak=0;hist_count=0
def observe(r):
 global peak
 assert len(r['token_ids'])==r['generated_tokens'],'Stored token count mismatch'
 peak=max(peak,r['memory']['peak_allocated'])
for s in historical['rows']:
 for c in ['Native','Dense-F','Physical-Original']:
  record=read(O/'historical_predictions'/(s['question_id']+'__'+c+'.json'));b=record['binding']
  for k,v in base.items():assert b[k]==v,('Historical binding',s['question_id'],k)
  assert b['sample_SHA256']==digest(s) and b['input_SHA256']==digest(historical['input_bindings'][s['question_id']])
  observe(record['result']);hist_count+=1
assert hist_count==3000 and len(list((O/'historical_predictions').glob('*.json')))==3000
requests={};request_count=0
for p in sorted((O/'request_predictions').glob('*/*.json')):
 rec=read(p);b=rec['binding']
 for k,v in base.items():assert b[k]==v,('Request binding',str(p),k)
 layout_sha=digest(rec['input_layout']);assert layout_sha==b['input_SHA256']
 assert p.name.split('__',1)[0]==b['request_SHA256']
 requests[str(p.relative_to(O))]=(digest(rec),layout_sha)
 observe(rec['result']);request_count+=1
 if request_count%5000==0:print('LOCAL_REQUEST_CHECKPOINTS_VERIFIED',request_count,flush=True)
samples={(s['benchmark'],s['protocol_label'],hashlib.sha256(s['key'].encode()).hexdigest()):s for s in manifest['rows']}
counts=Counter();pair_conditions=defaultdict(set);index_count=0
for p in sorted((O/'benchmark_predictions').glob('*/*/*.json')):
 if p.name in ['failure.json','completion.json']:continue
 parts=p.relative_to(O/'benchmark_predictions').parts;benchmark,label,name=parts;stem,c=name[:-5].split('__',1)
 s=samples[benchmark,label,stem];rec=read(p);b=rec['binding']
 for k,v in base.items():assert b[k]==v,('Index binding',str(p),k)
 assert rec['key']==s['key'] and rec['question_id']==s['question_id'] and rec['gold']==s['answer'] and rec['image_SHA256']==s['image_sha256']
 assert b['sample_SHA256']==digest(s) and b['input_manifest_SHA256']==manifest_sha and b['request_SHA256']==s['request_SHA256']
 generation_sha,layout_sha=requests[rec['request_path']]
 assert rec['generation_SHA256']==generation_sha and b['input_SHA256']==layout_sha
 counts[benchmark,label,c]+=1;pair_conditions[benchmark,label,s['key']].add(c);index_count+=1
 if index_count%10000==0:print('LOCAL_BENCHMARK_INDICES_VERIFIED',index_count,flush=True)
for p in (O/'smoke').glob('*.json'):
 for r in read(p).get('results',{}).values():observe(r)
prior=read(O/'seven_progress.json')['views']
for v in prior:
 for c in ['Native','Physical-Original']:assert counts[v['benchmark'],v['protocol_label'],c]==v['completed_questions'],('Completed/blocked view count mismatch',v)
 if v['status']=='BLOCKED':assert v['failure']['error']=='CUDA_OUT_OF_MEMORY'
receipt=dict(status='LOCAL_COPY_INTEGRITY_PASS_PENDING_BIND_MOUNTS',at_UTC=datetime.now(timezone.utc).isoformat(),authorization='User requested local disk execution with persistent checkpoint backup',local_root=str(L),copies=copies['roots'],persistent_originals_preserved=True,source_SHA256_verified=True,analysis_SHA256_verified=True,model_all_files_SHA256_verified=True,model_mtime_restored_from_frozen_manifest=timestamp_restored,model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',input_manifest_SHA256=manifest_sha,unchanged_input_paths=len(image_paths),data_copy_verification='Successful rsync transfer checksums; all 10,248 referenced image paths exist and nonempty; original on-use image SHA256 checks remain enabled; no redundant full-image hash sweep',historical_predictions_verified=hist_count,request_predictions_verified=request_count,benchmark_indices_verified=index_count,per_view_condition_counts=[dict(benchmark=b,protocol_label=v,condition=c,count=n) for (b,v,c),n in counts.items()],unpaired_completed_indices=sum(len(cs)!=2 for cs in pair_conditions.values()),prior_observed_peak_allocated=peak,previous_view_statuses=prior,old_worker_pause=read(L/'migration_checkpoint/pause.json'),frozen_inference_files_changed=False,previous_OOM_view_will_not_be_retried=True,new_GPU_inference_started=False)
(O/'local_disk_migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(L/'integrity_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('LOCAL_MIGRATION_INTEGRITY_PASS',json.dumps({k:receipt[k] for k in ['historical_predictions_verified','request_predictions_verified','benchmark_indices_verified','unchanged_input_paths','unpaired_completed_indices','prior_observed_peak_allocated']}),flush=True)
PY
