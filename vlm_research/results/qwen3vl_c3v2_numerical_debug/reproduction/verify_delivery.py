"""CPU-only verification of delivered evidence; never loads a model on GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,csv,hashlib,subprocess,datetime,shutil
import numpy as np
import torch
torch.set_num_threads(1)
W=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-debug')
O=W/'vlm_research/results/qwen3vl_c3v2_numerical_debug'
OLD=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer')
P=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_debug')
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=W)
def blobs(requests):
 payload=('\n'.join(requests)+'\n').encode();raw=subprocess.check_output(['git','cat-file','--batch'],input=payload,cwd=W);pos=0;result=[]
 for request in requests:
  e=raw.index(b'\n',pos);head=raw[pos:e].decode().split();assert len(head)==3,(request,head)
  n=int(head[2]);data=raw[e+1:e+1+n];assert raw[e+1+n:e+2+n]==b'\n';pos=e+2+n;result.append(data)
 return result
checks={};model=read(OLD/'vlm_research/results/qwen3vl_c3v2_physical_transfer/model_manifest.json');mh=[]
for name,item in model['files'].items():
 p=pathlib.Path(model['model'])/name;actual=sha(p);assert actual==item['sha256'];assert p.stat().st_size==item['bytes'];mh.append(dict(name=name,sha256=actual,bytes=p.stat().st_size))
write(O/'model_hash_revalidation.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_files_match_original_manifest=True,files=mh,timing='Full byte rehash at CPU delivery verification after GPU diagnosis. Prior manifest and path bindings were reused at launch; this audit does not claim a new pre-GPU full-byte rehash.'))
checks['all_model_files_match_old_freeze']=True
bindings=[]
for name in ['source_manifest.json','source_manifest_probes_v1.json','source_manifest_probes_v2.json']:
 m=read(O/name)
 for path,digest in m['files'].items():bindings.append((m['commit']+':'+path,digest,path,name))
for b,data in zip(bindings,blobs([b[0] for b in bindings])):
 assert hashlib.sha256(data).hexdigest()==b[1],b
 if b[3]=='source_manifest.json':assert sha(W/b[2])==b[1],b
checks['all_three_source_manifests_match_exact_git_commits']=True
historical=read(O/'historical_bindings.json');req=[]
for b in historical:
 p=pathlib.Path(b['path']);rel=p.relative_to(OLD).as_posix() if p.is_absolute() else b['path'];req.append(b['commit']+':'+rel)
for b,data in zip(historical,blobs(req)):
 assert hashlib.sha256(data).hexdigest()==b['sha256']
 live=pathlib.Path(b['path']) if pathlib.Path(b['path']).is_absolute() else OLD/b['path'];assert sha(live)==b['sha256']
checks['old_source_reports_stop_checkpoints_unchanged']=len(historical)
for x in read(O/'sample_manifest.json')['samples']:
 assert sha(x['image_path'])==x['image_sha256']
checks['fixed_images_sha_verified']=True
for x in read(O/'raw_tensor_index.json')['files']:assert sha(x['path'])==x['sha256']
checks['all_external_tensor_hashes_verified']=True
z=np.load(O/'arrays/effective_logits.npz');desc=read(O/'arrays/effective_logits.json');assert sha(O/'arrays/effective_logits.npz')==desc['sha256']
for folder in ['shard1','repair_shard0']:assert np.array_equal(z[folder+'__N'],z[folder+'__Noop'])
rows=read(O/'shard1/factor_isolation.json')+read(O/'repair_shard0/repair_validation.json')['comparisons']
for i,r in enumerate(rows):
 folder='shard1' if i<8 else 'repair_shard0';a=torch.from_numpy(z[folder+'__'+r['reference']][r['step']]);b=torch.from_numpy(z[folder+'__'+r['candidate']][r['step']]);d=a-b
 for key,val in [('max_abs',d.abs().max().item()),('mean_abs',d.abs().mean().item()),('rmse',d.square().mean().sqrt().item())]:assert val==r[key],(folder,key,val,r[key])
 assert bool(a.argmax()==b.argmax())==r['top1_equal']
checks['all_12_paired_logit_rows_recomputed_exactly']=True
checks['native_noop_all_effective_logits_equal']=True
for c in ['N','Noop','F','B','W22-functional','P','C']:
 a=torch.load(P/'shard0'/f'{c}_logits.pt',map_location='cpu',weights_only=True);b=torch.load(P/'shard1'/f'{c}_logits.pt',map_location='cpu',weights_only=True)
 assert a['token_ids']==b['token_ids'] and all(torch.equal(x,y) for x,y in zip(a['logits'],b['logits']))
checks['two_gpu_baseline_all_vectors_and_tokens_strict_equal']=True
for rank in range(2):
 n=read(O/f'shard{rank}/native_validation.json');assert n['native_hf_logits_equal'] and n['native_hf_tokens_equal'] and n['native_noop_equal']
 for c in ['F','B','W22-functional','P','C']:assert read(O/f'shard{rank}/probe_{c}_validation.json')['probe_logits_equal']
checks['native_hf_and_readonly_probe_records_pass']=True
e=np.load(O/'arrays/first_divergence_evidence.npz');ed=read(O/'arrays/first_divergence_evidence.json');assert sha(O/'arrays/first_divergence_evidence.npz')==ed['sha256']
def tensor(key):
 x=torch.from_numpy(e[key].copy());d=next(d for d in ed['arrays'] if d['key']==key);return x.view(torch.bfloat16) if d['dtype']=='torch.bfloat16' else x
first=[]
for prefix in ['L1_F_B','L22_W_P','repaired_L22_F_C']:
 stages=[d['key'].split('__')[1] for d in ed['arrays'] if d['key'].startswith(prefix+'__') and d['key'].endswith('__reference')]
 for stage in stages:
  a=tensor(prefix+'__'+stage+'__reference');b=tensor(prefix+'__'+stage+'__candidate')
  if not torch.equal(a,b):first.append(dict(contrast=prefix,first_saved_nonzero_stage=stage,changed_elements=int((a!=b).sum())));break
assert [r['first_saved_nonzero_stage'] for r in first]==['attention_AV_output','k_proj','k_proj'];assert first[0]['changed_elements']==4
checks['compact_bit_exact_first_divergence_archives_recomputed']=first
ka=read(O/'kv_cache_audit.json');assert len(ka['rows'])==360;assert ka['all_prefill_stored_keys_values_equal_recorded_projection_gather'] and ka['all_decode_prefixes_unchanged'] and ka['all_original_positions_correct'];checks['kv_content_position_and_append_checks']=360
pa=read(O/'position_and_mask_audit.json');assert pa['all_common_mask_permissions_equal'] and pa['all_common_position_ids_equal'] and pa['all_common_rotary_equal'];checks['positions_masks_mrope_pass']=True
cleanup=read(O/'cpu_exception_cleanup.json');assert cleanup['exception_exit_hooks_and_patches_restored'] and cleanup['normal_exit_hooks_and_patches_restored'];checks['normal_and_exception_hook_cleanup']=True
json_count=csv_count=0
for p in O.rglob('*.json'):read(p);json_count+=1
for p in O.rglob('*.csv'):
 with p.open(newline='') as f:
  r=csv.DictReader(f);assert r.fieldnames is not None
  for row in r:assert None not in row,(str(p),row)
 csv_count+=1
checks['all_json_csv_parsed']=dict(json=json_count,csv=csv_count)
figs=[]
for p in sorted((O/'figures').glob('*.pdf')):
 info=subprocess.check_output(['pdfinfo',str(p)],text=True);assert next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:'))=='1'
 text=subprocess.check_output(['pdftotext',str(p),'-'],text=True);assert len(text.strip())>40;assert p.with_suffix('.png').exists();figs.append(p.name)
assert len(figs)==5;checks['all_five_single_page_vector_pdf_and_png_pairs']=figs
assert git('rev-parse','main').decode().strip()=='7970be981e3462560cb66d83e1014eaa1b1ebdae';checks['main_unchanged']=True
free=shutil.disk_usage('/data').free;assert free>=20*1024**3;checks['disk_free_GiB']=free/1024**3
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True);procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
for uuid in ['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']:assert uuid not in procs
pidchecks=[]
for p in P.glob('*.pid'):
 pid=int(p.read_text().strip());alive=pathlib.Path('/proc')/str(pid);assert not alive.exists();pidchecks.append(pid)
write(O/'gpu_release_audit.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,processes=procs,own_worker_pids_exited=pidchecks,both_assigned_devices_released=True,other_processes_killed=False,no_gpu_work_after_failed_repair=True))
checks['gpu_workers_exited_devices_free']=True
repair=read(O/'repair_shard0/repair_validation.json');assert repair['status']=='ORIGINAL_CASE_FAILED_STOP_EXPANSION' and repair['extra_samples_run']==0
write(O/'validation.json',dict(status='DELIVERY_EVIDENCE_VALIDATED',scientific_outcome='ROOT_CAUSE_IDENTIFIED_REPAIR_FAILED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,repair_accepted=False,formal_quality_or_performance_ready=False,extra_smokes_run=False,full_model_accuracy_run=False,benchmark_run=False,limitations=['One fixed input only; no generalization claim','Detailed probes are Prefill; remaining Decode difference not fully operator-localized','BF16 shape sensitivity isolated; internal CUDA kernel not profiled','Repair dummy key relabeled only in CPU alignment sidecar; raw arrays unchanged','Model full-byte rehash performed at delivery stage']))
print(json.dumps(checks,indent=2),flush=True)
