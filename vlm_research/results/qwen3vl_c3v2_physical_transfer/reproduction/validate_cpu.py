import pathlib,json,csv,hashlib,subprocess,datetime,ast,collections
W=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer');O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
checks={}
for p in O.rglob('*.json'):read(p)
checks['all_json_parse']=True
src=read(O/'source_manifest.json');bound={}
for name,h in src['files'].items():
 assert sha(W/name)==h
 blob=subprocess.check_output(['git','show',src['commit']+':'+name],cwd=W)
 assert hashlib.sha256(blob).hexdigest()==h
 ast.parse((W/name).read_text())
checks['execution_source_matches_frozen_git_and_sha']=True
model=read(O/'model_manifest.json');modelroot=pathlib.Path(model['model'])
for name,item in model['files'].items():assert (modelroot/name).stat().st_size==item['bytes'] and sha(modelroot/name)==item['sha256'],name
checks['model_all_files_rehashed']=True
m=read(O/'sample_manifest.json');ss={s['id']:s for s in m['samples']};inputs=read(O/'input_bindings.json');protocol=read(O/'protocol.json')
assert sha(O/'sample_manifest.json')==protocol['data_manifest_sha256'] and sha(O/'input_bindings.json')==protocol['input_bindings_sha256']
for s in ss.values():assert sha(s['image_path'])==s['image_sha256']
checks['all_frozen_image_shas_rehashed']=True
groups=collections.defaultdict(set)
for s in ss.values():groups[s['image_sha256']].add(s['shard'])
assert all(len(x)==1 for x in groups.values())
checks['whole_image_shards_disjoint_and_complete']=True
index=[]
for rank in range(2):
 for p in (O/f'shard{rank}/smoke').glob('*.json'):
  d=read(p);s=d['sample'];r=d['result'];assert s==ss[s['id']]
  index.append(dict(path=str(p.relative_to(O)),sha256=sha(p),bytes=p.stat().st_size,phase='FAILED_SMOKE_ONLY',rank=rank,condition=r['condition'],sample_id=s['id'],image_sha256=s['image_sha256'],source_commit=src['commit'],source_manifest_sha256=sha(O/'source_manifest.json'),input_bindings_sha256=sha(O/'input_bindings.json'),input_tensor_hashes=inputs[s['id']]['input_tensor_hashes'],model_manifest_sha256=sha(O/'model_manifest.json')))
assert len(index)==12
write(O/'checkpoint_index.json',index)
checks['12_smoke_checkpoint_bindings_indexed']=True
for c in ['F','P','B','C','W22-functional','A16-functional']:
 pairs=[read(O/f'shard{r}/smoke/GQA__06905471__{c}.json')['result'] for r in range(2)]
 assert pairs[0]['token_ids']==pairs[1]['token_ids']==[2152,151645]
 assert pairs[0]['audit']==pairs[1]['audit']
checks['two_gpu_saved_execution_audits_equal']=True
for row in csv.DictReader((O/'smoke_physical_execution.csv').open()):
 raw=read(O/f"shard{row['rank']}/smoke/GQA__06905471__{row['condition']}.json")['result']['audit']
 assert float(row['language_matrix_gflops'])==sum(x['flops'] for x in raw['projection_shapes'])/1e9
 assert int(row['kv_live_bytes'])==sum(x['live_bytes'] for x in raw['kv']['layers'])
 assert int(row['kv_capacity_bytes'])==sum(x['capacity_bytes'] for x in raw['kv']['layers'])
checks['shape_flops_and_kv_recomputed']=True
for d in ['GQA','VQAv2']:assert read(O/'image_overlap_audit.json')['postfreeze_image_id_audit']['overlap_image_ids'][d]==[]
checks['historical_id_audit_zero_overlap']=True
gate=read(O/'STOP.json');fail=ast.literal_eval(gate['error'][len('AssertionError('):-1]);assert fail['max_abs']==.4453125 and fail['mean_abs']==.08348099887371063 and fail['top1_equal']
assert protocol['smoke_tolerance']['functional_physical_mean_abs']==.02
assert 'Ran 12 tests' in (O/'cpu_tests.txt').read_text() and (O/'cpu_tests.txt').read_text().rstrip().endswith('OK')
checks['frozen_failure_and_cpu_test_log_verified']=True
assert not any((O/f'shard{r}/{phase}').exists() for r in range(2) for phase in ['pilot','quality','benchmark','physical_audit'])
assert all(row['n']=='0' and row['accuracy']=='' for row in csv.DictReader((O/'accuracy_summary.csv').open()))
assert len(list(csv.DictReader((O/'paired_comparisons.csv').open())))==0
checks['no_formal_quality_timing_bootstrap_after_failed_gate']=True
for p in (O/'figures').glob('*.pdf'):
 info=subprocess.check_output(['pdfinfo',str(p)],text=True);assert next(x for x in info.splitlines() if x.startswith('Pages:')).split(':')[1].strip()=='1'
 text=subprocess.check_output(['pdftotext',str(p),'-'],text=True);assert len(text.strip())>30
checks['all_three_vector_pdfs_one_page_and_text_readable']=True
main=subprocess.check_output(['git','rev-parse','main'],cwd=W,text=True).strip();assert main=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
checks['main_unchanged']=True
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert 'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6' not in procs and 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185' not in procs
now=datetime.datetime.now(datetime.timezone.utc)
write(O/'gpu_release_audit.json',dict(at=now.isoformat(),gpu_query=gpu,process_query=procs,released=True,other_processes_killed=False,new_gpu_tasks_after_numerical_failure=0))
checks['both_assigned_gpu_processes_exited']=True
write(O/'validation.json',dict(status='DELIVERY_EVIDENCE_VALIDATED_EXPERIMENT_GATE_FAILED',checks=checks,numerical_gate_passed=False,all_gpu_smokes_passed=False,at=now.isoformat(),source_commit=src['commit'],main=main,formal_generations=0,distinct_smoke_questions=1,limitations=['Native/No-op raw logits not persisted','No per-layer raw hidden arrays for localizing numerical error','Diagnostic answer masking not pure Native-relative isolation','No formal quality, bootstrap, latency or speedup results']))
completion=read(O/'completion.json');completion.update(delivery_complete=True,at=now.isoformat(),elapsed_minutes=(now.timestamp()-read(O/'time_budget.json')['start_epoch'])/60,reason='Both GPUs F-vs-C first-step mean absolute logit error .0834809989 exceeds predeclared .02; all later experiments stopped',experiment_complete=False)
write(O/'completion.json',completion)
files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='artifact_index.json'}
write(O/'artifact_index.json',dict(files=files,index_excludes_self=True,experiment_status=completion['status'],validation_status='PASS_FOR_DELIVERY_ONLY'))
print('VALIDATED',len(checks),'checks;',len(files),'artifacts;',round(completion['elapsed_minutes'],2),'minutes; numerical gate remains FAILED',flush=True)
