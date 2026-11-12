from pathlib import Path
import json,hashlib,shutil,subprocess,sys,time,concurrent.futures,sqlite3
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
N=Path('/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert json.loads((A/'coordinator_status.json').read_text())['status']=='COMPLETE_ALL_PHASES'
assert json.loads((A/'refinement_coordinator_status.json').read_text())['status']=='COMPLETE'
# Make each raw SQLite checkpoint independently restorable without WAL/SHM.
checkpoint_checks=[]
for dbpath in sorted(A.rglob('checkpoint.sqlite')):
 con=sqlite3.connect(str(dbpath))
 integrity=con.execute('PRAGMA integrity_check').fetchall()
 assert integrity==[('ok',)],(str(dbpath),integrity)
 n=con.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
 jp=dbpath.parent/'requests.jsonl'
 json_rows=[json.loads(l) for l in jp.read_text().splitlines() if l.strip()] if jp.exists() else []
 expected=len(json_rows)
 assert n==expected,(str(dbpath),n,expected)
 for stored,expected_row in zip(con.execute('SELECT id,payload,sha256 FROM requests ORDER BY id'),json_rows):
  row_id,payload,digest=stored
  assert hashlib.sha256(payload.encode()).hexdigest()==digest and json.loads(payload)==expected_row and row_id==expected_row['id'],('PAYLOAD_SHA_JSONL_MISMATCH',str(dbpath),row_id)
 checkpoint=con.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
 assert checkpoint[0]==0,(str(dbpath),checkpoint)
 con.close()
 wal=Path(str(dbpath)+'-wal')
 assert not wal.exists() or wal.stat().st_size==0,str(wal)
 checkpoint_checks.append(dict(path=str(dbpath.relative_to(A)),status='PASS',rows=n,sqlite_integrity='ok',payload_SHA256_JSONL_exact=True,wal_checkpoint=checkpoint,independent_main_database=True))
save(A/'checkpoint_storage_validation.json',dict(status='PASS',checkpoints=checkpoint_checks,WAL_SHM_not_needed_for_restore=True))
subprocess.run([sys.executable,'-B',str(A/'code/analyze_sources_v1.py')],check=True)
subprocess.run([sys.executable,'-B',str(A/'code/snapshot_runtime_dependencies.py')],check=True)
subprocess.run([sys.executable,'-B',str(A/'code/validate_all_v1.py')],check=True)
assert json.loads((A/'validation.json').read_text())['status']=='PASS_WITH_REJECTED_CANDIDATES'
versions=[]
for p in sorted((A/'code').glob('run_*.py')):
 versions.append(dict(path=str(p.relative_to(A)),sha256=sha(p)))
save(A/'execution_runner_versions.json',dict(versions=versions,actual_usage={'Qwen/screen':'code/run_optimization.py','Qwen/other_primary_phases':'code/run_optimization_v2.py','InternVL/successful_primary':'code/run_optimization_v3.py','LLaVA/successful_primary':'code/run_optimization_v4.py','refine_*':'code/run_refinement_e.py'},history='Early environment and Native cache metadata failures retained separately; no algorithm/timing/gate relaxation'))
source_files=[p for p in A.rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.so'] and '__pycache__' not in p.parts]
source_rows=[dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(source_files)]
save(A/'source_manifest.json',dict(files=source_rows,frozen_original_manifest='frozen_source_manifest.json',modified_files='changed_source_manifest.json',runtime_and_model_sources='runtime_dependency_manifest.json',runners='execution_runner_versions.json',no_original_source_overwrite=True))
commands=['# Actual completed commands, one GPU process at a time; model/data reused, no Git writes.',
 '# Qwen screen uses v1; other Qwen primary phases v2; successful InternVL phases v3, LLaVA phases v4.',
 '# All method-specific setup remains inside the frozen generate() timing boundary.']
for kind in ['Qwen','InternVL','LLaVA']:
 env='/home/featurize/ate-local-envs/'+({'Qwen':'qwen3vl-torch260-tf4576','InternVL':'internvl3-torch260-tf449','LLaVA':'ate-torch260-cu124-tf449'}[kind])+'/bin/python'
 for G in sorted((A/kind).iterdir()):
  if not G.is_dir() or not (G/'completion.json').exists():continue
  c=json.loads((G/'completion.json').read_text())
  if not c['status'].startswith('COMPLETE'):continue
  refine=G.name.startswith('refine_');phase=G.name[len('refine_'):] if refine else G.name
  script='run_refinement_e.py' if refine else 'run_optimization.py' if kind=='Qwen' and phase=='screen' else 'run_optimization_v2.py' if kind=='Qwen' else 'run_optimization_v3.py' if kind=='InternVL' else 'run_optimization_v4.py'
  commands.append(env+' -B '+str(A/'code'/script)+' --kind '+kind+' --phase '+phase)
commands+=['# Existing output directories deliberately reject overwrite. For reproduction, restore the preserved source snapshot and original resource/config bindings to a NEW workspace and update A consistently; never reexecute over completed directories.',
 '# CPU final validation: /environment/miniconda3/bin/python -B '+str(A/'code/validate_all_v1.py'),
 '# No model weights/images are in this artifact. Restore model/data separately via trusted receipts and the previous environment restore template.']
(A/'run_commands.txt').write_text('\n'.join(commands)+'\n')
(A/'reproduction_readme.md').write_text("""# 复现说明

本目录保留服务器完整实验证据和独立源码。正式结果为原30固定输入、两独立Session、32输出Token；不包含全量质量复测。原始输入Gold与模型Inputs分开，Tensor SHA核对见各阶段input_validation.json。

先读protocol.json及protocol_refinement_addendum.json，再读cross_model_overhead_audit.md、optimization_changes.md、numerical_equivalence_report.md和final_report_zh.md。真实执行命令及Python环境见run_commands.txt；源码版本绑定见execution_runner_versions.json、source_manifest.json及changed_source_manifest.json。

opt_b/c/d/e是独立版本，不覆盖冻结实现。失败候选、零请求启动失败和仪器元数据修正均保留；失败数据不算正式性能。生产包为最终权威源码，初始build脚本仅为过程记录，勿把不含后续补充修正的单个脚本直接覆盖生产包。

原始冻结源按绝对路径保存在frozen_sources/下。新实例应恢复本地SSD模型、图片、原环境及原Runtime resources（input manifest、image mapping、Lt bridge），在新唯一输出路径复制包并一致更新入口A/W/R/O路径和源码清单；不能在既有目录重新覆盖运行。模型Revision与可信完整回执在model_manifest.json，HF及InternVL实际代码另存runtime_source_snapshots/。

Profiler为单个固定中等输入的32Token诊断，只有同步请求边界内事件参与计数。它的墙钟时间不进入正式效率CSV；GPU分配事件与峰值是实测，不把理论KV、总显存或Mask当Persistent KV。InternVL Vision编码位于Language Forward Scope外，但仍计入E2E总窗口。

未推送GitHub，未备份到Windows。服务器完整持久化目录由外部svh_cross_model_physical_optimization_delivery.json回执确认。SHA256SUMS包含evidence_manifest.json，不自包含自身；evidence_manifest不列自身及SHA256SUMS，以避免循环哈希。
""")
paths=[p for p in sorted(A.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.tmp','-wal','-shm')) and p.name not in ['evidence_manifest.json','SHA256SUMS']]
assert all(not p.is_symlink() for p in paths)
def item(p):return dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=sha(p))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(item,paths))
save(A/'evidence_manifest.json',dict(status='VALIDATED_COMPLETE',files=entries,total_bytes=sum(x['bytes'] for x in entries),excluded=['__pycache__','*.tmp','SQLite WAL/SHM after successful checkpoint','manifest self-hashes as documented'],models_or_original_images_included=False,git_pushed=False))
sha_entries=entries+[item(A/'evidence_manifest.json')]
(A/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in sha_entries))
all_entries=sha_entries+[item(A/'SHA256SUMS')]
required=sum(x['bytes'] for x in all_entries)
assert shutil.disk_usage(N.parent).free>required+1024**3
assert not N.exists(),('DESTINATION_EXISTS_NO_OVERWRITE',str(N))
N.mkdir()
print('COPY_BEGIN',len(all_entries),required,flush=True)
for x in all_entries:
 source=A/x['path'];dest=N/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
def verify(x):
 p=N/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p);return True
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=sum(pool.map(verify,all_entries))
receipt=dict(status='PASS',destination=str(N),source=str(A),files_verified=verified,total_bytes=required,evidence_manifest_sha256=sha(N/'evidence_manifest.json'),SHA256SUMS_sha256=sha(N/'SHA256SUMS'),formal_requests=json.loads((N/'validation.json').read_text())['formal_requests'],GPU_release=json.loads((N/'resource_release.json').read_text()),git_pushed=False,windows_backup=False,completed_time_ns=time.time_ns(),no_historical_files_overwritten=True)
save(N.parent/'svh_cross_model_physical_optimization_delivery.json',receipt)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
