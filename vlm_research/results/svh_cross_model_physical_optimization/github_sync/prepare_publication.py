from pathlib import Path
import json,hashlib,shutil,tarfile,gzip,re,subprocess,sys,collections
N=Path('/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization')
P=Path('/home/featurize/ate-experiments/svh-cross-model-github-sync-20261113')
REL=Path('vlm_research/results/svh_cross_model_physical_optimization');T=P/REL
BRANCH='codex/svh-cross-model-physical-optimization-3090'
BASE='e281754ef17b3accc09d6938b5b9f020048b2aec'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not T.exists(),'NO_OVERWRITE_EXISTING_PUBLICATION'
assert json.loads((N/'validation.json').read_text())['status']=='PASS_WITH_REJECTED_CANDIDATES'
receipt=json.loads((N.parent/'svh_cross_model_physical_optimization_delivery.json').read_text())
assert receipt['status']=='PASS' and receipt['formal_requests']==1200
assert subprocess.check_output(['git','-C',str(P),'branch','--show-current'],text=True).strip()==BRANCH
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==BASE
evidence=json.loads((N/'evidence_manifest.json').read_text())
original={r['path']:dict(r) for r in evidence['files']}
for name in ['evidence_manifest.json','SHA256SUMS']:
 p=N/name;original[name]=dict(path=name,bytes=p.stat().st_size,sha256=sha(p))
assert len(original)==receipt['files_verified']
selected=[]
for rel,r in original.items():
 q=Path(rel);top=q.parts[0];include=False
 if len(q.parts)==1:include=q.suffix in ['.json','.csv','.md','.txt','.log'] or q.name=='SHA256SUMS'
 elif top in ['opt_b','opt_c','opt_d','opt_e','code','source_diffs','runtime_source_snapshots','frozen_runtime_resources','frozen_reference_evidence']:include=True
 elif top in ['Qwen','InternVL','LLaVA']:include=q.suffix=='.json'
 if include:selected.append(rel)
frozen=[rel for rel in original if rel.startswith('frozen_sources/')]
assert len(frozen)==6390 and all(Path(r).suffix in ['.py','.cpp','.so'] for r in frozen)
all_frozen=list(frozen)
excluded_suffixes=[
 'external_baselines/apet/official_source/video-llava/scripts/eval_gpt_mmvet.py',
 'external_baselines/duet_vlm/official_source/scripts/videollava/eval_gpt_mmvet.py',
 'external_baselines/fastv/official_source/src/FastV/llava-hf/transformers/src/transformers/testing_utils.py',
 'external_baselines/fastv/official_source/src/transformers/src/transformers/testing_utils.py',
]
excluded_sources=[rel for rel in frozen if any(rel.endswith(s) for s in excluded_suffixes)]
assert len(excluded_sources)==4
required_paths=set()
for name in ['runtime_source_guard.json','additional_frozen_sources.json','runtime_dependency_manifest.json']:
 for r in json.loads((N/name).read_text()):required_paths.add(r['path'])
for rel in excluded_sources:
 original_absolute='/'+rel[len('frozen_sources/'):]
 assert original_absolute not in required_paths,('CANNOT_EXCLUDE_RUNTIME_DEPENDENCY',original_absolute)
frozen=[rel for rel in frozen if rel not in excluded_sources]
assert len(frozen)==6386

patterns=[
 ('PRIVATE_KEY',re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----')),
 ('GITHUB_TOKEN',re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})')),
 ('HF_TOKEN',re.compile(rb'hf_[A-Za-z0-9]{30,}')),
 ('AWS_ACCESS_KEY',re.compile(rb'AKIA[0-9A-Z]{16}')),
]
hardcoded=re.compile(rb'(?i)\b(?:password|passwd|api_key|access_token|github_token|hf_token)\s*[:=]\s*["\x27]([A-Za-z0-9_.-]{8,})["\x27]')
def placeholder(v):
 s=v.decode('ascii',errors='ignore').lower()
 return len(set(s))<=2 or any(w in s for w in ['your_','example','placeholder','password','dummy','test_token','insert_'])
issues=[]
for rel in sorted(set(selected+frozen)):
 p=N/rel;assert p.is_file() and not p.is_symlink(),rel
 assert p.stat().st_size==original[rel]['bytes'] and sha(p)==original[rel]['sha256'],('SOURCE_SHA_MISMATCH',rel)
 if p.suffix not in ['.gz','.so']:
  data=p.read_bytes()
  for label,pattern in patterns:
   matches=[m for m in pattern.finditer(data) if not placeholder(m.group())]
   if matches:issues.append(dict(path=rel,kind=label,count=len(matches)))
  for m in hardcoded.finditer(data):
   if not placeholder(m.group(1)):issues.append(dict(path=rel,kind='HARDCODED_CREDENTIAL_REQUIRES_REVIEW',line=data[:m.start()].count(b'\n')+1))
assert not issues,('SECRET_SCAN_REVIEW_REQUIRED',issues[:12])
T.mkdir(parents=True)
for rel in selected:
 p=N/rel;d=T/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);assert sha(d)==original[rel]['sha256']
archive_dir=T/'source_archives';archive_dir.mkdir();archive=archive_dir/'frozen_sources.tar.gz'
save(archive_dir/'excluded_unused_source_files.json',dict(original_frozen_sources=len(all_frozen),archived_sources=len(frozen),excluded=[dict(path=rel,sha256=original[rel]['sha256'],reason='Unused external test/API-evaluation script containing credential-like literals. Not in recorded runtime or additional dependency guards; original retained on server.') for rel in excluded_sources],runtime_dependencies_excluded=False))

with archive.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=6) as gz:
  with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
   for rel in sorted(frozen):
    p=N/rel;info=tarfile.TarInfo(rel);info.size=p.stat().st_size;info.mode=p.stat().st_mode&0o777
    info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
    with p.open('rb') as f:tar.addfile(info,f)
members=[dict(path=rel,bytes=original[rel]['bytes'],sha256=original[rel]['sha256']) for rel in sorted(frozen)]
save(archive_dir/'frozen_sources_members.json',dict(archive='frozen_sources.tar.gz',archive_sha256=sha(archive),files=members,source_file_count=len(members),uncompressed_bytes=sum(r['bytes'] for r in members)))
verifier=r'''from pathlib import Path
import argparse,json,tarfile,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');ap.add_argument('--extract-to',type=Path);args=ap.parse_args()
root=Path(__file__).resolve().parent;manifest=json.loads((root/'frozen_sources_members.json').read_text());archive=root/manifest['archive']
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['archive_sha256']
expected={r['path']:r for r in manifest['files']};dest=args.extract_to.resolve() if args.extract_to else None
if dest:
 assert not dest.exists(),'Extraction destination must be new; never overwrite frozen sources.'
 dest.mkdir(parents=True)
seen=set()
with tarfile.open(archive,'r|gz') as tar:
 for m in tar:
  assert m.isfile() and m.name in expected and m.name not in seen
  q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts;assert m.size==expected[m.name]['bytes']
  src=tar.extractfile(m);h=hashlib.sha256();out=None
  if dest:
   target=dest/q;target.parent.mkdir(parents=True,exist_ok=True);out=target.open('xb')
  try:
   while True:
    b=src.read(1024*1024)
    if not b:break
    h.update(b)
    if out:out.write(b)
  finally:
   if out:out.close()
  assert h.hexdigest()==expected[m.name]['sha256'],m.name
  seen.add(m.name)
assert seen==set(expected)
print(json.dumps(dict(status='PASS',source_files=len(seen),extracted_to=str(dest) if dest else None)))
'''
(archive_dir/'verify_and_extract_sources.py').write_text(verifier)
subprocess.run(['/environment/miniconda3/bin/python','-B',str(archive_dir/'verify_and_extract_sources.py'),'--verify-only'],check=True)
extra=T/'github_sync';extra.mkdir();shutil.copy2(Path(__file__),extra/'prepare_publication.py')
shutil.copy2(N.parent/'svh_cross_model_physical_optimization_delivery.json',extra/'server_delivery_receipt.json')
license_rows=[]
official_root=Path('/home/featurize/work/ATE/vlm_research/external_baselines')
for p in sorted(official_root.glob('*/official_source/*')):
 if p.is_file() and p.name.upper().startswith(('LICENSE','COPYING','NOTICE')):
  rel=Path('third_party_notices')/p.parent.parent.name/p.name;d=T/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
  license_rows.append(dict(original=str(p),published=str(rel),sha256=sha(p)))
save(T/'third_party_notices/manifest.json',dict(files=license_rows,scope='Existing official-source license notices; publication addition, not experiment code changes.'))
readme='''# SVH / SVH-R 跨模型物理优化：GitHub 交付

本分支在实验完成后，根据用户后续“推送github”的明确授权发布。原实验报告、源码、CSV、协议和验证文件保持原始字节及SHA256。原报告中的“未推送GitHub”、validation.json中的no_git_writes、原交付回执中的git_pushed=false描述实验结束时的状态；本次上传是后续独立操作，不改变实验事实。

- [正式中文报告](final_report_zh.md)
- [两轮正式效率均值/中位数](optimized_efficiency_32tokens.csv)
- [同Session同题配对比较](paired_comparisons.csv)
- [1200条正式逐请求数据及Token IDs](efficiency_per_request.csv)
- [真实逐层KV布局](kv_physical_audit.csv)
- [数值与完整输出回归](numerical_equivalence_report.md)
- [验证结果](validation.json)
- [上传/服务器保留文件清单](GITHUB_ARTIFACT_MANIFEST.json)

本轮完成1200个正式请求；全部使用GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6、原GQA15+TextVQA Paper-OCR15、固定32输出Token及31步真实Decode。Native/原版/优化版在各自Session内交错执行；全部慢样本和负结果保留。没有重跑质量Benchmark。失败的去补零D候选没有进入正式效率表，InternVL无稳定E2E收益的版本不建议替换默认原版。

## 代码与依赖

opt_b/、opt_c/、opt_d/、opt_e/保留完整独立适配包，code/保留实际Runner、CPU验证、统计、Profiler解析及全部版本。source_diffs/供直接审阅修改；execution_runner_versions.json绑定真实执行版本。初始build脚本是过程记录，最终独立包和SHA为权威，不要仅执行某个早期build脚本覆盖最终代码。

原始冻结源码清单含6390个文件，可发布源码归档source_archives/frozen_sources.tar.gz包含6386个文件。4个未参与本轮运行、且含凭据样式常量的外部测试/API评测脚本仅保留服务器原件，路径与原SHA见source_archives/excluded_unused_source_files.json；实际运行依赖未排除。归档不含模型权重、图片、数据集、密钥或系统缓存；保留原绝对路径对应的目录结构及逐文件SHA256。它们可以解压到新的普通目录，不会写入原始/home路径。

    cd vlm_research/results/svh_cross_model_physical_optimization
    python source_archives/verify_and_extract_sources.py --verify-only
    python source_archives/verify_and_extract_sources.py --extract-to /new/local/ate-source-restore

SHA校验原始字节，不以重新格式化代码或JSON代替原文件。runtime_source_snapshots/保存实际HF/模型代码，frozen_runtime_resources/保存必要小型资源；model_manifest.json记录模型Revision与可信完整回执，不含权重。官方已有许可证位于third_party_notices/。

原始复现上下文及运行命令见[reproduction_readme.md](reproduction_readme.md)、[run_commands.txt](run_commands.txt)、protocol.json及protocol_refinement_addendum.json。脚本保留冻结服务器的路径与GPU UUID断言；换实例复现时，应在新独立工作区恢复模型/图片/环境及归档源码，审查并统一更新路径与协议，不要覆盖已有输出，也不要把不同GPU的时间拼接。

[原18条件已发布证据](https://github.com/reyuyu/ATE/tree/108cbdc1e2f782b6b59bd6c87a44cd28868330e8/vlm_research/results/rtx3090_unified_efficiency_18conditions)是历史来源，最终冻结源码与SHA优先于旧分支默认文件。

## GitHub范围与服务器完整原件

GitHub包含轻量完整证据、所有逐请求/逐层CSV、数值与实际Audit Mode结果、环境记录、代码与冻结源码归档。新实验完整SQLite检查点、原始requests.jsonl、连续GPU/CPU/进程遥测及Profiler trace仅保存在服务器：

/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization/

GITHUB_ARTIFACT_MANIFEST.json逐文件区分GITHUB_FILE、GITHUB_SOURCE_ARCHIVE、SERVER_ONLY；不能声称SERVER_ONLY文件存在于GitHub。frozen_reference_evidence/内的历史逐题记录用于核对冻结原版输出，和本轮仅保存在服务器的新原始Checkpoint分开标记。

原SHA256SUMS和evidence_manifest.json对应服务器完整8604文件，故其中部分路径不会以原始文件形式出现在GitHub。检查本次GitHub发布内容请使用：

    sha256sum -c GITHUB_SHA256SUMS

GITHUB_SHA256SUMS不自包含自身，以避免循环哈希。归档内容通过独立解压流逐文件验SHA。GitHub Commit与远端核验回执另外保存在服务器结果目录旁的svh_cross_model_physical_optimization_github_sync_receipt.json，避免Commit自哈希循环。

本分支不修改main，不创建PR；本次同步不运行任何GPU任务。
'''
(T/'GITHUB_PUBLICATION.md').write_text(readme)
files_before=[p for p in sorted(T.rglob('*')) if p.is_file()]
assert all(not p.is_symlink() and p.stat().st_size<50*1024*1024 for p in files_before)
validation=dict(status='PASS',original_experiment_status='PASS_WITH_REJECTED_CANDIDATES',formal_requests=1200,copied_original_files=len(selected),all_copied_files_byte_identical=True,frozen_source_archive_members=len(members),excluded_unused_source_files=len(excluded_sources),all_recorded_runtime_dependencies_preserved=True,all_archive_member_SHA256_verified=True,source_archive_bytes=archive.stat().st_size,secret_scan='PASS_NO_CREDENTIAL_MATERIAL_FOUND',models_or_original_images_or_system_caches_included=False,full_checkpoints_and_profiler_traces='SERVER_ONLY',original_results_unmodified=True,no_GPU_work=True,branch=BRANCH,parent=BASE)
save(T/'github_sync/publication_validation.json',validation)
upload_rows=[dict(path=str(p.relative_to(T)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(T.rglob('*')) if p.is_file()]
direct=set(selected);routes=[]
for rel,r in sorted(original.items()):
 status='GITHUB_FILE' if rel in direct else 'GITHUB_SOURCE_ARCHIVE' if rel in frozen else 'SERVER_ONLY'
 routes.append(dict(path=rel,bytes=r['bytes'],sha256=r['sha256'],status=status,reason='Unused credential-like external test; server original retained' if rel in excluded_sources else None,archive='source_archives/frozen_sources.tar.gz' if status=='GITHUB_SOURCE_ARCHIVE' else None))
save(T/'GITHUB_ARTIFACT_MANIFEST.json',dict(repository='https://github.com/reyuyu/ATE',branch=BRANCH,source_server_directory=str(N),source_evidence_manifest_SHA256=receipt['evidence_manifest_sha256'],source_total_files=len(original),published_files=upload_rows,original_file_routes=routes,self_hash_excluded=True,post_push_commit_record='Server-side external GitHub sync receipt; no self-referential Commit field.'))
all_published=[dict(path=str(p.relative_to(T)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(T.rglob('*')) if p.is_file()]
(T/'GITHUB_SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in all_published))
all_published.append(dict(path='GITHUB_SHA256SUMS',bytes=(T/'GITHUB_SHA256SUMS').stat().st_size,sha256=sha(T/'GITHUB_SHA256SUMS')))
outside=P.parent/'svh-cross-model-publication-precommit-20261113.json'
save(outside,dict(status='READY_FOR_REVIEW',worktree=str(P),result_path=str(T),branch=BRANCH,parent=BASE,published_files=len(all_published),published_bytes=sum(r['bytes'] for r in all_published),max_file_bytes=max(r['bytes'] for r in all_published),archive_bytes=archive.stat().st_size,source_routes=dict(collections.Counter(r['status'] for r in routes)),files=all_published,validation=validation))
print(json.dumps({k:v for k,v in json.loads(outside.read_text()).items() if k not in ['files','validation']},ensure_ascii=False),flush=True)
