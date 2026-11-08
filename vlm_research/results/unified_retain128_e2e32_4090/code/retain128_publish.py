"""CPU-only evidence finalization and bounded Git publication, no model load."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile,urllib.request,time
from datetime import datetime,timezone

R=Path('/home/featurize/work/ATE')
O=R/'vlm_research/results/unified_retain128_e2e32_4090'
A=R/'vlm_research/external_baselines/retain128_compatibility'
BASE='5ee2a0dbe78df336229b55fed14a4a6d0d880bed'
BRANCH='codex/unified-retain128-e2e32-4090'
REPO='reyuyu/ATE'
UUID='GPU-724fa9c3-d33a-df3f-e349-533cb4634a53'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def git(*args,env=None,raw=False):
 p=subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false',*args],cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,check=False)
 if p.returncode:raise RuntimeError('git '+str(args[0])+' failed: '+p.stderr.decode(errors='replace')[-2000:])
 return p.stdout if raw else p.stdout.decode().strip()
def entries():
 out=[]
 for root in (O,A):
  for p in sorted(root.rglob('*')):
   if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp'):continue
   rel=p.relative_to(R).as_posix();out.append(dict(path=rel,absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 return out
def should_publish(f):
 p=Path(f['path']);parts=p.parts
 if f['bytes']>8*1024*1024:return False
 if any(n in parts for n in ['preformal_failures','references','__pycache__']):return False
 if p.name in ['checkpoint.sqlite','checkpoint.sqlite-shm','checkpoint.sqlite-wal','run.lock','launch.json','github_upload_receipt.json','publication_index','publication_commit_message.txt']:return False
 if p.suffix in ('.sqlite','.html','.lock','.so'):return False
 return True
def finalize():
 v=read(O/'validation.json');assert v['status']=='PASS_AUDITED_CONDITIONS_PARTIAL_REQUESTED_METHODS' and v['formal_requests']==150
 assert v['GPU_released'] and v['quality_requests']==v['OCRBench_requests']==0
 release=read(O/'gpu_release.json');assert release['compute_processes']=='' and release['GPU'].startswith(UUID)
 assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()==''
 for p in (O/'.gitattributes',A/'.gitattributes'):p.write_text('* -text\n')
 # The first harness-only failure happened before any formal requests. Keep it
 # separate, including original code/protocol/checkpoint, and disclose it.
 failure=O/'preformal_failures/attempt001/progress.json'
 if failure.exists():
  x=read(failure);assert x['formal_completed']==0 and x['status']=='BLOCKED'
  save('preformal_failure_summary.json',dict(formal_requests=0,cause='Smoke observer expected-layout table lacked BridgeNative alias; KeyError before formal timing',resolution='BridgeNative and disabled V2Drop map to Native layout in the new observer; no algorithm or timing change',original_progress=x,full_original_evidence_retained=str(failure.parent)))
 report=O/'report_zh.md';s=report.read_text()
 url='https://github.com/'+REPO+'/blob/'+BRANCH+'/vlm_research/results/unified_retain128_e2e32_4090/report_zh.md'
 section='\n## GitHub 发布\n\n分支：`'+BRANCH+'`。报告链接：'+url+' 。完整Commit及远端读取校验记录保存在工作盘 `github_upload_receipt.json`；该回执在提交后生成，不属于自身所指提交。未上传大文件及其SHA256见 `github_publication_manifest.json`。\n\n首次Smoke因检查器漏列BridgeNative标签而停止，正式请求为0；修复仅作用于新检查器，失败原件保留在 `preformal_failures/attempt001/`，摘要随提交发布。\n'
 if '## GitHub 发布' not in s:report.write_text(s+section)
 # A receipt cannot honestly be recorded before push; that status is pending
 # here, with post-push evidence in a separate local-only receipt.
 v['GitHub_upload']='PENDING_AT_COMMIT; consult post-commit github_upload_receipt.json on persistent disk'
 save('validation.json',v)
 files=entries()
 self_names={'evidence_manifest.json','SHA256SUMS','absolute_file_paths.txt','github_publication_manifest.json','github_upload_receipt.json'}
 material=[f for f in files if Path(f['path']).name not in self_names]
 save('evidence_manifest.json',dict(schema=1,hash='SHA256',self_referential_manifests_excluded=sorted(self_names),files=material))
 (O/'SHA256SUMS').write_text(''.join(f['sha256']+'  '+f['path']+'\n' for f in material))
 (O/'absolute_file_paths.txt').write_text('\n'.join(sorted({f['absolute_path'] for f in entries()}|{str(O/n) for n in self_names}))+'\n')
 files=entries();pub=[f for f in files if should_publish(f) and Path(f['path']).name not in ['github_publication_manifest.json','github_upload_receipt.json']]
 omit=[f for f in files if not should_publish(f)]
 save('github_publication_manifest.json',dict(repository=REPO,branch=BRANCH,parent_commit=BASE,formal_requests=150,requested_max=180,FastV='BLOCKED_EVIDENCE_NO_FORMAL_RUN',published_files=pub,manifest_itself='Included; excluded from self-hashing',persistent_disk_only=omit,post_commit_receipt='github_upload_receipt.json stays on persistent disk; created after commit',Windows_backup=False,models_images_credentials_uploaded=False))
 pub.append(dict(path=(O/'github_publication_manifest.json').relative_to(R).as_posix(),bytes=(O/'github_publication_manifest.json').stat().st_size,sha256=sha(O/'github_publication_manifest.json')))
 assert all(f['bytes']<8*1024*1024 for f in pub)
 print(json.dumps(dict(status='EVIDENCE_READY',files=len(pub),bytes=sum(f['bytes'] for f in pub),omitted=len(omit))))

def publish():
 manifest=read(O/'github_publication_manifest.json');files=manifest['published_files'];paths=[f['path'] for f in files]
 paths.append((O/'github_publication_manifest.json').relative_to(R).as_posix())
 assert len(paths)==len(set(paths))
 for f in files:assert sha(R/f['path'])==f['sha256'],f['path']
 # Conservative bounded scan; do not print secrets or authentication values.
 for name in paths:
  p=R/name;assert p.is_file() and not p.is_symlink()
  assert all(n not in p.parts for n in ['.ssh','.git']) and p.suffix not in ('.safetensors','.bin','.pt','.pth')
  if p.suffix in ('.py','.json','.md','.csv','.txt','.log'):
   content=p.read_bytes()
   assert (b'-----BEGIN '+b'OPENSSH PRIVATE KEY'+b'-----') not in content
   assert (b'-----BEGIN '+b'RSA PRIVATE KEY'+b'-----') not in content
 assert git('status','--porcelain','-uno')=='','Tracked checkout changed'
 git('fetch','--no-tags','origin','refs/heads/codex/unified-e2e32-eight-methods-4090')
 assert git('rev-parse','FETCH_HEAD')==BASE
 remote_main_before=git('ls-remote','origin','refs/heads/main');checkout_head=git('rev-parse','HEAD')
 remote_before=git('ls-remote','origin','refs/heads/'+BRANCH)
 assert remote_before=='','Refusing to overwrite existing remote branch'
 work=Path(tempfile.mkdtemp(prefix='ate-retain128-publish-',dir='/home/featurize'))
 env=os.environ.copy();env['GIT_INDEX_FILE']=str(work/'index')
 git('read-tree',BASE,env=env)
 for start in range(0,len(paths),80):git('add','--sparse','-f','--',*paths[start:start+80],env=env)
 changed=git('diff','--cached','--name-status',BASE,env=env).splitlines()
 assert len(changed)==len(paths) and all(l.startswith('A\t') for l in changed)
 assert {l.split('\t')[1] for l in changed}==set(paths)
 for name in paths:
  data=git('show',':'+name,env=env,raw=True);assert hashlib.sha256(data).hexdigest()==sha(R/name),name
 tree=git('write-tree',env=env)
 msg=work/'message.txt';msg.write_text('Record Retain-128 E2E@32 efficiency audit on one RTX 4090\n\nNative and four admissible external configurations: 150 fixed requests.\nFastV remains blocked without a verified average128 preset.\nDisclose SparseVLM recycling, ApET released-parameter transfer and DUET C+all.\nPhysical KV, common timing, source/input hashes and Native drift checks included.\nNo quality benchmark or Windows backup.\n')
 env.update(GIT_AUTHOR_NAME='Codex',GIT_AUTHOR_EMAIL='codex@openai.com',GIT_COMMITTER_NAME='Codex',GIT_COMMITTER_EMAIL='codex@openai.com')
 commit=git('commit-tree',tree,'-p',BASE,'-F',str(msg),env=env)
 local=git('show-ref','--verify','--hash','refs/heads/'+BRANCH) if subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+BRANCH],cwd=R).returncode==0 else ''
 assert local in ('',commit),'Refusing to overwrite unrelated local branch'
 if not local:git('update-ref','refs/heads/'+BRANCH,commit,'0'*40)
 print(json.dumps(dict(status='COMMIT_READY',commit=commit,tree=tree,files=len(paths))),flush=True)
 pushed=git('-c','pack.threads=1','-c','pack.windowMemory=16m','-c','pack.deltaCacheSize=16m','-c','pack.window=5','-c','pack.depth=10','push','origin','refs/heads/'+BRANCH)
 remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==commit
 assert git('rev-parse',commit+'^{tree}')==tree
 # Retrieve every uploaded blob from GitHub by immutable commit to establish
 # existence and byte equality, not merely a successful local push message.
 req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/git/trees/'+tree+'?recursive=1',headers={'User-Agent':'ATE-reproduction-evidence-verifier'})
 with urllib.request.urlopen(req,timeout=60) as f:remote_tree=json.load(f)
 assert not remote_tree.get('truncated',False)
 remote_blobs={v['path']:v['sha'] for v in remote_tree['tree'] if v['type']=='blob'}
 for name in paths:assert remote_blobs[name]==git('rev-parse',commit+':'+name),name
 report=(O/'report_zh.md').relative_to(R).as_posix()
 raw='https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+report
 req=urllib.request.Request(raw,headers={'User-Agent':'ATE-reproduction-evidence-verifier'})
 with urllib.request.urlopen(req,timeout=60) as f:report_bytes=f.read()
 assert hashlib.sha256(report_bytes).hexdigest()==sha(O/'report_zh.md')
 assert git('ls-remote','origin','refs/heads/main')==remote_main_before and git('rev-parse','HEAD')==checkout_head
 receipt=dict(status='PASS_REMOTE_VERIFIED',at=datetime.now(timezone.utc).isoformat(),repository=REPO,branch=BRANCH,commit=commit,tree=tree,parent=BASE,remote_ref_exact_match=True,all_published_remote_blob_ids_verified=True,published_file_count=len(paths),report_SHA256=sha(O/'report_zh.md'),raw_report_bytes_match=True,report_url='https://github.com/'+REPO+'/blob/'+commit+'/'+report,main_unchanged=True,working_checkout_unchanged=True,Windows_backup=False)
 save('github_upload_receipt.json',receipt)
 assert git('status','--porcelain','-uno')==''
 print(json.dumps(receipt),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args()
 if a.publish:publish()
 else:finalize()
