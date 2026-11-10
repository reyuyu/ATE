"""Publish only audited lightweight pilot evidence. No PR, main change, or force push."""
import hashlib,json,os,re,shutil,subprocess,time,urllib.request
from pathlib import Path
from .common import *
BRANCH='codex/internvl3-svh-physical-d-pilot-3090'
SSH='ssh -i /home/featurize/.ssh/ate_github_internvl3090_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20'
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def gitrun(*args):subprocess.run(['git','-C',str(ROOT),*args],check=True)
def remote(*args):return git('-c','core.sshCommand='+SSH,*args)
def main():
 assert read(RUN/'validation.json')['artifact_integrity']=='PASS'
 assert read(RUN/'persistence_verification.json')['status']=='PASS'
 assert read(RUN/'gpu_final_release.json')['own_GPU_released']
 assert git('branch','--show-current')==BRANCH
 package=ROOT/'vlm_research/internvl3_svh_physical_d_pilot'
 # Publication code is also part of the lightweight source delivery.
 uploaded=[x for x in package.iterdir() if x.is_file() and x.suffix in ('.py','.md')]
 server_only=[]
 exclude_dirs={'quality_checkpoints','efficiency_checkpoints'}
 for f in OUT.rglob('*'):
  if not f.is_file():continue
  rel=f.relative_to(OUT)
  heavy=any(p in exclude_dirs or p=='__pycache__' for p in rel.parts) or f.suffix in ('.log','.tmp','.pyc') or f.name in ['active.json','launch.json','continuation_launch.json','progress.json','gpu_release.json']
  if heavy or f.stat().st_size>8*1024*1024:
   server_only.append(f);continue
  if f.name not in ['artifact_manifest.json','SHA256SUMS','evidence_manifest.json','github_push_receipt.json']:uploaded.append(f)
 assert uploaded and all(f.stat().st_size<8*1024*1024 for f in uploaded)
 # Every file is inspected for private-key material and common real credential formats.
 for f in uploaded:
  if f.suffix in ('.json','.csv','.py','.md'):
   raw=f.read_bytes()
   assert not re.search(rb'(?m)^-----BEGIN (?:OPENSSH|RSA) PRIVATE KEY-----\r?\n',raw)
   assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}',raw)
 server_hashes={str(f.relative_to(OUT)):dict(sha256=sha(f),bytes=f.stat().st_size,absolute_path=str(f)) for f in server_only}
 phases={
  'initial_strict_failed':dict(protocol_sha256=sha(OUT/'engineering_attempts/attempt01_frozen_gate_failure/protocol.json'),source_commit='d98e07420d5ec949089db100607ec1a7c86486b4',status='FAILED',formal_requests=0),
  'user_authorized_continuation':dict(protocol_sha256=sha(OUT/'protocol.json'),source_commit=read(OUT/'environment.json')['source_commit'],status='COMPLETE_WITH_NUMERICAL_WAIVER',quality_requests=360,efficiency_requests=450)}
 evidence=dict(status='COMPLETE_WITH_NUMERICAL_WAIVER',GPU_UUID=GPU,model_revision=REV,phases=phases,
  input_manifest_sha256=sha(OUT/'sample_manifest.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),
  independent_validation_sha256=sha(OUT/'validation.json'),server_persistence_sha256=sha(OUT/'persistence_verification.json'))
 save(OUT/'evidence_manifest.json',evidence);uploaded.append(OUT/'evidence_manifest.json')
 artifact=dict(status='COMPLETE_WITH_NUMERICAL_WAIVER',strict_numerical_gate='FAILED',branch=BRANCH,
  GitHub_evidence_commit_sha=None,GitHub_publication_commit='Resolved by remote branch head and server-only github_push_receipt.json; a commit cannot contain its own hash.',
  source_commit=read(OUT/'environment.json')['source_commit'],model_revision=REV,GPU_UUID=GPU,
  protocol_sha256=sha(OUT/'protocol.json'),input_manifest_sha256=sha(OUT/'sample_manifest.json'),phases=phases,
  uploaded_files={str(f.relative_to(ROOT)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(uploaded)},
  server_only_files=server_hashes,
  self_reference_exclusions=['artifact_manifest.json','SHA256SUMS'],
  later_server_only_receipt=str(OUT/'github_push_receipt.json'),
  all_original_checkpoints_preserved=True)
 save(OUT/'artifact_manifest.json',artifact);uploaded.append(OUT/'artifact_manifest.json')
 def checksums():
  text=''.join(sha(f)+'  '+str(f.relative_to(ROOT))+'\n' for f in sorted(uploaded))
  (OUT/'SHA256SUMS').write_text(text)
 checksums();upload_paths=uploaded+[OUT/'SHA256SUMS']
 paths=[str(f.relative_to(ROOT)) for f in upload_paths]
 gitrun('add','--sparse','-f','--',*paths)
 staged=git('diff','--cached','--name-status').splitlines()
 assert staged
 allowed=('vlm_research/internvl3_svh_physical_d_pilot/','vlm_research/results/internvl3_svh_physical_d_pilot_3090/')
 assert all(line.split('\t')[-1].startswith(allowed) and not line.startswith('D') for line in staged)
 gitrun('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Publish InternVL3 Physical-D paired pilot with numerical waiver disclosed')
 evidence_commit=git('rev-parse','HEAD')
 artifact['GitHub_evidence_commit_sha']=evidence_commit
 save(OUT/'artifact_manifest.json',artifact);checksums()
 gitrun('add','--sparse','-f','--',str((OUT/'artifact_manifest.json').relative_to(ROOT)),str((OUT/'SHA256SUMS').relative_to(ROOT)))
 gitrun('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Bind immutable pilot evidence commit and final checksums')
 head=git('rev-parse','HEAD')
 try:
  gitrun('-c','core.sshCommand='+SSH,'-c','pack.threads=1','-c','pack.windowMemory=32m','-c','pack.deltaCacheSize=16m','push','origin','HEAD:refs/heads/'+BRANCH)
  remote_head=remote('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote_head==head
 except BaseException as e:
  save(OUT/'github_push_receipt.json',dict(status='FAILED',local_commit=head,evidence_commit=evidence_commit,error=repr(e),branch=BRANCH,files_preserved=True));raise
 # Remote subtree blob identities and selected actual raw contents must agree.
 tree_verified=0
 for prefix in ['vlm_research/internvl3_svh_physical_d_pilot','vlm_research/results/internvl3_svh_physical_d_pilot_3090']:
  tree_sha=git('rev-parse',head+':'+prefix)
  url='https://api.github.com/repos/reyuyu/ATE/git/trees/'+tree_sha+'?recursive=1'
  req=urllib.request.Request(url,headers={'User-Agent':'ATE-pilot-evidence-verification'})
  data=json.loads(urllib.request.urlopen(req,timeout=60).read());assert not data.get('truncated')
  actual={x['path']:x['sha'] for x in data['tree'] if x['type']=='blob'}
  expected={}
  for line in git('ls-tree','-r',head,'--',prefix).splitlines():
   meta,path=line.split('\t',1);expected[path[len(prefix)+1:]]=meta.split()[2]
  assert actual==expected,(prefix,len(actual),len(expected))
  tree_verified+=len(actual)
 raw_verified={}
 for rel in ['vlm_research/internvl3_svh_physical_d_pilot/engine.py','vlm_research/internvl3_svh_physical_d_pilot/continue_pilot.py',
  'vlm_research/results/internvl3_svh_physical_d_pilot_3090/report_zh.md',
  'vlm_research/results/internvl3_svh_physical_d_pilot_3090/accuracy_summary.csv',
  'vlm_research/results/internvl3_svh_physical_d_pilot_3090/efficiency_summary.csv',
  'vlm_research/results/internvl3_svh_physical_d_pilot_3090/validation.json',
  'vlm_research/results/internvl3_svh_physical_d_pilot_3090/artifact_manifest.json']:
  url='https://raw.githubusercontent.com/reyuyu/ATE/'+head+'/'+rel
  content=urllib.request.urlopen(url,timeout=60).read();h=hashlib.sha256(content).hexdigest();assert h==sha(ROOT/rel)
  raw_verified[rel]=h
 receipt=dict(status='PASS',branch=BRANCH,local_commit=head,remote_commit=remote_head,evidence_commit=evidence_commit,
  remote_tree_blobs_verified=tree_verified,raw_file_SHA256_verified=raw_verified,
  uploaded_lightweight_files=len(paths),uploaded_bytes=sum(f.stat().st_size for f in upload_paths),
  server_only_files=len(server_hashes),no_PR_created=True,no_main_change=True,no_force_push=True)
 save(OUT/'github_push_receipt.json',receipt);save(RUN/'github_push_receipt.json',receipt)
 # Copy exact final GitHub manifests/checksums back to the SSD without altering committed files.
 for name in ['artifact_manifest.json','evidence_manifest.json','SHA256SUMS']:
  shutil.copy2(OUT/name,RUN/name)
 print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
