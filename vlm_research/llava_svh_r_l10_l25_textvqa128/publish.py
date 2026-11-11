from common import *
import subprocess,concurrent.futures,urllib.request,urllib.error,time,re
S=read(RUN/'state.json');ROOT=Path(S['root']);DEST=ROOT/S['result']
SSH='ssh -i /home/featurize/.ssh/ate_deep_exit_14890_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o HostName=github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15'
os.environ['GIT_SSH_COMMAND']=SSH
def git(*args):
 p=subprocess.run(['git','-c','pack.threads=1','-c','pack.windowMemory=32m','-c','pack.deltaCacheSize=16m','-c','core.autocrlf=false','-C',str(ROOT),*args],capture_output=True,text=True,timeout=180)
 if p.returncode:raise RuntimeError(str(args)+': '+p.stderr[:3000])
 return p.stdout.strip()
def remote_head():return git('ls-remote','origin','refs/heads/'+S['branch']).split()[0]
def main():
 assert git('branch','--show-current')==S['branch']
 git('merge-base','--is-ancestor',S['parent'],'HEAD')
 assert set(git('diff','--cached','--name-only').splitlines()) <= set(x['path'] for x in read(RUN/'publish_plan.json')['files'])
 plan=read(RUN/'publish_plan.json');paths=[x['path'] for x in plan['files']]
 for x in plan['files']:
  path=x['path'];p=ROOT/path;assert path.startswith(S['pkg']+'/') or path.startswith(S['result']+'/')
  assert 'server_only' not in p.parts and p.stat().st_size==x['bytes']<10*1024**2 and sha(p)==x['sha256']
  raw=p.read_bytes()
  assert b'BEGIN '+b'OPENSSH PRIVATE'+b' KEY' not in raw and b'BEGIN '+b'RSA PRIVATE'+b' KEY' not in raw
  assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}',raw),path
 git('add','-f','--',*paths);staged=git('diff','--cached','--name-only').splitlines();assert set(staged)<=set(paths)
 if staged:git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Evaluate LLaVA SVH-R L10-L25 on frozen TextVQA128')
 commit=git('rev-parse','HEAD')
 receipt=dict(status='LOCAL_COMMIT_CREATED',branch=S['branch'],commit=commit,parent=S['parent'],files=len(paths),protocol_sha256=sha(DEST/'protocol.json'),source_manifest_sha256=sha(DEST/'source_manifest.json'),server_data_preserved=True)
 write(DEST/'server_only/git_publish_receipt.json',receipt)
 check=git('ls-remote','origin','refs/heads/'+S['branch']).split()
 if not check or check[0]!=commit:git('push','origin','HEAD:refs/heads/'+S['branch'])
 assert remote_head()==commit
 def verify(x):
  url='https://raw.githubusercontent.com/reyuyu/ATE/'+commit+'/'+x['path']
  for attempt in range(3):
   try:
    req=urllib.request.Request(url,headers={'User-Agent':'ATE-evidence-audit'})
    with urllib.request.urlopen(req,timeout=60) as f:data=f.read()
    assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'],x['path']
    return dict(**x,remote_Git_blob=git('rev-parse',commit+':'+x['path']),verification='independent GitHub immutable-Commit HTTPS bytes')
   except (OSError,urllib.error.URLError):
    if attempt==2:raise
    time.sleep(2*(attempt+1))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:verified=list(pool.map(verify,plan['files']))
 assert remote_head()==commit and not git('diff','HEAD','--',*paths)
 receipt.update(status='PUSHED_REMOTE_ALL_FILES_VERIFIED',remote_head=commit,verified_files=verified,branch_url='https://github.com/reyuyu/ATE/tree/'+S['branch'],commit_url='https://github.com/reyuyu/ATE/commit/'+commit,report_url='https://github.com/reyuyu/ATE/blob/'+commit+'/'+S['result']+'/report_zh.md',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(DEST/'server_only/git_publish_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='verified_files'},ensure_ascii=False))
if __name__=='__main__':main()
