"""Two resumable Windows-to-local-disk uploads; verify every remote archive."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import os,sys,json,time,shlex
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.gpu_smoke_tools'))
import paramiko
H=Path(__file__).resolve().parent;plan=json.loads((H/'datasets_restore_plan.json').read_text(encoding='utf-8'))
REMOTE=plan['local_data_root']+'/backup_packages';OUT=plan['output']
def connect():
 c=paramiko.SSHClient();c.load_host_keys(str(H/'known_hosts'));c.set_missing_host_key_policy(paramiko.RejectPolicy())
 c.connect('workspace.featurize.cn',port=20520,username='featurize',password=os.environ['ATE_20520_SSH_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=25)
 c.get_transport().set_keepalive(20);return c
def run(c,command):
 _,o,e=c.exec_command(command);out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
 if code:raise RuntimeError((code,out[-1500:],err[-1500:]))
 return out
def remote_sha(c,p):return run(c,'sha256sum -- '+shlex.quote(p)).split()[0]
master=connect()
try:
 print(run(master,(H/'bootstrap.sh').read_text(encoding='utf-8')),flush=True)
 with master.open_sftp() as s:
  try:
   with s.open(OUT+'/datasets_restore_plan.json','rb') as f:assert json.loads(f.read())==plan,'Refuse changing an existing restore plan'
  except FileNotFoundError:s.put(str(H/'datasets_restore_plan.json'),OUT+'/datasets_restore_plan.json')
 progress={'status':'TRANSFERRING','endpoint':plan['endpoint'],'expected_packages':len(plan['packages']),'expected_bytes':sum(e['bytes'] for e in plan['packages']),
   'completed':[],'failed':[],'started_at_UTC':datetime.now(timezone.utc).isoformat()}
 def save():
  raw=(json.dumps(progress,ensure_ascii=False,indent=2)+'\n').encode();(H/'transfer_progress.json').write_bytes(raw)
  with master.open_sftp() as s:
   with s.open(OUT+'/transfer_progress.json.tmp','wb') as f:f.write(raw)
   s.posix_rename(OUT+'/transfer_progress.json.tmp',OUT+'/transfer_progress.json')
 save()
 def one(entry):
  for attempt in range(3):
   c=None
   try:
    c=connect();s=paramiko.SFTPClient.from_transport(c.get_transport(),window_size=64*1024**2,max_packet_size=1024**2)
    target=REMOTE+'/'+entry['name'];part=target+'.part';src=Path(entry['local']);assert src.stat().st_size==entry['bytes']
    try:
     existing=s.stat(target);assert existing.st_size==entry['bytes'] and remote_sha(c,target)==entry['sha256'],'Existing archive differs; preserve it'
     print('REUSED_VERIFIED_ARCHIVE',entry['name'],flush=True);return {'name':entry['name'],'bytes':entry['bytes'],'sha256':entry['sha256'],'status':'PASS','reused':True}
    except FileNotFoundError:pass
    try:offset=s.stat(part).st_size
    except FileNotFoundError:offset=0
    assert offset<=entry['bytes'];count=offset;last=time.monotonic()
    with src.open('rb') as f,s.file(part,'ab' if offset else 'wb') as dest:
     f.seek(offset);dest.set_pipelined(True)
     for b in iter(lambda:f.read(256*1024),b''):
      dest.write(b);count+=len(b)
      if time.monotonic()-last>20:print('SFTP_PROGRESS',entry['name'],count,'/',entry['bytes'],flush=True);last=time.monotonic()
    assert s.stat(part).st_size==entry['bytes'] and remote_sha(c,part)==entry['sha256'],'Archive transfer SHA256 mismatch'
    s.posix_rename(part,target);s.close();print('ARCHIVE_TRANSFER_VERIFIED',entry['name'],entry['bytes'],flush=True)
    return {'name':entry['name'],'bytes':entry['bytes'],'sha256':entry['sha256'],'status':'PASS','resumed_bytes':offset}
   except (OSError,paramiko.SSHException) as e:
    if attempt==2:raise
    print('RETRY_RESUMABLE_UPLOAD',entry['name'],type(e).__name__,flush=True);time.sleep(3)
   finally:
    if c:c.close()
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(one,e):e for e in plan['packages']}
  for job in as_completed(jobs):
   e=jobs[job]
   try:progress['completed'].append(job.result())
   except Exception as exc:progress['failed'].append({'name':e['name'],'error':repr(exc)})
   progress['updated_at_UTC']=datetime.now(timezone.utc).isoformat();save()
 progress['status']='ALL_ARCHIVES_SHA256_VERIFIED' if not progress['failed'] else 'PARTIAL'
 progress['finished_at_UTC']=datetime.now(timezone.utc).isoformat();save()
 print('TRANSFER_FINAL',progress['status'],len(progress['completed']),'/',len(plan['packages']),flush=True)
 if progress['failed']:raise RuntimeError(progress['failed'])
finally:master.close()
