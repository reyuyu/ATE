"""Pinned HF recovery; streaming downloads and full content checks, no torch."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os,requests,threading,time,struct

O=Path('/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000')
M=Path('/home/featurize/ate-local-models/InternVL3-8B')
REV='853e3a797a661694b1b8ece0cb72dc2b23e3dac9';REPO='OpenGVLab/InternVL3-8B'
lock=threading.Lock();progress={}
def save(n,v):
 p=O/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def note(n,count,total):
 with lock:
  progress[n]=dict(bytes=count,expected=total)
  save('model_progress.json',dict(status='DOWNLOADING',revision=REV,files=progress,bytes=sum(v['bytes'] for v in progress.values()),at=datetime.now(timezone.utc).isoformat()))
def verify(p,info):
 assert p.stat().st_size==info['size'],p.name+' size'
 h=sha(p)
 if info.get('lfs'):assert h==info['lfs']['sha256'],p.name+' LFS SHA256'
 else:assert hashlib.sha1(b'blob '+str(info['size']).encode()+b'\0'+p.read_bytes()).hexdigest()==info['blobId'],p.name+' Git blob'
 old=previous['files'].get(p.name)
 if old:assert h==old['sha256'],p.name+' historical SHA256'
 return dict(bytes=info['size'],sha256=h,git_blob=info['blobId'],lfs_sha256=info.get('lfs',{}).get('sha256'))
def download(info):
 n=info['rfilename'];p=M/n;part=M/(n+'.part')
 if p.exists():v=verify(p,info);note(n,info['size'],info['size']);return n,dict(v,reused=True)
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0;assert offset<=info['size']
   if offset<info['size']:
    url='https://huggingface.co/'+REPO+'/resolve/'+REV+'/'+n+'?download=true&ate_recovery='+str(int(time.time()))
    with requests.get(url,headers={'Range':'bytes='+str(offset)+'-'} if offset else {},stream=True,timeout=(30,90)) as r:
     r.raise_for_status()
     if offset and r.status_code!=206:offset=0
     if offset:assert r.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
     with part.open('ab' if offset else 'wb') as f:
      count=offset;last=time.monotonic();note(n,count,info['size'])
      for block in r.iter_content(4*1024**2):
       if block:f.write(block);count+=len(block)
       if time.monotonic()-last>10:note(n,count,info['size']);last=time.monotonic()
   v=verify(part,info);part.replace(p);note(n,info['size'],info['size']);print('VERIFIED',n,info['size'],flush=True);return n,dict(v,reused=False)
  except AssertionError:raise
  except Exception as e:
   print('TRANSPORT_RETRY',n,attempt+1,type(e).__name__,getattr(getattr(e,'response',None),'status_code',None),flush=True)
   if attempt==3:raise RuntimeError('HF download failed: '+n+'; '+type(e).__name__) from None
   time.sleep(3)
try:
 previous={'files':{}}
 r=requests.get('https://huggingface.co/api/models/'+REPO+'/revision/'+REV+'?blobs=true',timeout=30);r.raise_for_status();meta=r.json();assert meta['sha']==REV
 save('official_model_api.json',meta);files=[i for i in meta['siblings'] if '/' not in i['rfilename']]
 assert len([i for i in files if i['rfilename'].endswith('.safetensors')])==4
 M.mkdir(parents=True,exist_ok=True);assert os.statvfs(M).f_bavail*os.statvfs(M).f_frsize>sum(i['size'] for i in files)+5*1024**3
 results={};failures=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  todo={pool.submit(download,i):i['rfilename'] for i in files}
  for future in as_completed(todo):
   try:n,v=future.result();results[n]=v
   except Exception as e:failures.append(dict(file=todo[future],error=str(e)))
 assert not failures,failures
 index=json.loads((M/'model.safetensors.index.json').read_text());tensors={};total=0;dtypes=set()
 for name in sorted(set(index['weight_map'].values())):
  with (M/name).open('rb') as f:
   length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
  for key,v in header.items():
   if key=='__metadata__':continue
   assert key not in tensors and index['weight_map'][key]==name;tensors[key]=name;dtypes.add(v['dtype']);total+=v['data_offsets'][1]-v['data_offsets'][0]
 assert tensors==index['weight_map'] and total==index['metadata']['total_size'] and dtypes=={'BF16'}
 save('model_verification.json',dict(status='PASS',model=REPO,revision=REV,path=str(M),files=results,bytes=sum(v['bytes'] for v in results.values()),all_content_SHA_verified=True,index_header_exact_match=True,tensors=len(tensors),dtypes=sorted(dtypes),weight_loading=False,GPU_inference=False))
 save('model_progress.json',dict(status='PASS',revision=REV,bytes=sum(v['bytes'] for v in results.values()),files=progress))
 print('MODEL_RECOVERY_PASS',len(results),flush=True)
except BaseException as e:
 save('model_failure.json',dict(status='BLOCKED',error=repr(e),files=progress));print('MODEL_BLOCKED',type(e).__name__,str(e),flush=True);raise
