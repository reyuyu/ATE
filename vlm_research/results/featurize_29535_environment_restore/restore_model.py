"""Pinned, resumable file restore; no model deserialization or CUDA calls."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,threading,time,urllib.request
REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
O=Path(os.environ.get('ATE_RECOVERY_OUT','/home/featurize/work/ATE/vlm_research/results/featurize_29535_environment_restore'))
M=Path(os.environ.get('ATE_MODEL_DIR','/home/featurize/ate-local-models/llava-1.5-7b-hf'))
meta=json.loads((O/'hf_revision_manifest.json').read_text());assert meta['revision']==REV
M.mkdir(parents=True,exist_ok=True)
lock=threading.Lock();state={};started=time.time()
def atomic(path,value):
 t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def update(name,**kw):
 with lock:
  state.setdefault(name,{}).update(kw)
  atomic(O/'model_download_progress.json',dict(status='IN_PROGRESS',revision=REV,model_path=str(M),elapsed_seconds=round(time.time()-started),files=state))
def hasher(row):
 if 'lfs' in row:return hashlib.sha256()
 h=hashlib.sha1();h.update(('blob '+str(row['size'])+'\0').encode());return h
def expected(row):return row.get('lfs',{}).get('sha256',row['blobId'])
def check(path,row):
 if not path.exists() or path.stat().st_size!=row['size']:return False
 h=hasher(row)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()==expected(row)
def download(row):
 name=row['rfilename'];path=M/name;part=path.with_name(path.name+'.part');size=row['size']
 if check(path,row):update(name,status='VERIFIED_REUSED',bytes=size,expected=size);return
 if path.exists():raise RuntimeError('Existing final model file fails pinned hash: '+name)
 for attempt in range(1,5):
  try:
   offset=part.stat().st_size if part.exists() else 0
   if offset>size:raise RuntimeError('Partial file exceeds pinned length')
   if offset==size:
    if not check(part,row):raise RuntimeError('Completed partial hash mismatch')
    part.replace(path);update(name,status='VERIFIED',bytes=size,expected=size);return
   # Cache-busting avoids stale signed LFS redirects; commit and byte hashes stay pinned.
   base='https://huggingface.co' if attempt==1 else 'https://hf-mirror.com'
   url=base+'/llava-hf/llava-1.5-7b-hf/resolve/'+REV+'/'+name+'?download=true&ate_restore=29535-'+str(attempt)
   req=urllib.request.Request(url,headers={'User-Agent':'ATE-pinned-environment-restore/1','Range':'bytes='+str(offset)+'-'})
   with urllib.request.urlopen(req,timeout=60) as response:
    if offset and response.status!=206:offset=0
    if response.status==206:
     assert response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'),'Bad Range response'
    h=hasher(row)
    if offset:
     with part.open('rb') as f:
      for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    done=offset;last=time.monotonic();update(name,status='DOWNLOADING',bytes=done,expected=size,attempt=attempt)
    with part.open('ab' if offset else 'wb') as f:
     while True:
      b=response.read(2**20)
      if not b:break
      f.write(b);h.update(b);done+=len(b)
      if done>size:raise RuntimeError('File exceeds pinned length')
      if time.monotonic()-last>10:update(name,status='DOWNLOADING',bytes=done,expected=size,attempt=attempt);last=time.monotonic()
     f.flush();os.fsync(f.fileno())
    assert done==size,'Incomplete response'
    assert h.hexdigest()==expected(row),'Pinned file hash mismatch'
    part.replace(path);update(name,status='VERIFIED',bytes=done,expected=size,hash_algorithm='SHA256' if 'lfs' in row else 'git-blob-SHA1',hash=expected(row))
    print('VERIFIED',name,size,flush=True);return
  except Exception as e:
   update(name,status='RETRY' if attempt<4 else 'FAILED',error_type=type(e).__name__,http_status=getattr(e,'code',None),attempt=attempt)
   print('DOWNLOAD_RETRY',name,attempt,type(e).__name__,getattr(e,'code',''),flush=True)
   if attempt==4:raise
   time.sleep(min(5*attempt,15))
small=[r for r in meta['files'] if r['size']<10**8];large=[r for r in meta['files'] if r['size']>=10**8]
for row in small:download(row)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(download,large):pass
index=json.loads((M/'model.safetensors.index.json').read_text());shards=set(index['weight_map'].values())
assert shards=={r['rfilename'] for r in large}
assert all((M/n).is_file() for n in shards)
receipt=dict(status='VERIFIED',model=meta['model'],revision=REV,model_path=str(M),files=state,total_bytes=sum(r['size'] for r in meta['files']),source='Hugging Face fixed revision; primary huggingface.co first, hf-mirror.com fallback; all content pinned to existing trusted LFS SHA256/git blob hashes',all_LFS_SHA256_verified=True,all_small_git_blob_hashes_verified=True,no_GPU_load=True)
atomic(M/'.ate_revision.json',receipt);atomic(O/'model_integrity.json',receipt)
with lock:atomic(O/'model_download_progress.json',dict(receipt,elapsed_seconds=round(time.time()-started)))
print('MODEL_RESTORE_VERIFIED',receipt['total_bytes'],flush=True)
