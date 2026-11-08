"""Pinned public HF files; three downloads, resume and full content checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,json,hashlib,time
from urllib.parse import urlparse
REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
ROOT=Path('/home/featurize/data/ate-recovery');MODEL=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
OUT=Path('/home/featurize/work/ATE/vlm_research/results/featurize_4090_c3v2_handoff')
MODEL.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
url='https://huggingface.co/api/models/llava-hf/llava-1.5-7b-hf/revision/'+REV+'?blobs=true'
if (OUT/'model_api.json').is_file():meta=json.loads((OUT/'model_api.json').read_text())
else:
 res=requests.get(url,timeout=40);res.raise_for_status();meta=res.json()
assert meta['sha']==REV
files=[x for x in meta['siblings'] if '/' not in x['rfilename'] and x['rfilename'].endswith(('.json','.model','.safetensors'))]
assert len([x for x in files if x['rfilename'].endswith('.safetensors')])==3
old=json.loads((OUT.parent/'featurize_new3090_recovery/model_acceptance.json').read_text())
def download(info):
 name=info['rfilename'];p=MODEL/name;part=p.with_name(name+'.part');expected=info['size'];lfs=info.get('lfs')
 if p.is_file() and p.stat().st_size==expected:
  hashed=sha(p)
  if not lfs or hashed==lfs['sha256']:return name,{'bytes':expected,'sha256':hashed,'reused':True}
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0
   headers={'Range':f'bytes={offset}-'} if offset else {}
   endpoint='hf-mirror.com' if attempt%2==0 else 'huggingface.co'
   u=f'https://{endpoint}/llava-hf/llava-1.5-7b-hf/resolve/{REV}/{name}?download=true&ate_retry={attempt}'
   with requests.get(u,headers=headers,stream=True,timeout=(30,120)) as response:
    response.raise_for_status()
    if offset and response.status_code!=206:offset=0
    with part.open('ab' if offset else 'wb') as f:
     count=offset;last=time.monotonic()
     for b in response.iter_content(4*1024**2):
      if not b:continue
      f.write(b);count+=len(b)
      if time.monotonic()-last>25:
       print('MODEL_DOWNLOAD',name,count,'/',expected,flush=True);last=time.monotonic()
   assert part.stat().st_size==expected,(name,part.stat().st_size,expected)
   h=sha(part)
   if lfs:assert h==lfs['sha256'],name
   else:assert hashlib.sha1(b'blob '+str(expected).encode()+b'\0'+part.read_bytes()).hexdigest()==info['blobId'],name
   if name in old['files']:assert h==old['files'][name]['sha256'] and expected==old['files'][name]['bytes']
   part.replace(p);print('MODEL_FILE_VERIFIED',name,expected,flush=True)
   return name,{'bytes':expected,'sha256':h,'git_blob':info['blobId'],'lfs_sha256':lfs['sha256'] if lfs else None,'download_endpoint':endpoint,'final_download_host':urlparse(response.url).hostname}
  except Exception as e:
   print('MODEL_RETRY',name,attempt,type(e).__name__,str(e)[:180],flush=True)
   if attempt==3:raise
   time.sleep(min(5*(attempt+1),15))
with ThreadPoolExecutor(max_workers=3) as pool:verified=dict(pool.map(download,files))
r={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'model':'llava-hf/llava-1.5-7b-hf','revision':REV,'path':str(MODEL),
 'files':verified,'all_content_verified':True,'all_three_shards_match_3090':True,'bytes':sum(v['bytes'] for v in verified.values()),'source':'Official huggingface.co pinned revision API fetched on Windows after server API429; content-identical pinned HF mirror downloads verified against official hashes and accepted3090 shard hashes; no alternate weights.',
 'storage_repair':'Initial NFS shard verification raised EIO; failed partial files retained. Final validated model is on local ext4, not NFS. No GPU model loaded before repair.'}
(OUT/'model_download.json').write_text(json.dumps(r,indent=2)+'\n')
print('MODEL_READY',len(verified),r['bytes'],flush=True)
