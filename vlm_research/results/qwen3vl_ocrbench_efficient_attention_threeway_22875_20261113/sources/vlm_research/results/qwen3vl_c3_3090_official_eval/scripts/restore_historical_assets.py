"""Restore exact frozen JPEG bytes from original MMEB archive, not substitute data."""
from pathlib import Path
import hashlib,json,os,time,urllib.request,zipfile
C=Path('/home/featurize/work/ate-qwen3090-control-29535')
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval')
D=Path('/home/featurize/work/ate-qwen3090-data');D.mkdir(exist_ok=True)
meta=next(r for r in json.loads((C/'original_gqa_image_source_probe.json').read_text()) if r.get('available_images')==918)
assert all(r['matches_frozen'] for r in meta['probes'])
row=meta['archive'];path=D/'MMEB-GQA-eb3a8af.zip';part=path.with_suffix('.zip.part')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
if path.exists():assert path.stat().st_size==row['size'] and sha(path)==row['lfs']['sha256']
else:
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0
   domain='https://huggingface.co' if attempt==0 else 'https://hf-mirror.com'
   url=domain+'/datasets/'+meta['dataset']+'/resolve/'+meta['revision']+'/'+row['rfilename']+'?download=true&ate_assets='+str(attempt)
   request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-','User-Agent':'ATE-exact-historical-assets/1'})
   with urllib.request.urlopen(request,timeout=60) as r:
    if offset and r.status!=206:offset=0
    if r.status==206:assert r.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
    with part.open('ab' if offset else 'wb') as f:
     while True:
      b=r.read(2**20)
      if not b:break
      f.write(b)
   assert part.stat().st_size==row['size'] and sha(part)==row['lfs']['sha256']
   part.replace(path);break
  except Exception as e:
   print('ORIGINAL_ASSET_DOWNLOAD_RETRY',attempt,type(e).__name__,flush=True)
   if attempt==3:raise
manifest=json.loads((W/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed/sample_manifest.json').read_text())
expected={Path(r['image_path']).name:r['image_sha256'] for r in manifest['samples']}
dest=D/'historical-gqa1000-images';dest.mkdir(exist_ok=True);images={}
with zipfile.ZipFile(path) as z:
 byname={Path(n).name:n for n in z.namelist() if not n.startswith('__MACOSX/')}
 assert set(expected)<=set(byname)
 for name,h in sorted(expected.items()):
  p=dest/name
  if p.exists():assert sha(p)==h
  else:
   raw=z.read(byname[name]);assert hashlib.sha256(raw).hexdigest()==h
   t=p.with_suffix('.tmp');t.write_bytes(raw);t.replace(p)
  images[name]=dict(path=str(p),sha256=h,bytes=p.stat().st_size)
result=dict(status='PASS',archive=str(path),archive_SHA256=row['lfs']['sha256'],dataset=meta['dataset'],revision=meta['revision'],images=images,question_count=len(manifest['samples']),image_count=len(images),all_images_exact_historical_SHA256=True,substitute_data=False)
(C/'historical_asset_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print('EXACT_HISTORICAL_GQA_ASSETS_RESTORED',len(images),'images',len(manifest['samples']),'questions',flush=True)
