from pathlib import Path
import json,hashlib,requests,concurrent.futures,time,traceback,os
O=Path(__file__).resolve().parent
Q=Path('/home/featurize/work/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
D=Path('/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct')
I=Path('/home/featurize/ate-local-models/InternVL3-8B')
D.mkdir(exist_ok=True);I.mkdir(exist_ok=True)
receipt=json.loads(Path('/home/featurize/work/ate-qwen3090-control-29535/model_integrity.json').read_text())
meta=json.loads((O/'internvl_hf_metadata.json').read_text())
def one(task):
 kind,name,row=task;dest=(D if kind=='qwen' else I)/name;dest.parent.mkdir(parents=True,exist_ok=True)
 size=row.get('size',row.get('bytes'));lfs=row.get('lfs');expected=row['hash'] if kind=='qwen' else (lfs['sha256'] if lfs else row['blobId']);algo=row['hash_algorithm'] if kind=='qwen' else ('SHA256' if lfs else 'git-blob-SHA1')
 h=hashlib.sha256() if '256' in algo else hashlib.sha1()
 if '256' not in algo:h.update(('blob '+str(size)+'\0').encode())
 temp=dest.with_name(dest.name+'.partial')
 if kind=='qwen':
  assert (Q/name).stat().st_size==size
  with (Q/name).open('rb') as inp,temp.open('wb') as out:
   for b in iter(lambda:inp.read(4*1024**2),b''):out.write(b);h.update(b)
 else:
  url='https://hf-mirror.com/OpenGVLab/InternVL3-8B/resolve/'+meta['sha']+'/'+name
  res=requests.get(url,stream=True,timeout=(25,120));res.raise_for_status()
  with temp.open('wb') as out:
   for b in res.iter_content(4*1024**2):out.write(b);h.update(b)
 assert temp.stat().st_size==size,(name,'size',temp.stat().st_size,size)
 assert h.hexdigest()==expected,(name,'hash',h.hexdigest(),expected)
 temp.replace(dest);print('VERIFIED',kind,name,size,flush=True)
 return kind,name,dict(bytes=size,hash=expected,hash_algorithm=algo,status='VERIFIED')
tasks=[('qwen',n,x) for n,x in receipt['files'].items()]
tasks += [('internvl',x['rfilename'],x) for x in meta['siblings'] if not x['rfilename'].startswith('examples/')]
rows={'qwen':{},'internvl':{}}
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for kind,name,row in pool.map(one,tasks):rows[kind][name]=row;(O/'extra_model_progress.json').write_text(json.dumps({k:len(v) for k,v in rows.items()}))
 for k,d,rev in [('qwen',D,receipt['revision']),('internvl',I,meta['sha'])]:
  obj=dict(status='VERIFIED',revision=rev,path=str(d),files=rows[k]);(d/'.ate_revision.json').write_text(json.dumps(obj,indent=2));(O/(k+'_model_integrity.json')).write_text(json.dumps(obj,indent=2))
 (O/'extra_models.exit').write_text('0')
except BaseException:
 (O/'extra_models_failure.txt').write_text(traceback.format_exc());(O/'extra_models.exit').write_text('1');raise
