"""Read the official eval.zip directory and only GQA text members by HTTP ranges."""
from pathlib import Path
import urllib.request,json,io,zipfile,hashlib,re
H=Path(__file__).resolve().parent
probe=json.loads((H/'official_zip_range_probe.json').read_text(encoding='utf-8'))
URL=probe['url'];total=int(probe['headers']['Content-Range'].split('/')[-1]);tail=(H/'official_zip_tail.bin').read_bytes();downloaded=0
class RemoteZip(io.RawIOBase):
 def __init__(self):self.pos=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else total+offset
  return self.pos
 def read(self,size=-1):
  global downloaded
  if size<0:size=total-self.pos
  size=min(size,total-self.pos)
  assert 0<=size<=4*1024**2,'Refuse large archive read'
  start=self.pos;end=start+size-1
  if start>=total-len(tail):raw=tail[start-(total-len(tail)):start-(total-len(tail))+size]
  else:
   req=urllib.request.Request(URL+'&ate_range='+str(start)+'-'+str(end),headers={'Range':f'bytes={start}-{end}','User-Agent':'ATE-GQA-protocol-audit'})
   with urllib.request.urlopen(req,timeout=35) as response:
    assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),dict(response.headers)
    raw=response.read(size+1);assert len(raw)==size
   downloaded+=len(raw)
  self.pos+=len(raw);return raw
dest=H/'official_sources';manifest=[]
with zipfile.ZipFile(RemoteZip()) as z:
 inventory=[{'name':n.filename,'bytes':n.file_size,'compressed_bytes':n.compress_size} for n in z.infolist()]
 print('OFFICIAL_EVAL_MEMBERS',json.dumps([x for x in inventory if 'gqa' in x['name'].lower()],indent=2),flush=True)
 for n in z.infolist():
  if 'gqa' not in n.filename.lower() or n.is_dir():continue
  if not ('llava_gqa_testdev_balanced.jsonl' in n.filename or ('7b' in n.filename.lower() and n.filename.endswith(('.json','.jsonl')))):continue
  assert n.file_size<10*1024**2 and n.compress_size<4*1024**2
  raw=z.read(n);name='official_eval_zip_'+n.filename.replace('/','__');p=dest/name;p.write_bytes(raw)
  manifest.append({'member':n.filename,'local_name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'zip_CRC':n.CRC})
  print('OFFICIAL_GQA_TEXT_MEMBER_VERIFIED',n.filename,len(raw),flush=True)
(dest/'official_eval_zip_members.json').write_text(json.dumps({'url':URL,'complete_archive_bytes':total,'full_archive_not_downloaded':True,'HTTP_range_bytes_fetched':downloaded+len(tail),'zip_directory':inventory,'extracted_GQA_text':manifest},indent=2)+'\n',encoding='utf-8')
