"""Inspect official eval.zip using bounded HTTP ranges, never fetch the archive."""
from pathlib import Path
import urllib.request,json,struct
URL='https://drive.usercontent.google.com/download?id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy&export=download&confirm=t'
here=Path(__file__).resolve().parent
req=urllib.request.Request(URL,headers={'Range':'bytes=-65536','User-Agent':'ATE-GQA-protocol-audit'})
try:
 with urllib.request.urlopen(req,timeout=30) as response:
  meta={'url':URL,'status':response.status,'headers':{k:v for k,v in response.headers.items() if k.lower() in {'content-type','content-range','content-length','accept-ranges'}}}
  if response.status!=206:raise RuntimeError('Server did not confirm bounded Range; will not download complete archive: '+str(meta))
  raw=response.read(65537);assert len(raw)<=65536
  i=raw.rfind(b'PK\x05\x06');assert i>=0
  eocd=struct.unpack_from('<4s4H2LH',raw,i)
  meta.update(central_directory_bytes=eocd[5],central_directory_offset=eocd[6],entries=eocd[4])
  (here/'official_zip_tail.bin').write_bytes(raw)
 print(json.dumps(meta),flush=True)
 (here/'official_zip_range_probe.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
except Exception as exc:
 print('BOUNDED_PROBE_UNAVAILABLE',str(exc),flush=True)
 (here/'official_zip_range_probe.json').write_text(json.dumps({'status':'UNRESOLVED','error':str(exc),'full_archive_not_downloaded':True},indent=2)+'\n',encoding='utf-8')
