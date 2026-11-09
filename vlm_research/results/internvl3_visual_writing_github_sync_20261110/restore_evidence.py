"""Restore/verify the published frozen evidence using the Python standard library only."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,os,tarfile,tempfile
S=Path(__file__).resolve().parent
W=S.parents[2]
def digest_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sums(p):
 return dict((n,h) for h,n in (line.split('  ',1) for line in p.read_text().splitlines()))
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive/checksum path')
 target=root.joinpath(*p.parts)
 if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes evidence directory')
 return target
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
 cfg=json.loads((S/'archive_manifest.json').read_text());published=json.loads((S/'publication_manifest.json').read_text())
 for item in published['files']:
  p=safe(W,item['path']);assert p.stat().st_size==item['bytes'] and digest_file(p)==item['sha256'],str(p)
 archived={};total=0
 for a in cfg['archives']:
  src=safe(W,a['path']);dest=safe(W,a['destination']);expected=sums(dest/'SHA256SUMS')
  assert digest_file(src)==a['sha256'],str(src)
  seen=set()
  with tarfile.open(src,'r|gz') as tar:
   for m in tar:
    assert m.isfile() and m.name in expected and m.name not in seen,m.name
    target=safe(dest,m.name);seen.add(m.name);h=hashlib.sha256();out=None;tmp=None
    if not args.verify_only:
     if target.exists():assert target.is_file() and digest_file(target)==expected[m.name],str(target)
     else:
      target.parent.mkdir(parents=True,exist_ok=True)
      out=tempfile.NamedTemporaryFile(prefix=target.name+'.restore-',dir=target.parent,delete=False);tmp=Path(out.name)
    try:
     f=tar.extractfile(m)
     for block in iter(lambda:f.read(1024*1024),b''):
      h.update(block)
      if out:out.write(block)
     assert h.hexdigest()==expected[m.name],m.name
     if out:
      out.close();out=None;os.chmod(tmp,m.mode & 0o777);os.replace(tmp,target);tmp=None
    finally:
     if out:out.close()
     if tmp:tmp.unlink(missing_ok=True)
  assert len(seen)==a['members'],a['path']
  archived.setdefault(a['destination'],set()).update(seen);total+=len(seen)
  print('ARCHIVE PASS',a['path'],len(seen))
 direct=0
 for frozen in cfg['frozen_results']:
  dest=safe(W,frozen['result_directory']);assert digest_file(dest/'SHA256SUMS')==frozen['original_SHA256SUMS']
  expected=sums(dest/'SHA256SUMS');assert len(expected)==frozen['files']
  for name,h in expected.items():
   if name not in archived[frozen['result_directory']]:assert digest_file(safe(dest,name))==h,name;direct+=1
 print('PASS: frozen checksums',total+direct,'archived',total,'direct',direct,'mode','verify-only' if args.verify_only else 'restore')
if __name__=='__main__':main()
