from pathlib import Path
import argparse,json,tarfile,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');ap.add_argument('--extract-to',type=Path);args=ap.parse_args()
root=Path(__file__).resolve().parent;manifest=json.loads((root/'frozen_sources_members.json').read_text());archive=root/manifest['archive']
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['archive_sha256']
expected={r['path']:r for r in manifest['files']};dest=args.extract_to.resolve() if args.extract_to else None
if dest:
 assert not dest.exists(),'Extraction destination must be new; never overwrite frozen sources.'
 dest.mkdir(parents=True)
seen=set()
with tarfile.open(archive,'r|gz') as tar:
 for m in tar:
  assert m.isfile() and m.name in expected and m.name not in seen
  q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts;assert m.size==expected[m.name]['bytes']
  src=tar.extractfile(m);h=hashlib.sha256();out=None
  if dest:
   target=dest/q;target.parent.mkdir(parents=True,exist_ok=True);out=target.open('xb')
  try:
   while True:
    b=src.read(1024*1024)
    if not b:break
    h.update(b)
    if out:out.write(b)
  finally:
   if out:out.close()
  assert h.hexdigest()==expected[m.name]['sha256'],m.name
  seen.add(m.name)
assert seen==set(expected)
print(json.dumps(dict(status='PASS',source_files=len(seen),extracted_to=str(dest) if dest else None)))
