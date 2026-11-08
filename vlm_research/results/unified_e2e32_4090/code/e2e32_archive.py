"""Package finalized evidence after GPU release; no GPU library imports."""
from e2e32_prepare import *
import tarfile
def main():
 assert read(OUT/'validation.json')['status']=='PASS'
 assert read(OUT/'gpu_release.json')['status']=='RELEASED'
 evidence=read(OUT/'evidence_manifest.json')['files']
 for relative,info in evidence.items():
  path=OUT/relative;assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],relative
 archive=Path('/home/featurize/ate-local-data/unified_e2e32_4090_complete.tar.gz');assert not archive.exists()
 with tarfile.open(archive,'w:gz',compresslevel=6) as t:
  for relative in [*evidence,'evidence_manifest.json']:t.add(OUT/relative,arcname='unified-e2e32-4090/'+relative,recursive=False)
 receipt=dict(status='PASS',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),verified_files=len(evidence),formal_requests=240,GPU_UUID=UUID)
 path=archive.with_name('unified_e2e32_4090_archive_receipt.json');path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
