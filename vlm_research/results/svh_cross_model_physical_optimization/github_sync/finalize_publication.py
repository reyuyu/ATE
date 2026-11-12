from pathlib import Path
import json,hashlib,subprocess
P=Path('/home/featurize/ate-experiments/svh-cross-model-github-sync-20261113')
T=P/'vlm_research/results/svh_cross_model_physical_optimization'
M=P.parent/'svh-cross-model-publication-precommit-20261113.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
(T/'.gitattributes').write_text('# Preserve frozen evidence bytes; keep this self-contained archive in ordinary Git.\n* -text -filter\n')
v=json.loads((T/'github_sync/publication_validation.json').read_text());v['git_attributes_preserve_original_bytes']=True;save(T/'github_sync/publication_validation.json',v)
a=json.loads((T/'GITHUB_ARTIFACT_MANIFEST.json').read_text())
a['published_files']=[dict(path=str(p.relative_to(T)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(T.rglob('*')) if p.is_file() and p.name not in ['GITHUB_ARTIFACT_MANIFEST.json','GITHUB_SHA256SUMS']]
save(T/'GITHUB_ARTIFACT_MANIFEST.json',a)
rows=[dict(path=str(p.relative_to(T)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(T.rglob('*')) if p.is_file() and p.name!='GITHUB_SHA256SUMS']
(T/'GITHUB_SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in rows))
rows.append(dict(path='GITHUB_SHA256SUMS',bytes=(T/'GITHUB_SHA256SUMS').stat().st_size,sha256=sha(T/'GITHUB_SHA256SUMS')))
m=json.loads(M.read_text());m.update(published_files=len(rows),published_bytes=sum(r['bytes'] for r in rows),max_file_bytes=max(r['bytes'] for r in rows),files=rows,validation=v);save(M,m)
subprocess.run(['sha256sum','-c','GITHUB_SHA256SUMS','--quiet'],cwd=T,check=True)
print(json.dumps(dict(status='PASS',published_files=len(rows),published_bytes=m['published_bytes'])))
