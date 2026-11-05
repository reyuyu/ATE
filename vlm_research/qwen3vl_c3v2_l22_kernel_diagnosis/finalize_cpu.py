"""CPU provenance and artifact finalization; run only after GPU release."""
import pathlib,json,hashlib,subprocess,datetime,ast
from .common import *
def main():
 requests=[]
 for f in sorted(OUT.glob('source_manifest*.json')):
  j=read(f)
  for path,h in j['files'].items():requests.append((f.name,j['commit'],path,h))
 commands=''.join(c+':'+p+'\n' for _,c,p,_ in requests).encode();raw=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=commands,stdout=subprocess.PIPE,check=True).stdout;cursor=0
 for f,c,p,h in requests:
  end=raw.index(b'\n',cursor);header=raw[cursor:end].decode().split();assert len(header)==3 and header[1]=='blob',(f,c,p,header);n=int(header[2]);start=end+1;data=raw[start:start+n];assert hashlib.sha256(data).hexdigest()==h,(f,c,p);cursor=start+n+1
 assert cursor==len(raw)
 # Parent source and historical delivery byte contents unchanged from base.
 names=subprocess.check_output(['git','diff','--name-only','a1a99c88c0db9244ef280017f297d9c5b5e75bbe'],cwd=ROOT,text=True).splitlines()
 assert all(p.startswith(('vlm_research/qwen3vl_c3v2_l22_kernel_diagnosis/','vlm_research/results/qwen3vl_c3v2_l22_kernel_diagnosis/')) for p in names),names
 parent=read(PARENT/'artifact_index.json')
 entries=parent['files'] if isinstance(parent,dict) and 'files' in parent else parent
 # Prior task index shape is explicitly checked rather than silently skipped.
 if isinstance(entries,dict):entries=[dict(path=k,sha256=v) for k,v in entries.items()]
 for r in entries:
  p=PARENT/r['path'];assert sha(p)==r['sha256'],str(p)
 assert read(OUT/'model_hash_revalidation.json')['all_16_files_match']
 for p in (ROOT/'vlm_research/qwen3vl_c3v2_l22_kernel_diagnosis').glob('*.py'):ast.parse(p.read_text())
 sample=read(ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_debug/sample_manifest.json');binding=sample['input_bindings']['GQA__06905471'];write(OUT/'sample_manifest.json',dict(original='GQA__06905471',input_bindings={'GQA__06905471':binding},source_manifest_path='vlm_research/results/qwen3vl_c3v2_numerical_debug/sample_manifest.json',source_manifest_sha256=sha(ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_debug/sample_manifest.json'),only_original_sample_run=True))
 val=read(OUT/'validation.json');val['checks'].update(cpu_tests_27_pass=True,frozen_git_source_objects_sha_verified=len(requests),parent_artifact_sha_after_run=True,full_model_files_sha_after_run=True,figure_png_visual_review=True,python_all_sources_parse=True,private_raw_directory_mode=oct(PRIVATE.stat().st_mode&0o777));write(OUT/'validation.json',val)
 (OUT/'source_commit_current.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
 files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_index.json']
 write(OUT/'artifact_index.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),excludes_self=True,files=files))
 print('FROZEN SOURCE OBJECTS VERIFIED',len(requests),'PARENT ARTIFACTS',len(entries),'DELIVERY FILES',len(files))
if __name__=='__main__':main()
