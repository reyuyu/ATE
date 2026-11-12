"""CPU-only provenance/index finalization; never launches GPU work."""
import subprocess,datetime,ast
from .common import *
def main():
 s=read(OUT/'source_manifest.json');pairs=list(s['files'].items());raw=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=''.join(s['commit']+':'+p+'\n' for p,_ in pairs).encode(),stdout=subprocess.PIPE,check=True).stdout;cursor=0
 for p,h in pairs:
  end=raw.index(b'\n',cursor);header=raw[cursor:end].decode().split();assert header[1]=='blob';n=int(header[2]);start=end+1;assert __import__('hashlib').sha256(raw[start:start+n]).hexdigest()==h and sha(ROOT/p)==h,p;cursor=start+n+1
 assert cursor==len(raw)
 changed=subprocess.check_output(['git','diff','--name-only','0fe03aa1bb1923aee8b234ab3dccf59d7b21deb3'],cwd=ROOT,text=True).splitlines();prefixes=('vlm_research/qwen3vl_c3v2_prefill_av_final/','vlm_research/results/qwen3vl_c3v2_prefill_av_final/');assert all(p.startswith(prefixes) for p in changed),changed
 for p in (ROOT/'vlm_research/qwen3vl_c3v2_prefill_av_final').glob('*.py'):ast.parse(p.read_text())
 assert read(OUT/'model_and_parent_hash_after_run.json')['all_model_files_match'] and read(OUT/'model_and_parent_hash_after_run.json')['all_parent_artifacts_unchanged']
 assert subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()==read(OUT/'initial_preflight.json')['local_main']
 assert 'Ran 36 tests' in (OUT/'cpu_tests.txt').read_text() and (OUT/'cpu_tests.txt').read_text().strip().endswith('OK')
 v=read(OUT/'validation.json');v.update(frozen_sources_Git_and_current_sha_checked=len(pairs),all_model_files_sha_after_run=True,all_parent_artifact_sha_after_run=True,local_main_unchanged=True,only_new_task_namespaces_changed=True,figure_png_visually_reviewed=True,private_raw_directory_mode=oct(PRIVATE.stat().st_mode&0o777));write(OUT/'validation.json',v)
 write(OUT/'analysis_source_manifest.json',dict(note='CPU-only analysis and plot sources, final delivery commit binds exact content.',files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'vlm_research/qwen3vl_c3v2_prefill_av_final').glob('*.py'))}))
 write(OUT/'artifact_index.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),excludes_self=True,files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_index.json']))
 print('FINAL PROVENANCE PASS',len(pairs),'frozen sources;',len(read(OUT/'artifact_index.json')['files']),'delivery files')
if __name__=='__main__':main()
