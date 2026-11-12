from pathlib import Path
import json,hashlib,tarfile,zipfile,shutil,subprocess
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-ocrbench1000-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
REL='vlm_research/results/internvl_svhr_new_ocrbench1000_22875_20261113';P=O/'publication'/REL
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'analysis_done.json').exists() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
protocol=json.loads((O/'protocol.json').read_text())
for path,h in protocol['source_SHA256'].items():assert sha(L/'late9_sources'/path)==h
assert sha(O/'publication/vlm_research/configs/interVL-SVH-R-new.json')==protocol['configuration_sha256']
T=Path('/home/featurize/work/ate-results/internvl-svh-r-new-ocrbench1000-22875-20261113');T.mkdir(parents=True,exist_ok=True)
with tarfile.open(O/'full_results.tar.gz','w:gz') as t:
 for f in ['predictions','inherited_late14_100.json','frozen_rows.json','input_paths.json','expected_bindings.json','configuration.json','protocol.json','cpu_audit.json','smoke.json','environment.json','progress.json','run.log','prepare.py','run.py','analyze.py','analysis_done.json','prereg_commit_receipt.json','prereg_github_verification.json']:
  t.add(O/f,arcname=f)
 t.add(L/'late9_sources',arcname='frozen_sources',filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
 t.add(L/'run_common.py',arcname='external_dependencies/run_common.py')
 t.add(Path('/home/featurize/ate-experiments/internvl3-ocrbench-native-official-22875-20261113/audit/official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py'),arcname='official_scoring/OCRBench_v1_example.py')
shutil.copy2(O/'full_results.tar.gz',T/'full_results.tar.gz');assert sha(O/'full_results.tar.gz')==sha(T/'full_results.tar.gz')
r=dict(full_archive=dict(bytes=(O/'full_results.tar.gz').stat().st_size,sha256=sha(O/'full_results.tar.gz'),local_path=str(O/'full_results.tar.gz'),persistent_path=str(T/'full_results.tar.gz'),persistent_copy_verified=True),windows_backup='PENDING_ACTIVE_TRANSFER')
(O/'archive_receipt.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(O/'archive_receipt.json',P/'archive_receipt.json');shutil.copy2(O/'verify_cpu.py',P/'verify_cpu.py');shutil.copy2(O/'package.py',P/'execution/package.py')
m=[dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='integrity_manifest.json'];(P/'integrity_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
subprocess.run(['/environment/miniconda3/bin/python','-B',str(P/'verify_cpu.py')],check=True)
with zipfile.ZipFile(O/'publication.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted((O/'publication').rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(O/'publication')))
zr=dict(bytes=(O/'publication.zip').stat().st_size,sha256=sha(O/'publication.zip'),files=len(m)+2);(O/'publication_zip_receipt.json').write_text(json.dumps(zr,indent=2)+'\n');print(json.dumps(dict(archive=r,publication=zr),indent=2))
