from pathlib import Path
import json,hashlib,tarfile,zipfile,shutil,subprocess
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-late14-22875-20261113');REL='vlm_research/results/internvl3_ocrbench_late14_22875_20261113';P=O/'publication'/REL
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'analysis_done.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for path,h in json.loads((O/'protocol.json').read_text())['source_SHA256'].items():
 assert sha(Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113/late9_sources')/path)==h
report=(P/'report_zh.md').read_text();report=report.replace('## 配对比较','本批样本上，Late14以14个回看层达到与Late20、All24相同的77分，逐题得分向量也一致，但分别有10条、8条回答文本不同。它相对Late9恢复21题、恶化1题，相对Native仍少2题。Late14与Late20/All24的配对Bootstrap区间为[0,0]只是本样本差分全为0的结果，不能推断总体方法等价。可把Late14作为后续独立验证候选，但本轮不再扩展，也没有验证效率收益。\n\n## 配对比较',1);(P/'report_zh.md').write_text(report,encoding='utf-8')
T=Path('/home/featurize/work/ate-results/internvl3-ocrbench-late14-22875-20261113');T.mkdir(parents=True,exist_ok=True)
with tarfile.open(O/'full_results.tar.gz','w:gz') as t:
 for f in ['predictions','numerical','selected_rows.json','baselines.json','expected_bindings.json','input_paths.json','protocol.json','cpu_audit.json','smoke.json','environment.json','progress.json','run.log','prepare.py','run.py','analyze.py','analysis_done.json']:
  t.add(O/f,arcname=f)
shutil.copy2(O/'full_results.tar.gz',T/'full_results.tar.gz');assert sha(O/'full_results.tar.gz')==sha(T/'full_results.tar.gz')
r=dict(full_archive=dict(bytes=(O/'full_results.tar.gz').stat().st_size,sha256=sha(O/'full_results.tar.gz'),local_path=str(O/'full_results.tar.gz'),persistent_path=str(T/'full_results.tar.gz'),persistent_copy_verified=True),windows_backup='PENDING_ACTIVE_TRANSFER')
(O/'archive_receipt.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(O/'archive_receipt.json',P/'archive_receipt.json')
shutil.copy2(O/'verify_cpu.py',P/'verify_cpu.py');shutil.copy2(O/'package.py',P/'execution/package.py')
m=[dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='integrity_manifest.json'];(P/'integrity_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
subprocess.run(['/environment/miniconda3/bin/python','-B',str(P/'verify_cpu.py')],check=True)
with zipfile.ZipFile(O/'publication.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(P.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(O/'publication')))
zr=dict(bytes=(O/'publication.zip').stat().st_size,sha256=sha(O/'publication.zip'),files=len(m)+1);(O/'publication_zip_receipt.json').write_text(json.dumps(zr,indent=2)+'\n');print(json.dumps(dict(archive=r,publication=zr),indent=2))
