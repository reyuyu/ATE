"""Archive small full raw evidence locally and on persistent disk; publish no weights/images."""
from pathlib import Path
import json,hashlib,shutil,tarfile,zipfile,subprocess,csv
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-wider-reaccess-22875-20261113')
REL='vlm_research/results/internvl3_ocrbench_wider_reaccess_22875_20261113';P=O/'publication'/REL
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'analysis_done.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
summary=json.loads((P/'summary.json').read_text());report=(P/'report_zh.md').read_text()
finding='本轮固定100题中，扩大回看范围将Late9的57分恢复到76–77分：Late16=76，Late20=77，All24=77，Native=79。Late20相对Late9净增20题（21题改善、1题退步），相对Native仍少2题；All24没有额外增加得分。因此L8–L27是值得后续独立验证的较小回看集合，本轮不自动扩大评测，也不宣称两方法等价或有效率收益。'
report=report.replace('## 同一批100题结果',finding+'\n\n## 同一批100题结果',1).replace('100张输入绑定','100题输入绑定')
category=list(csv.DictReader((P/'category_scores.csv').open(encoding='utf-8-sig')))
report+='\n## 分类得分（正确题数）\n\n| 类别 | 题数 | Native | Late9 | Late16 | Late20 | All24 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for cat in sorted(json.loads((O/'protocol.json').read_text())['allocation']):
 d={x['method']:x for x in category if x['category']==cat}
 report+='| '+cat+' | '+d['Native']['n']+' | '+' | '.join(d[m]['correct'] for m in ['Native','Late9','Late16_L12_27','Late20_L8_27','All24_ExceptL2_5'])+' |\n'
(P/'report_zh.md').write_text(report,encoding='utf-8')
for path,h in json.loads((O/'protocol.json').read_text())['source_SHA256'].items():
 assert sha(Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113/late9_sources')/path)==h
T=Path('/home/featurize/work/ate-results/internvl3-ocrbench-wider-reaccess-22875-20261113');T.mkdir(parents=True,exist_ok=True)
with tarfile.open(O/'full_results.tar.gz','w:gz') as t:
 for name in ['predictions','numerical','selected_rows.json','input_paths.json','baselines.json','expected_bindings.json','protocol.json','cpu_audit.json','smoke.json','environment.json','progress.json','run.log','prepare.py','run.py','analyze.py','analysis_done.json']:
  t.add(O/name,arcname=name)
shutil.copy2(O/'full_results.tar.gz',T/'full_results.tar.gz');assert sha(O/'full_results.tar.gz')==sha(T/'full_results.tar.gz')
receipt=dict(full_archive=dict(bytes=(O/'full_results.tar.gz').stat().st_size,sha256=sha(O/'full_results.tar.gz'),local_path=str(O/'full_results.tar.gz'),persistent_path=str(T/'full_results.tar.gz'),persistent_copy_verified=True),windows_backup='PENDING_ACTIVE_TRANSFER')
(O/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(O/'archive_receipt.json',P/'archive_receipt.json')
shutil.copy2(O/'verify_cpu.py',P/'verify_cpu.py');shutil.copy2(O/'package.py',P/'execution/package.py')
files=[dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='integrity_manifest.json']
(P/'integrity_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
subprocess.run(['/environment/miniconda3/bin/python','-B',str(P/'verify_cpu.py')],check=True)
with zipfile.ZipFile(O/'publication.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(P.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(O/'publication')))
r=dict(bytes=(O/'publication.zip').stat().st_size,sha256=sha(O/'publication.zip'),files=len(files)+1)
(O/'publication_zip_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(archive=receipt,publication=r),indent=2))
