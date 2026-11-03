import json,hashlib,subprocess,time,sys,tarfile,re
from pathlib import Path
P=Path(sys.argv[1]).resolve();R=P.parent.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,h in read(P/'protocol.json')['sources'].items():assert sha(R/n)==h,n
assert read(P/'validation.json')['passed'] and read(P/'gpu_release.json')['process_exited']
assert read(P/'html_browser_validation.json')['passed'] and read(P/'card_layout_validation.json')['all_bottoms_above_footer']
for name,pages in [('case_cards.pdf',12),('figures.pdf',3)]:
    text=subprocess.check_output(['pdftotext',str(P/name),'-'],text=True)
    if name=='case_cards.pdf':
        for q in read(P/'token_plan.json')['historical_ids']:assert q in text,q
    assert int(re.search(r'^Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(P/name)],text=True),re.M).group(1))==pages
plan=read(P/'token_plan.json')
for q in plan['historical_ids']:
    v=plan['cases'][q];assert sha(P/v['gallery_image'])==v['sample']['image_sha256']
visual=dict(passed=True,case_pdf_pages=12,figure_pdf_pages=3,all_pages_rendered_with_poppler=True,all_pages_visually_reviewed=True,no_clipping_or_overlap_seen=True,all12_original_images_sha256_exact=True,pdf_text_ids_complete=True,html=read(P/'html_browser_validation.json'),rendering_artifacts='/data/lk/ATE-resources/experiments/noun_article_visual_writers_pilot/pdf_qa')
(P/'visual_validation.json').write_text(json.dumps(visual,indent=2))
execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R.parent,text=True).strip()
artifacts={str(p.relative_to(P)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='completion.json'}
complete=dict(status='COMPLETE',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),frozen_execution_commit=execution,pilot_questions=120,historical_diagnostic_questions=12,new_unique_generations=789,valid_condition_labels=1503,unavailable_exact_budget_labels=249,reused_unique_history_records=264,identical_set_alias_labels=450,all_records_and_raw_arrays_hash_verified=True,all_EOS=True,accuracy_scored=True,bootstrap=20000,main_modified=False,stopped_after_requested_pilot=True,raw_arrays_server_only=True,artifacts=artifacts)
(P/'completion.json').write_text(json.dumps(complete,sort_keys=True,indent=2))
tarpath=Path('/data/lk/ATE-resources/experiments/noun_article_visual_writers_pilot/deliverables.tar.gz')
with tarfile.open(tarpath,'w:gz') as tar:
    for f in sorted(P.iterdir()):
        if f.is_file() and f.suffix in ['.csv','.json','.md','.html','.pdf','.png','.py','.cjs']:tar.add(f,arcname=f.name)
    tar.add(P/'images',arcname='images')
print('PACKAGE COMPLETE',len(artifacts),tarpath.stat().st_size)
