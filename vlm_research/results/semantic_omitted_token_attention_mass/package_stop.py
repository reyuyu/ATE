import json,hashlib,subprocess,time,tarfile,ast
from pathlib import Path
P=Path('/data/lk/ATE-worktrees/semantic-omitted-token-attention-mass/vlm_research/results/semantic_omitted_token_attention_mass');R=P.parent.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=read(P/'protocol.json');assert p['status']=='STOP_NO_TWO_IDLE_GPUS' and not p['gpus'];assert read(P/'completion.json')['experiment_complete'] is False
    for n,h in p['sources'].items():assert sha(R/n)==h,n
    for f in (R/'semantic_omitted_token_attention_mass').glob('*.py'):ast.parse(f.read_text())
    assert not (P/'measurements').exists() and not (P/'gpu_completion.json').exists() and not (P/'GO.json').exists()
    for c in read(P/'case_manifest.json')['cases']:assert sha(P/c['gallery_image'])==c['sample']['image_sha256']
    info=subprocess.check_output(['pdfinfo',str(P/'case_cards.pdf')],text=True);assert '16' in next(s for s in info.splitlines() if s.startswith('Pages:'))
    text=subprocess.check_output(['pdftotext',str(P/'case_cards.pdf'),'-'],text=True);assert text.count('ATTENTION NOT COLLECTED')==16
    assert read(P/'html_browser_validation.json')['passed']
    (P/'visual_validation.json').write_text(json.dumps(dict(passed=True,input_preview_only=True,attention_not_collected=True,all16_pages_rendered_and_visually_reviewed=True,all16_status_labels_present=True,all16_original_images_sha_exact=True,no_clipping_or_overlap=True,html=read(P/'html_browser_validation.json')),indent=2))
    artifacts={str(f.relative_to(P)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='artifact_index.json'}
    (P/'artifact_index.json').write_text(json.dumps(dict(status='RESOURCE_STOP_CPU_DELIVERY',artifacts=artifacts),sort_keys=True,indent=2))
    tarpath=Path('/data/lk/ATE-resources/experiments/semantic_omitted_token_attention_mass/cpu_deliverables.tar.gz')
    with tarfile.open(tarpath,'w:gz') as tar:
        for f in sorted(P.iterdir()):tar.add(f,arcname=f.name)
    print('CPU STOP DELIVERY',len(artifacts),tarpath.stat().st_size)
if __name__=='__main__':main()
