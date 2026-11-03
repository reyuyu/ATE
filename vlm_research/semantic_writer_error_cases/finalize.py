"""Finish ONLY the user-authorized no-idle-GPU read-only fallback."""
import base64,datetime,re,shutil
from html.parser import HTMLParser
from .common import *
from .run import verify

class GalleryParser(HTMLParser):
    def __init__(self):super().__init__();self.case=None;self.cases={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='article':self.case=a['id'].removeprefix('case-');self.cases[self.case]=dict(images=[],groups=[])
        if self.case and tag=='img':self.cases[self.case]['images'].append(hashlib.sha256(base64.b64decode(a['src'].split(',',1)[1])).hexdigest())
        if self.case and tag=='span' and a.get('class','').startswith('chip '):self.cases[self.case]['groups'].append(a['class'].split()[1])
    def handle_endtag(self,tag):
        if tag=='article':self.case=None
def main():
    p=verify();assert p['gpu'] is None and not torch.cuda.is_initialized()
    assert not (OUT/'worker_started.json').exists() and not (OUT/'gpu_completion.json').exists() and not (OUT/'STOP.json').exists()
    assert read(OUT/'readonly_validation.json')['passed'] and read(OUT/'html_browser_validation.json')['passed']
    review=read(OUT/'visual_review.json');assert review['passed'] and review['pdf_sha256']==sha256(OUT/'case_cards.pdf') and review['html_sha256']==sha256(OUT/'gallery.html')
    rv=read(OUT/'render_validation.json');assert rv['source_sha256']==sha256(ROOT/'semantic_writer_error_cases/render.py')
    assert rv['artifacts']=={n:sha256(OUT/n) for n in ['gallery.html','case_cards.pdf']}
    cases=selected();parser=GalleryParser();parser.feed((OUT/'gallery.html').read_text(encoding='utf-8'));assert len(parser.cases)==16
    specs={}
    for c in cases:
        q=c['question_id'];spec=c['spec'];sel=set(spec['writers']['Semantic-k'])
        assert parser.cases[q]['images']==[c['sample']['image_sha256']]
        states=['selected' if set(g['positions'])<=sel else 'question' if g['question'] else 'template' for g in spec['groups']]
        assert parser.cases[q]['groups']==states
        name=c['sample']['task']+'__'+q+'.json';src=HIST/'token_manifests'/name;assert read(src)==spec and sha256(src)==gitsha(src)
        dst=OUT/'token_manifests'/name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst);assert sha256(dst)==sha256(src)
        specs[name]=dict(sha256=sha256(dst),source=str(src),base_commit=BASE)
    atomic_json(OUT/'token_manifest_index.json',specs)
    tests=(OUT/'cpu_tests_stdout.txt').read_text();assert 'Ran 17 tests' in tests and re.search(r'\bOK\b',tests)
    atomic_json(OUT/'cpu_validation.json',dict(passed=True,tests=17,new_candidate_constraint_tests=6,inherited_protocol_tests=11,log_sha256=sha256(OUT/'cpu_tests_stdout.txt'),CUDA_initialized=False,new_GPU_smoke_executed=False))
    status=read(OUT/'causal_validation_status.json');assert status['new_generations']==0 and status['status']=='NOT_RUN_NO_IDLE_GPU'
    atomic_json(OUT/'resource_audit.json',dict(status='NO_GPU_ALLOCATED',experiment_model_process_started=False,new_GQA_generations=0,
        initial_idle_observation='Before CPU preparation, GPUs idle; NOT a reservation.',launch_snapshot_sha256=sha256(OUT/'gpu_selection.json'),
        launch_snapshot_time_utc=datetime.datetime.fromtimestamp((OUT/'gpu_selection.json').stat().st_mtime,datetime.timezone.utc).isoformat(),
        selected_gpu=None,other_user_processes_untouched=True,release_not_applicable=True,no_background_GPU_polling=True))
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT.parent,text=True).splitlines()
    assert all(n.startswith(('vlm_research/semantic_writer_error_cases/','vlm_research/results/semantic_writer_error_cases/')) for n in changed)
    atomic_json(OUT/'validation.json',dict(passed=True,scope='read-only cases, history audit, HTML/PDF delivery; NO new causal model runs',
        historical_complete_answer_ACC_rescored=True,history_untouched=True,selected_harm=12,selected_rescue=4,unique_images=16,
        confirmatory_harm=11,confirmatory_rescue=4,mandatory_exploratory_image_exception='001020372',
        whole_group_candidates_locked=True,all_three_token_highlight_classes_exact=True,all_original_image_bytes_exact=True,
        all_16_PDF_pages_visually_reviewed=True,HTML_desktop_mobile_checked=True,causal_validation='NOT_RUN_NO_IDLE_GPU',
        source_hashes={str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'semantic_writer_error_cases').glob('*.py'))}))
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completion.json'}
    assert all((OUT/n).stat().st_size<98_000_000 for n in artifacts)
    atomic_json(OUT/'completion.json',dict(status='READ_ONLY_COMPLETE',base_commit=BASE,read_only_deliverables_complete=True,
        requested_causal_validation='NOT_RUN_NO_IDLE_GPU',selected_harm_cases=12,rescue_counterexamples=4,new_model_generations=0,
        historical_logical_outputs_reused=96,case_PDF_pages=16,artifacts=artifacts,no_further_experiments=True))
    print('READ_ONLY_COMPLETE',len(artifacts),'artifacts; causal GPU tests NOT RUN',flush=True)
if __name__=='__main__':main()
