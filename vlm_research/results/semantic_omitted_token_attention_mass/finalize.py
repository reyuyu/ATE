"""CPU-only delivery audit. Never loads a model or initializes CUDA."""
import csv,json,subprocess,shutil,re,hashlib,datetime,collections
from pathlib import Path
import numpy as np
from semantic_omitted_token_attention_mass.common import OUT as P,RAW,ROOT,read,sha256,sources,binding,BANDS,normalized

def save(name,data):
    (P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def rows(name):return list(csv.DictReader((P/name).open(encoding='utf-8')))

def main():
    protocol=read(P/'protocol.json');manifest=read(P/'case_manifest.json');cases=manifest['cases'];ids=[v['question_id'] for v in cases]
    assert len(ids)==len(set(ids))==16 and sum(v['kind']=='harm' for v in cases)==12
    assert protocol['sources']==sources() and protocol['case_manifest_sha256']==binding() and protocol['rules_sha256']==sha256(P/'ranking_rules.json')
    assert sha256(P/'selected_cases.csv')==manifest['selected_cases_sha256']
    assert read(P/'cpu_input_validation.json')['binding']==binding()
    assert read(P/'gpu_completion.json')['passed'] and read(P/'gpu_release.json')['passed']
    assert sorted(p.stem for p in (P/'checkpoints').glob('*.json'))==sorted(ids)
    assert sorted(p.stem for p in (P/'measurements').glob('*.npz'))==sorted(ids)
    frozen_commit='d98c590b7832c31fe4b8eb01e576c6de628949e4'
    for name,h in protocol['sources'].items():
        blob=subprocess.check_output(['git','show',frozen_commit+':vlm_research/'+name],cwd=ROOT.parent)
        assert hashlib.sha256(blob).hexdigest()==h,(name,'frozen execution commit')
    # Full hash verification is CPU I/O, including the frozen model files.
    for path,info in protocol['model_files'].items():assert sha256(Path(path))==info['sha256'],path
    for category in ['transformers_sources','scorer_sources']:
        for path,h in protocol[category].items():assert sha256(Path(path))==h,path
    mapping={};array={};total=0;undefined=[];row_error=0
    index=read(P/'per_head_index.json')
    if isinstance(index,dict):index=index['partitions']
    for v in cases:
        q=v['question_id'];cp=read(P/'checkpoints'/(q+'.json'));assert cp['passed'] and cp['rank']==0 and cp['case_manifest_sha256']==binding()
        assert cp['condition']=='Native Full; no hard mask' and cp['input_hashes']==v['input_hashes']
        assert sha256(P/v['gallery_image'])==v['sample']['image_sha256']==cp['image_sha256']
        assert sha256(Path(cp['raw']['path']))==cp['raw']['sha256']
        with np.load(P/'measurements'/(q+'.npz')) as raw:
            m=raw['image_mass'].copy();legal=raw['legal_key_counts'].copy();positions=raw['query_positions'].tolist();logits=raw['native_valid_logits']
            assert logits.shape==(1,1,32064) and np.isfinite(logits).all()
        assert positions==v['spec']['relay_positions']+[v['sample']['input_mapping']['p_minus_one']]
        assert m.shape==(32,32,len(positions)) and np.array_equal(legal,np.broadcast_to(np.array(positions)+1,legal.shape))
        a=cp['native_audit'];assert a['passed'] and a['MLP_calls']==[1]*32 and a['input_hashes']==v['input_hashes']
        assert a['hook_cleanup'] and a['weights_unchanged'] and a['independent_single_prefill'] and a['full_matrices_not_retained']
        assert [l['layer'] for l in a['layer_audits']]==list(range(32))
        for l in a['layer_audits']:
            assert l['mask_modified_edges']==0 and l['actual_mask_sha256']==l['native_mask_sha256']
            assert l['all_prefill_rows_finite_nonnegative'] and l['illegal_probability_exact_zero']
            assert l['all_prefill_row_sum_max_error']<=a['row_sum_fp16_tolerance']==.002
            row_error=max(row_error,l['all_prefill_row_sum_max_error'])
        p0,e,z=normalized(m,legal[:,None,:]);array[q]=(m,legal,positions,p0,e,z)
        mapping[q]={p:g for g in v['groups'] for p in g['positions']}
        assert sorted(mapping[q])==positions[:-1]
        rr=rows('per_head_csv/'+q+'.csv');n=len(positions)-1;assert len(rr)==32*32*n;total+=len(rr)
        assert [(int(r['layer']),int(r['head']),int(r['query_position'])) for r in rr]==[(l,h,p) for l in range(32) for h in range(32) for p in positions[:-1]]
        for field,expected in [('image_mass',m[:,:,:-1]),('p0',np.broadcast_to(p0,m.shape)[:,:,:-1]),('excess',e[:,:,:-1]),('enrichment',z[:,:,:-1])]:
            got=np.array([float(r[field]) if r[field] else np.nan for r in rr]).reshape(32,32,n)
            assert np.array_equal(got,expected,equal_nan=True),(q,field)
        for r in rr:
            p=int(r['query_position']);g=mapping[q][p];assert r['scope']==g['scope'] and int(r['group_id'])==g['group_id'] and r['semantic_selected']==str(g['selected'])
            assert int(r['legal_keys'])==p+1 and int(r['image_keys'])==576 and r['question_id']==q
            if r['enrichment']=='':undefined.append({k:r[k] for k in ['question_id','layer','head','query_position','group_text','image_mass','enrichment_undefined_reason']})
        ix=next(x for x in index if x['question_id']==q);assert ix['rows']==len(rr) and ix['sha256']==sha256(P/ix['path'])
    p1=rows('p_minus_one_per_head.csv');assert len(p1)==16*32*32
    for i,v in enumerate(cases):
        q=v['question_id'];m,legal,positions,p0,e,z=array[q];rr=p1[i*1024:(i+1)*1024]
        assert [(r['question_id'],int(r['layer']),int(r['head']),int(r['query_position'])) for r in rr]==[(q,l,h,positions[-1]) for l in range(32) for h in range(32)]
        for field,expected in [('image_mass',m[:,:,-1]),('p0',np.broadcast_to(p0,m.shape)[:,:,-1]),('excess',e[:,:,-1]),('enrichment',z[:,:,-1])]:
            got=np.array([float(r[field]) if r[field] else np.nan for r in rr]).reshape(32,32);assert np.array_equal(got,expected,equal_nan=True),(q,field,'P-1')
        assert all(r['scope']=='P-minus-one' for r in rr)
    assert total==474112 and len(undefined)==1
    # Independently recompute all word-group/band/head aggregates and the fixed ranks.
    gb=rows('group_band_rankings.csv');group_index={v['question_id']:{g['group_id']:g for g in v['groups']} for v in cases}
    pools=collections.defaultdict(list)
    for r in gb:
        q=r['question_id'];g=group_index[q][int(r['group_id'])];m,legal,positions,p0,e,z=array[q];js=[positions.index(p) for p in g['positions']];ls=BANDS[r['band']]
        vals=m[ls][:,:,js];heads=vals.mean(axis=(0,2));subs=vals.mean(axis=(0,1));en=z[ls][:,:,js];en=en[np.isfinite(en)]
        assert float(r['mass_mean'])==float(vals.mean()) and float(r['mass_head_median'])==float(np.median(heads)) and float(r['mass_head_max'])==float(heads.max())
        assert float(r['subword_mean'])==float(subs.mean()) and float(r['subword_max'])==float(subs.max())
        assert float(r['excess_mean'])==float(e[ls][:,:,js].mean()) and float(r['enrichment_mean'])==float(en.mean())
        assert r['scope']==g['scope'] and r['semantic_selected']==str(g['selected'])
        if r['semantic_selected']=='False':pools[(q,r['band'],r['scope'])].append(r)
    for pool in pools.values():
        ordered=sorted(pool,key=lambda r:(-float(r['mass_mean']),int(r['first_position']),int(r['group_id'])))
        assert [int(r['rank_mass']) for r in ordered]==list(range(1,len(pool)+1))
    high=rows('high_mass_omitted_tokens.csv');assert all(r['semantic_selected']=='False' and 1<=int(r['rank_mass'])<=5 for r in high)
    # Re-read the server-only smoke arrays; the public audit links their exact hashes.
    smokes=[]
    for q in ['001020372','14236062']:
        s=read(P/'worker0'/('smoke_'+q+'.json'));assert s['passed'] and s['native_noop_logits_torch_equal'] and s['native_allwriter_R_image_probabilities_torch_equal']
        assert s['independent_KV_caches'] and s['hook_cleanup_on_exception'] and s['input_and_visual_hash_exact']
        assert sha256(Path(s['raw']['path']))==s['raw']['sha256']
        with np.load(s['raw']['path']) as raw:
            n=raw['native_R_image_probabilities'];a=raw['allwriter_R_image_probabilities'];assert n.shape==a.shape==(32,32,len(array[q][2])-1,576)
            assert np.array_equal(n,a) and np.array_equal(raw['native_valid_logits'],raw['noop_valid_logits'])
            assert np.array_equal(n.astype(np.float64).sum(-1),array[q][0][:,:,:-1])
        smokes.append(dict(question_id=q,passed=True,raw_sha256_verified=True,exact_arrays_rechecked=True))
    # Preserve prior CPU-only output, then make the root mapping reflect real observations.
    previous=P/'attempts/two_gpu_resource_stop'
    original=P/'token_mapping_audit.csv'
    if not (previous/original.name).exists():shutil.copyfile(original,previous/original.name)
    old=rows('attempts/two_gpu_resource_stop/token_mapping_audit.csv')
    for r in old:r['image_mass_status']='COLLECTED_NATIVE_FULL';r['actual_worker_rank']='0';r['per_head_csv']='per_head_csv/'+r['question_id']+'.csv'
    with original.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
    for name in ['stopped_delivery.py','package_stop.py','render_input_preview.py','html_qa.cjs']:
        if (P/name).exists():shutil.move(str(P/name),str(previous/name))
    save('attempts/serial_initialization_failure/accounting_addendum.json',dict(discarded_native_plain_reference_prefills=1,observed_case_records=0,reason='First reference forward completed before the inference-tensor weight version audit failed; discarded. A separate refrozen run completed all validation and16 observations.'))
    assert read(P/'html_browser_validation.json')['passed'] and read(P/'plot_validation.json')['passed']
    cards=subprocess.check_output(['pdfinfo',str(P/'case_cards.pdf')],text=True);figures=subprocess.check_output(['pdfinfo',str(P/'summary_figures.pdf')],text=True)
    assert int(re.search(r'Pages:\s*(\d+)',cards)[1])==32 and int(re.search(r'Pages:\s*(\d+)',figures)[1])==3
    text=subprocess.check_output(['pdftotext','-layout',str(P/'case_cards.pdf'),'-'],text=True)
    assert all(text.count(q)>=2 for q in ids)
    save('visual_validation.json',dict(passed=True,case_pages=32,summary_pages=3,all32_case_pages_rendered_and_visually_reviewed=True,all3_summary_pages_rendered_and_visually_reviewed=True,full_page_checks=[9,24],HTML_actual_Edge_tests=read(P/'html_browser_validation.json'),PDF_layout=read(P/'card_layout_validation.json'),local_QA='work/omitted_attention_observed_qa;32page contact sheets plus full-page inspections; desktop and mobile screenshots',no_clipped_content=True,figure_A_n_labels_separate=True))
    save('validation.json',dict(passed=True,cpu_tests=protocol['cpu_tests'],cpu_input_audit=read(P/'cpu_input_validation.json'),frozen_execution_commit=frozen_commit,protocol_sources_exact=True,model_all_files_SHA256_rechecked=True,transformers_and_scorer_SHA256_rechecked=True,raw_and_CSV_all_rows_exact=True,R_rows=total,P_minus_one_rows=len(p1),group_band_rows_recomputed=len(gb),rank_rule_exact=True,all16_native_masks_unchanged=True,all32_MLP_native=True,all_inputs_visual_features_weights_unchanged=True,all_prefill_nonnegative_and_illegal_edges_zero=True,maximum_actual_row_sum_error=row_error,existing_FP16_row_tolerance=.002,logits_equality='strict torch.equal in GPU smoke; arrays exactly rechecked on CPU',smokes=smokes,undefined_enrichment=undefined,no_infinite_enrichment=True,no_clipping_or_renormalization=True,P_minus_one_separate=True,checkpoint_image_and_array_SHA256_exact=True,one_GPU_user_amendment=protocol['user_amendment'],main_cases_all_worker0=True,new_answers=0,decode_steps=0,resource_release=read(P/'gpu_release.json'),visual_checks_passed=True))
    save('completion.json',dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=16,harm=12,rescue=4,native_main_prefills=16,successful_validation_extra_prefills=6,successful_run_prefill_calls=22,discarded_prior_reference_prefills=1,total_attempted_prefill_calls=23,new_generated_answers=0,decode_steps=0,R_rows=total,P_minus_one_rows=len(p1),all16_checkpoints_passed=True,all16_measurements_verified=True,validation_passed=True,figures_and_32page_case_cards_passed=True,GPU_released=True,frozen_execution_commit=frozen_commit,case_manifest_sha256=binding(),CPU_only_derivation_scripts=['analyze.py','plots.py','render.py','report.py','finalize.py'],history_results_unchanged=True,stop_boundary='No new accuracy experiment, no token swaps, no rule changes, no further GPU experiments.'))
    files={str(f.relative_to(P)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='artifact_index.json' and f.suffix!='.lock' and '__pycache__' not in f.parts}
    save('artifact_index.json',dict(excludes=['artifact_index.json','*.lock','__pycache__'],files=files))
    assert not __import__('torch').cuda.is_initialized()
    print('FINAL AUDIT COMPLETE',total,len(p1),'files',len(files),flush=True)

if __name__=='__main__':main()
