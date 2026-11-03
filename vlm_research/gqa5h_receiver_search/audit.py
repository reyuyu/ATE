"""Validate all immutable requests, compact evidence, tokens and derived scores."""
import collections,io,unittest
from .common import *
from .tests import SearchTests
from .cpu_cache import enable
enable()

def main():
    started=time.monotonic();verify_sources();f=read(P/'winner_freeze.json');splits=read(P/'split_manifest.json');assert not (P/'STOP.json').exists()
    if (P/'expanded3000/manifest.json').exists():
        from .expansion import activate
        from . import expansion as expansion_module,worker as backend
        expansion=activate();assert sha(expansion_module.__file__)==expansion['execution_source_sha256'] and sha(backend.__file__)==expansion['backend_source_sha256'];splits=dict(splits,metadata={**splits['metadata'],**expansion['metadata']});assert all(read(P/f'expanded3000/worker{i}/completion.json')['passed'] for i in range(2))
    assert sha(P/'split_manifest.json')==f['split_sha256'] and sha(P/'selection_rules.json')==f['selection_rules_sha256'] and sha(P/'final_writer_manifest.json')==f['final_writer_manifest_sha256']
    assert read(P/'deployment_validation.json')['passed']
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SearchTests));(P/'final_cpu_tests.log').write_text(stream.getvalue());assert test.wasSuccessful()
    index=[];features={};worker_counts=collections.Counter();anomalies=collections.Counter();raw_seconds=0.;cuda_ms=0.
    for path in sorted((P/'checkpoints').glob('*/*.json')):
        r=read(path);q=r['question_id'];s=spec(q);g=r['generation'];audit=g['audit'];assert path==result_path(q,r['writers'])
        if r['job_id']=='expanded-winner-1800':assert r['worker_source_sha256']==expansion['backend_source_sha256']
        else:
            source_map=read(P/'jobs'/r['job_id']/'manifest.json')['source_hashes'];assert r['worker_source_sha256']==next(v for k,v in source_map.items() if k.endswith('/gqa5h_receiver_search/worker.py'))
        assert r['input_hashes']==s['input_hashes'] and g['writer_positions']==r['writers'] and set(r['writers'])<=set(s['R'])
        assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['prediction']==gqa_prediction(g['text'])
        assert audit['passed'] and audit['all_32_layers'] and audit['all_native_masks_equal'] and audit['all_non_target_masks_exact'] and audit['absolute_positions_exact'] and audit['BOS_self_legal']
        assert audit['forbidden_max_by_layer']==[0.]*32 and r['MLP_calls']==[g['length']]*32 and r['weights_unchanged'] and r['hook_cleanup']
        assert len(g['token_ids'])==g['length']<=32
        trace=g['cache_lengths'];assert len(trace)==g['length'] and trace[0]['before']==0 and trace[0]['after']==g['query_layout']['prompt_length']
        for j,t in enumerate(trace[1:],1):assert t['before']==trace[j-1]['after'] and t['after']==t['before']+1 and t['consumed_token']==g['token_ids'][j-1]
        assert r['raw']['bytes']==Path(r['raw']['path']).stat().st_size and sha(r['raw']['path'])==r['raw']['sha256']
        with np.load(r['raw']['path']) as z:
            a=z['audit'];assert (a[:,0]==0).all() and (a[:,4:]==1).all() and np.isfinite(a).all() and z['token_ids'].tolist()==g['token_ids']
            meta=json.loads(str(z['record_metadata']));assert meta=={k:v for k,v in r.items() if k!='raw'}
        if q in features:assert features[q]==r['projected_visual_sha256']
        features[q]=r['projected_visual_sha256'];worker_counts[r['worker']]+=1;raw_seconds+=r['seconds'];cuda_ms+=r['cuda_event_ms']
        for key in ['empty','truncated','repeated_trigram']:anomalies[key]+=bool(g[key])
        anomalies['no_EOS']+=not g['eos']
        index.append(dict(question_id=q,writers_sha256=digest(r['writers']),checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256'],raw_bytes=r['raw']['bytes'],worker=r['worker'],job_id=r['job_id'],input_spec_sha256=sha(P/'input_specs'/f'{q}.json'),all_audits_passed=True))
    writecsv(P/'checkpoint_sha_index.csv',index)
    all_tasks={};final_job=P/'jobs/final-winner-600';final_req=read(final_job/'requests.json')['requests'];assert len(final_req)==600 and {r['question_id'] for r in final_req}==set(splits['ids']['FINAL'])
    assert all(r['force_new'] and r['candidate_id']==f['candidate_id'] for r in final_req)
    for manifest in sorted((P/'jobs').glob('*/manifest.json')):
        m=read(manifest);assert sha(manifest.parent/'requests.json')==m['request_sha256'];assert all(read(manifest.parent/f'worker{j}_done.json')['passed'] for j in range(2))
        assert len({(r['question_id'],canonical(r['writers'])) for r in m['tasks']})==len(m['tasks'])
        for row in m['tasks']:
            q=row['question_id'];worker=int(digest(['worker-v1',splits['metadata'][q]['image_sha256']]),16)%2;assert worker==row['worker'];record=read(result_path(q,row['writers']));assert record['worker']==worker
            if q in all_tasks:assert all_tasks[q]==worker
            all_tasks[q]=worker
    assert read(final_job/'manifest.json')['new_generations_requested']==600
    assert all(read(result_path(r['question_id'],r['writers']))['job_id']=='final-winner-600' for r in final_req)
    assert read(final_job/'manifest.json')['monotonic']>f['freeze_monotonic']
    for stage in ['DEV','TUNE','FINAL']:
        rr=list(csv.DictReader((P/'baselines'/stage/'per_question.csv').open()));assert len(rr)==8*len(splits['ids'][stage]);assert {r['question_id'] for r in rr}==set(splits['ids'][stage])
    derived_count=0;history=History()
    derived_files=[P/'per_question.csv',P/'expanded3000/per_question.csv']+[m.parent/'results.csv' for m in (P/'jobs').glob('*/manifest.json') if read(m)['kind']!='candidate']
    for file in derived_files:
        for row in csv.DictReader(file.open()):
            q=row['question_id'];assert int(row['correct'])==int(gqa_prediction(row['answer'])==gold(q));writers=json.loads(row['writers']);path=result_path(q,writers)
            if path.exists():g=read(path)['generation']
            else:g=score_record(q,'derived-verification',writers,history)['generation']
            assert g['text']==row['answer'] and g['token_ids']==json.loads(row['token_ids']);derived_count+=1
            for label,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-first-Q80','Q80'),('Semantic-k','S')]:
                if short+'_correct' in row and row[short+'_correct']!='':
                    b=history.get(q,label)['generation'];correct=int(row['correct']);assert int(row[short+'_correct'])==b['correct'] and int(row[short+'_harm'])==int(b['correct'] and not correct) and int(row[short+'_fix'])==int(not b['correct'] and correct)
            if 'causal_harm_label' in row:assert int(row['causal_harm_label'])==int(row['Q100_correct'])-int(row['correct'])
    history.close();atomic_json(P/'derived_history_recheck_index.json',history.files)
    model=[]
    for name,expected in read(P/'protocol.json')['model_files'].items():assert sha(name)==expected['sha256'];model.append(dict(path=name,sha256=expected['sha256']))
    for worker in range(2):assert read(P/f'worker{worker}/smoke.json')['passed']
    result=dict(passed=True,UTC=utc(),seconds=time.monotonic()-started,CPU_tests=test.testsRun,unique_new_generations=len(index),worker_generation_counts=dict(worker_counts),independent_cache_trace_verified=True,all32_MLP_verified=True,all_non_target_masks_exact=True,forbidden_actual_probability_max=0.,model_hashes_reverified=len(model),projected_features_unchanged_across_conditions=True,input_hashes_exact=True,hooks_cleaned=True,all_Gold_scores_recomputed=True,derived_rows_verified=derived_count,final_single_winner=True,FINAL_fresh_generations=600,FINAL_freeze_before_outputs=True,final_token_manifest_sha256=f['final_writer_manifest_sha256'],image_shards_disjoint=True,output_anomalies=dict(anomalies),formal_inference_seconds_sum=raw_seconds,formal_CUDA_event_ms_sum=cuda_ms,raw_archives_location='Server resource directory only; compact audits and token IDs, no complete vocabulary logits.',source_compatibility=read(P/'historical_inference_source_compatibility.json'))
    atomic_json(P/'validation.json',result);event('FINAL_AUDIT_PASSED',seconds=time.monotonic()-started,unique_generations=len(index));print(json.dumps({k:v for k,v in result.items() if k!='source_compatibility'}),flush=True)

if __name__=='__main__':main()
