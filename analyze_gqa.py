"""CPU-only official GQA types, functional-program diagnostics, and historical joins."""
import csv
import gzip
import io
import subprocess
from collections import Counter,defaultdict
from full_common import *
from statistics_local import paired


def classify(raw):
    types=raw.get('types',{})
    program=raw.get('semantic',[])
    queries=[(i,x) for i,x in enumerate(program) if x['operation']=='query']
    query_arg=queries[-1][1]['argument'] if queries else 'NOT_QUERY'
    qindex=queries[-1][0] if queries else len(program)
    pre=program[:qindex]
    rels=sum(x['operation']=='relate' for x in pre)
    filters=sum(x['operation'].startswith('filter') for x in pre)
    depth=[]
    for i,x in enumerate(program):
        assert all(isinstance(d,int) and 0<=d<i for d in x['dependencies']),x
        depth.append(1+max([depth[d] for d in x['dependencies']],default=0))
    return dict(structural=types.get('structural','UNKNOWN'),semantic=types.get('semantic','UNKNOWN'),
        detailed=types.get('detailed','UNKNOWN'),query_argument=query_arg,program_length=len(program),
        program_depth=max(depth,default=0),relate_count=rels,filter_count=filters,
        locate_then_attribute=bool(types.get('structural')=='query' and types.get('semantic')=='attr' and rels+filters>0),
        locate_any=bool(rels+filters>0),multiple_relate=bool(rels>=2),
        program_json=program)


def git_text(branch,path):
    return subprocess.check_output(['git','show',branch+':'+path],cwd=PRIOR,text=True)


def history(current):
    annotation=Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_val_balanced.parquet')
    specs=[('codex/qwen3vl-c3-quality-recovery','qwen3vl_c3_quality_recovery'),
        ('codex/qwen3vl-keep-l0-21-gqa1000-reproduction','qwen3vl_keep_l0_21_gqa1000_reproduction'),
        ('codex/qwen3vl-keep-l0-21-joint-drop','qwen3vl_keep_l0_21_gqa3000')]
    ids=set()
    for branch,name in specs:
        m=json.loads(git_text(branch,'vlm_research/results/'+name+'/sample_manifest.json'))
        ids.update(str(r['question_id']) for r in m['samples'])
    # Use the existing CPU-only Arrow environment without changing the frozen Qwen environment.
    reader='/data/lk/深度学习/.envs/video-model/bin/python'
    code=('import json,sys; import pyarrow.parquet as pq; ids=set(json.load(sys.stdin)); '
          'rows=pq.read_table(sys.argv[1],columns=["id","imageId","question","answer","types"]).to_pylist(); '
          'print(json.dumps({str(r["id"]):r for r in rows if str(r["id"]) in ids}))')
    official_val=json.loads(subprocess.check_output([reader,'-c',code,str(annotation)],input=json.dumps(sorted(ids)),text=True))
    assert set(official_val)==ids
    summaries=[]
    bindings=[]
    for branch,name in specs:
        root='vlm_research/results/'+name+'/'
        commit=subprocess.check_output(['git','rev-parse',branch],cwd=PRIOR,text=True).strip()
        manifest_text=git_text(branch,root+'sample_manifest.json')
        manifest=json.loads(manifest_text)
        samples=manifest['samples'] if isinstance(manifest,dict) and 'samples' in manifest else manifest
        if isinstance(samples,dict):
            # Do not guess an unknown manifest schema.
            write(OUT/(name+'_schema_unresolved.json'),dict(keys=list(samples),commit=commit))
            continue
        records=list(csv.DictReader(io.StringIO(git_text(branch,root+'per_question.csv'))))
        manifest_index={str(r['question_id']):r for r in samples}
        assert len(manifest_index)==len(samples)
        verified_images={}
        for qid,raw in manifest_index.items():
            source=official_val[qid]
            assert str(raw['image_id'])==str(source['imageId'])
            assert raw['question']==source['question'] and raw.get('gold',raw.get('answer'))==source['answer']
            if raw.get('types'):
                assert raw['types']==source['types']
            raw['types']=source['types']
            if raw.get('image_path') and Path(raw['image_path']).exists():
                image=raw['image_path']
                if image not in verified_images:
                    verified_images[image]=sha(image)
                assert verified_images[image]==raw['image_sha256']
        joined=[]
        for qid,raw in manifest_index.items():
            for r in current:
                if r['question_id']==qid and r['image_sha256']==raw['image_sha256']:
                    assert r['gold']==raw['gold'] and r['question']==raw['question']
                    joined.append(qid)
        coverage=Counter()
        semantics=Counter()
        for raw in samples:
            coverage[raw.get('types',{}).get('structural','UNKNOWN')]+=1
            semantics[raw.get('types',{}).get('semantic','UNKNOWN')]+=1
        details=Counter(r['types']['detailed'] for r in samples)
        for axis,dist in [('structural',coverage),('semantic',semantics),('detailed',details)]:
            for label,n in sorted(dist.items()):
                summaries.append(dict(cohort=name,commit=commit,axis=axis,type=label,n=n,share_pct=100*n/len(samples),
                    metadata_source='QID/image ID/question/gold bound to original GQA val_balanced parquet; image SHA checked when local',
                    current_qid_image_join=len(joined)))
        current_sha={r['image_sha256'] for r in current}
        bindings.append(dict(cohort=name,commit=commit,samples=len(samples),manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
            strict_question_id_image_sha_matches=len(joined),current_image_overlap_questions=sum(r['image_sha256'] in current_sha for r in samples),
            type_coverage=dict(coverage),semantic_coverage=dict(semantics),detailed_coverage=dict(details),
            annotation_path=str(annotation),annotation_sha256=sha(annotation),metadata_bound_questions=len(samples),
            image_files_sha_checked=len(verified_images),
            cross_cohort_score_pooling_allowed=False,reason='Different cohort/prompts/scorers/interventions; only bound IDs support per-question comparisons'))
        # Compare restoration by official historical labels only, never label val questions from testdev by question shape.
        for axis in ['structural','semantic']:
            gg=defaultdict(list)
            for r in records:
                qid=str(r['question_id'])
                if qid not in manifest_index:
                    continue
                raw=manifest_index[qid]
                assert r['image_sha256']==raw['image_sha256']
                if 'All-Writer__correct' in r and 'Keep-L0-L21__correct' in r:
                    a=float(r['Keep-L0-L21__correct']=='True');b=float(r['All-Writer__correct']=='True')
                elif 'F__correct' in r and 'E__correct' in r:
                    a=float(r['F__correct']=='True');b=float(r['E__correct']=='True')
                elif 'All_Writer_correct' in r and 'Keep_correct' in r:
                    a=float(r['Keep_correct']);b=float(r['All_Writer_correct'])
                elif 'All_Writer__correct' in r and 'Keep__correct' in r:
                    a=float(r['Keep__correct']=='True');b=float(r['All_Writer__correct']=='True')
                else:
                    continue
                gg[raw.get('types',{}).get(axis,'UNKNOWN')].append(dict(image_sha256=r['image_sha256'],native_score=a,svh_score=b))
            for label,group in gg.items():
                summaries.append(dict(cohort=name,commit=commit,axis=axis+'_deep_writer_restoration',type=label,
                    **paired(group),interpretation='Within historical cohort only; not current SVH vs Native'))
    table(OUT/'gqa_historical_type_comparison.csv',summaries)
    write(OUT/'gqa_historical_binding.json',bindings)
    return bindings


def main():
    official=json.loads(GQA_PATH.read_text())
    protocol=json.loads((FIRST/'protocol.json').read_text())
    assert len(official)==12578
    for path,h in protocol['scorer_sources'].items():
        assert sha(path)==h
    with gzip.open(protocol['inventory_path'],'rt') as stream:
        inventory={r['key']:r for r in map(json.loads,stream) if r['benchmark']=='GQA'}
    assert len(inventory)==12578
    pairs=defaultdict(dict)
    for state,r in original_records(['GQA']):
        assert state=='COMPLETE' and r['condition'] in ['Native','SVH']
        row=inventory[r['key']]
        qid=str(r['question_id'])
        raw=official[qid]
        assert row['question_id']==qid and row['image_id']==raw['imageId']
        assert r['question']==row['question']==raw['question']
        assert r['gold']==row['gold']==raw['answer']
        assert r['image_sha256']==row['image_sha256']
        assert r['protocol_sha256']==sha(FIRST/'protocol.json')
        assert r['prompt_sha256']==row['prompt_sha256']
        assert digest(r['input_binding'])==r['input_binding_sha256']
        strict=float(r['text'].strip().lower()==raw['answer'].strip().lower())
        assert strict==r['scoring']['score']
        assert r['condition'] not in pairs[qid]
        pairs[qid][r['condition']]={k:r[k] for k in ['text','token_ids','image_sha256','input_binding_sha256',
            'input_length','image_tokens','empty','truncated','eos','prompt_sha256','protocol_sha256']}
        pairs[qid][r['condition']]['score']=strict
    assert set(pairs)==set(official)
    images={r['image_path']:r['image_sha256'] for r in inventory.values()}
    for path,h in images.items():
        assert sha(path)==h,('Image binding changed',path)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(protocol['model'],local_files_only=True)
    data=[]
    for qid in sorted(pairs):
        p=pairs[qid]
        assert set(p)=={'Native','SVH'}
        a,b=p['Native'],p['SVH']
        assert a['input_binding_sha256']==b['input_binding_sha256']
        raw=official[qid]
        qids=processor.tokenizer(raw['question'],add_special_tokens=False)['input_ids']
        item=dict(question_id=qid,image_id=raw['imageId'],image_sha256=a['image_sha256'],question=raw['question'],gold=raw['answer'],
            native_text=a['text'],svh_text=b['text'],native_score=a['score'],svh_score=b['score'],
            first_token_flip=a['token_ids'][:1]!=b['token_ids'][:1],input_length=a['input_length'],image_tokens=a['image_tokens'],
            question_tokens=len(qids),gold_words=len(raw['answer'].split()),
            yes_no=raw['answer'] in ['yes','no'],empty=a['empty'] or b['empty'],truncated=a['truncated'] or b['truncated'],
            eos_both=a['eos'] and b['eos'],input_binding_sha256=a['input_binding_sha256'],prompt_sha256=a['prompt_sha256'],
            source_protocol_sha256=a['protocol_sha256'],**classify(raw))
        data.append(item)
    table(OUT/'gqa_per_question_types.csv',data)
    overall=paired(data)
    assert overall['n']==12578
    total_delta=overall['delta_pp']
    total_harms=overall['harms']
    main=[]
    detailed=[]
    all_ci=[]
    for axis in ['structural','semantic','detailed']:
        groups=defaultdict(list)
        for r in data:
            groups[r[axis]].append(r)
        for label,group in sorted(groups.items()):
            stats=paired(group)
            contribution=len(group)/len(data)*stats['delta_pp']
            row=dict(axis=axis,type=label,**stats,share_pct=100*len(group)/len(data),
                net_lost_questions=stats['harms']-stats['rescues'],
                share_all_harms_pct=100*stats['harms']/total_harms,
                contribution_pp=contribution,share_net_loss_pct=100*contribution/total_delta,
                status='OFFICIAL_METADATA_EXPLORATORY')
            main.append(row)
            all_ci.append(row)
        assert abs(sum(r['contribution_pp'] for r in main if r['axis']==axis)-total_delta)<1e-10
    query=[r for r in data if r['structural']=='query']
    nonquery=[r for r in data if r['structural']!='query']
    for name,group in [('Query',query),('NonQuery',nonquery)]:
        all_ci.append(dict(axis='QueryVsNonQuery',type=name,**paired(group),status='OFFICIAL_STRUCTURAL_SPLIT'))
    for axis in ['semantic','detailed','query_argument']:
        groups=defaultdict(list)
        for r in query:
            groups[r[axis]].append(r)
        for label,group in sorted(groups.items()):
            detailed.append(dict(axis=axis,type=label,**paired(group),
                status='OFFICIAL_TYPE' if axis!='query_argument' else 'OFFICIAL_PROGRAM_ARGUMENT',
                share_query_pct=100*len(group)/len(query)))
    # Freeze operational diagnostics in code, not by looking for favorable subgroups.
    diagnostics={
        'locate_then_attribute':lambda r:str(r['locate_then_attribute']),
        'attribute_query_location':lambda r:('located_by_relate_or_filter' if r['locate_any'] else 'direct_select') if r['semantic']=='attr' else None,
        'relation_hops':lambda r:str(min(r['relate_count'],2)) + ('+' if r['relate_count']>=2 else ''),
        'program_depth':lambda r:'1-2' if r['program_depth']<=2 else '3-4' if r['program_depth']<=4 else '5+',
        'gold_words':lambda r:'single' if r['gold_words']==1 else 'multiple',
        'answer_form':lambda r:'yes_no' if r['yes_no'] else 'open',
        'image_tokens':lambda r:'<=256' if r['image_tokens']<=256 else '257-512' if r['image_tokens']<=512 else '>512',
        'question_token_length':lambda r:'<=10' if r['question_tokens']<=10 else '11-20' if r['question_tokens']<=20 else '>20',
        'count_comparison_program':lambda r:str(any(x['operation'].startswith(('count','same','different','common','choose less','choose bigger','choose smaller','choose more')) for x in r['program_json']))}
    for axis,fun in diagnostics.items():
        groups=defaultdict(list)
        for r in query:
            label=fun(r)
            if label is not None:
                groups[label].append(r)
        for label,group in sorted(groups.items()):
            detailed.append(dict(axis=axis,type=label,**paired(group),status='HEURISTIC_DIAGNOSTIC',
                source='Official program/token count; operational grouping, not official task label',share_query_pct=100*len(group)/len(query)))
    # Explicit relative paired effects; each bootstrap draw samples all relevant images together.
    contrasts=[]
    def contrast(label,left,right):
        import numpy as np
        # Difference in SVH-Native deltas; group sizes are random under shared cluster draws.
        union=sorted({r['image_sha256'] for r in left+right})
        ix={h:i for i,h in enumerate(union)}
        mat=np.zeros((len(union),4))
        for offset,group in [(0,left),(2,right)]:
            for r in group:
                mat[ix[r['image_sha256']],offset]+=r['svh_score']-r['native_score']
                mat[ix[r['image_sha256']],offset+1]+=1
        rng=np.random.default_rng(SEED)
        vals=[]
        for begin in range(0,20000,128):
            draw=rng.integers(0,len(union),(min(128,20000-begin),len(union)))
            sums=mat[draw].sum(axis=1)
            good=(sums[:,1]>0)&(sums[:,3]>0)
            vals.extend((100*(sums[good,0]/sums[good,1]-sums[good,2]/sums[good,3])).tolist())
        low,high=np.quantile(vals,[.025,.975])
        contrasts.append(dict(comparison=label,left_n=len(left),right_n=len(right),
            delta_difference_pp=100*(sum(r['svh_score']-r['native_score'] for r in left)/len(left)-sum(r['svh_score']-r['native_score'] for r in right)/len(right)),
            ci_low_pp=float(low),ci_high_pp=float(high),status='EXPLORATORY_DIFFERENCE_IN_PAIRED_EFFECTS'))
    contrast('Query - NonQuery',query,nonquery)
    for t in ['verify','choose','compare','logical']:
        contrast('Query - '+t,query,[r for r in data if r['structural']==t])
    attr=[r for r in query if r['semantic']=='attr']
    contrast('Query semantic attr - other Query',attr,[r for r in query if r['semantic']!='attr'])
    contrast('Located attribute - direct attribute',[r for r in attr if r['locate_any']],[r for r in attr if not r['locate_any']])
    table(OUT/'gqa_type_distribution.csv',main)
    table(OUT/'gqa_query_subtype_analysis.csv',detailed)
    table(OUT/'gqa_paired_type_bootstrap.csv',all_ci)
    table(OUT/'gqa_type_effect_contrasts.csv',contrasts)
    # Fixed hash selection inside each predefined cell; include harms AND rescues.
    cases=[]
    for structural in ['query','verify','choose','compare','logical']:
        for transition in ['harm','rescue']:
            group=[r for r in data if r['structural']==structural and
                (r['native_score']>r['svh_score'] if transition=='harm' else r['native_score']<r['svh_score'])]
            for r in sorted(group,key=lambda r:digest([SEED,'case',r['question_id']]))[:4]:
                cases.append(dict(selection_cell=structural+'|'+transition,selection='first 4 SHA-ranked cases, fixed seed',**r))
    sub=[r for r in detailed if r['axis']=='detailed' and r['n']>=30]
    for summary in sorted(sub,key=lambda r:(abs(r['delta_pp']),r['type']))[:3]:
        group=[r for r in query if r['detailed']==summary['type']]
        for r in sorted(group,key=lambda r:digest([SEED,'counterexample',r['question_id']]))[:2]:
            cases.append(dict(selection_cell='small_net_drop_subtype|'+summary['type'],selection='descriptive smallest absolute drop n>=30; then fixed hash',**r))
    table(OUT/'gqa_error_cases.csv',cases)
    historical=history(data)
    audit=dict(status='PASS_CPU_BINDING_AND_RESCORING',question_count=len(data),image_clusters=len({r['image_sha256'] for r in data}),
        native=overall['native_acc'],svh=overall['svh_acc'],delta_pp=overall['delta_pp'],harms=overall['harms'],rescues=overall['rescues'],
        overall_ci=[overall['ci_low_pp'],overall['ci_high_pp']],
        official_metadata_sha256=sha(GQA_PATH),original_protocol_sha256=sha(FIRST/'protocol.json'),
        metadata_fields=['types.structural','types.semantic','types.detailed','semantic program operations/arguments/dependencies'],
        structural_coverage=dict(Counter(r['structural'] for r in data)),unknown_rows=sum(r['structural']=='UNKNOWN' for r in data),
        all_image_files_sha256_verified=len(images),all_paired_input_bindings_equal=True,all_scores_recomputed_exactly=True,
        contributions_sum_to_global_delta=True,bootstrap_replicates=20000,heuristics_separately_labeled=True,
        historical_binding=historical,no_gpu_calls=True)
    write(OUT/'gqa_metadata_binding_audit.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    main()
