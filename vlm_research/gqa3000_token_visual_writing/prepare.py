"""Freeze every input-only decision and Gold boundary before any model evaluation."""
import argparse,collections,io,shutil,subprocess,time,unittest
from itertools import combinations
import networkx as nx
from PIL import Image
from transformers import AutoProcessor
from paper_rq1_single_layer_gqa6.run import tensor_hashes
from .common import *
from .selection import make_selection,STOP

def cpu():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','relay_source_causality.tests','visual_write_window.tests','semantic_visual_writers.tests','gqa3000_token_visual_writing.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not torch.cuda.is_initialized()

def image_shards(samples):
    clusters=collections.defaultdict(list)
    for s in samples:clusters[s['image_sha256']].append(s)
    totals=[0,0];tasks=[collections.Counter(),collections.Counter()];assign={}
    for im,ss in sorted(clusters.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        counts=collections.Counter(s['task'] for s in ss)
        rank=min(range(2),key=lambda k:(totals[k],sum(tasks[k][t]*n for t,n in counts.items()),k))
        assign[im]=rank;totals[rank]+=len(ss);tasks[rank].update(counts)
    return dict(image_assignments=assign,question_counts=totals,task_counts=[dict(c) for c in tasks],
        image_counts=[sum(k==rank for k in assign.values()) for rank in range(2)],whole_image_clusters=True,disjoint=True)

def swap_pairs(samples,old_images):
    bytext=collections.defaultdict(list)
    for s in samples:bytext[(s['question'],digest(s['input_mapping']['prompt_ids']))].append(s)
    edges=[]
    for group in bytext.values():
        for a,b in combinations(sorted(group,key=lambda s:(s['image_sha256'],s['question_id'])),2):
            if a['image_sha256']!=b['image_sha256'] and a['answer']!=b['answer']:edges.append((a,b))
    def matching(edges,excluded,only_new):
        g=nx.Graph();chosen={}
        for a,b in sorted(edges,key=lambda ab:(ab[0]['image_sha256'],ab[1]['image_sha256'],ab[0]['question_id'],ab[1]['question_id'])):
            ia,ib=a['image_sha256'],b['image_sha256']
            if ia in excluded or ib in excluded or only_new and (ia in old_images or ib in old_images):continue
            edge=tuple(sorted([ia,ib]));weight=int(hashlib.sha256((a['question_id']+'|'+b['question_id']).encode()).hexdigest()[:16],16)
            if edge not in chosen or weight>chosen[edge][0]:chosen[edge]=(weight,a,b)
        for (ia,ib),(weight,a,b) in sorted(chosen.items()):g.add_edge(ia,ib,weight=weight)
        result=[]
        for ia,ib in nx.max_weight_matching(g,maxcardinality=True,weight='weight'):
            _,a,b=chosen[tuple(sorted([ia,ib]))];result.append((a,b))
        return sorted(result,key=lambda ab:(ab[0]['question_id'],ab[1]['question_id']))
    selected=matching(edges,set(),True)[:40];used={s['image_sha256'] for ab in selected for s in ab}
    if len(selected)<40:selected+=matching(edges,used,False)[:40-len(selected)]
    return dict(target=40,actual=len(selected),candidate_edges=len(edges),selection='Maximum cardinality matching of different-Gold, same-prompt image pairs; prefer images outside Semantic240, then deterministic fallback; no image used twice.',
        pairs=[dict(pair_id=f'pair_{i:03d}',question=a['question'],question_ids=[a['question_id'],b['question_id']],
            sample_keys=[key(a),key(b)],image_sha256=[a['image_sha256'],b['image_sha256']],gold=[a['answer'],b['answer']],
            prompt_ids_sha256=digest(a['input_mapping']['prompt_ids']),both_images_outside_exploration=all(s['image_sha256'] not in old_images for s in [a,b])) for i,(a,b) in enumerate(selected)])

def design():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection_index.json').exists(),'Input-only design already exists; do not replace it'
    torch.set_num_threads(4);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    assert sha256(HISTORY/'sample_manifest.json')==MANIFEST_SHA
    for commit,path,expected in [(BASE,'vlm_research/results/rq2_visual_sufficiency_full3000/sample_manifest.json',MANIFEST_SHA),
        (SEM_COMMIT,'vlm_research/results/semantic_visual_writers_240/sample_manifest.json',sha256(SEM/'sample_manifest.json'))]:
        assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT.parent)).hexdigest()==expected
    shutil.copyfile(HISTORY/'sample_manifest.json',OUT/'sample_manifest.json')
    samples=read(OUT/'sample_manifest.json')['samples'];assert len(samples)==3000 and len({s['question_id'] for s in samples})==3000
    assert all(sum(s['task']==t for s in samples)==500 for t in TASKS)
    old=read(SEM/'sample_manifest.json')['samples'];old_ids={s['question_id'] for s in old};old_images={s['image_sha256'] for s in old}
    old_specs=read(SEM/'token_group_manifest.json')['main'];index={};gold={};inputs={};stats=collections.Counter();nested=collections.Counter();image_hashes={};started=time.perf_counter()
    for i,s in enumerate(samples):
        m=s['input_mapping'];chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
        assert digest(chat)==m['chat_prompt_sha256'];l=layout(s)
        spec=make_selection(processor.tokenizer,s['question'],chat,m['prompt_ids'],l['I'],l['R'])
        if key(s) in old_specs:
            assert spec['natural']==old_specs[key(s)]['selection'] and spec['legacy_selection_binding']==old_specs[key(s)]['selection_binding'];stats['historical_sets_exact']+=1
        expanded=chat.replace('<image>','<image>'*576)
        combined=processor.tokenizer(expanded+' '+s['answer'],add_special_tokens=True)['input_ids']
        assert combined[:len(m['prompt_ids'])]==m['prompt_ids'],'Gold boundary changed a prompt token'
        ids=combined[len(m['prompt_ids']):]
        assert ids and all(v not in [0,1,2,32000] for v in ids) and combined.count(1)==1 and 2 not in combined
        assert processor.tokenizer.decode(ids).strip()==s['answer'].strip()
        gold[key(s)]=dict(answer=s['answer'],token_ids=ids,target_query_positions=[len(m['prompt_ids'])-1+t for t in range(len(ids))],
            context_ids_sha256=digest(combined),prefix_ids_sha256=digest(m['prompt_ids']),boundary_text='one ASCII space plus official Gold',
            bos_added=False,eos_in_primary_nll=False,legacy_gold_ids_equal=ids==m['gold_continuation_ids'])
        stats['legacy_gold_ids_equal']+=int(ids==m['gold_continuation_ids']);stats['gold_tokens']+=len(ids);stats['multi_token_gold']+=int(len(ids)>1)
        if s['image_path'] not in image_hashes:image_hashes[s['image_path']]=sha256(s['image_path'])
        assert image_hashes[s['image_path']]==s['image_sha256']
        with Image.open(s['image_path']) as source:image=source.convert('RGB')
        batch=processor(images=image,text=chat,return_tensors='pt').to('cpu',dtype=torch.float16)
        assert batch['input_ids'][0].tolist()==m['prompt_ids']
        inputs[key(s)]=tensor_hashes(batch)
        path=OUT/'token_manifests'/(key(s)+'.json');atomic_json(path,spec)
        index[key(s)]=dict(path=str(path.relative_to(OUT)),sha256=sha256(path),unique_count=len(spec['unique']),selection_binding=spec['selection_binding'])
        stats['unique_main_evaluations']+=len(spec['unique']);stats['natural_k_zero']+=int(spec['natural']['k']==0)
        stats['curve_budget_shortfalls']+=sum(not v['exact_target_matched'] for v in spec['budgets'].values())
        for name,ok in spec['nested'].items():nested[name]+=int(not ok)
        if (i+1)%250==0:print('INPUT DESIGN',i+1,'/3000','unique',stats['unique_main_evaluations'],'seconds',round(time.perf_counter()-started,1),flush=True)
    assert stats['historical_sets_exact']==240
    shards=image_shards(samples);confirm=[key(s) for s in samples if s['question_id'] not in old_ids and s['image_sha256'] not in old_images]
    split=dict(confirmatory_keys=confirm,confirmatory_questions=len(confirm),confirmatory_images=len({s['image_sha256'] for s in samples if key(s) in set(confirm)}),
        exploration_question_ids=sorted(old_ids),exploration_image_sha256=sorted(old_images),exploration_manifest_sha256=sha256(SEM/'sample_manifest.json'),
        task_counts=dict(collections.Counter(s['task'] for s in samples if key(s) in set(confirm))),excluded_questions=3000-len(confirm),
        rule='Exclude all historical Semantic240 question IDs AND every question on any of its image SHA256 clusters; recomputed from this specific exploration cohort.')
    swaps=swap_pairs(samples,old_images)
    for pair in swaps['pairs']:
        a,b=[read(OUT/index[k]['path']) for k in pair['sample_keys']]
        assert a['writers']==b['writers'] and a['selection_binding']==b['selection_binding']
    smoke_ids=['00150746','11697551','11127723'];multi=next(s['question_id'] for s in sorted(samples,key=lambda s:s['question_id']) if len(gold[key(s)]['token_ids'])>1)
    if multi not in smoke_ids:smoke_ids.append(multi)
    atomic_json(OUT/'selection_index.json',index);atomic_json(OUT/'gold_manifest.json',gold);atomic_json(OUT/'input_hashes.json',inputs)
    atomic_json(OUT/'split_integrity.json',split);atomic_json(OUT/'shards.json',shards);atomic_json(OUT/'image_swap_manifest.json',swaps)
    atomic_json(OUT/'input_design_audit.json',dict(passed=True,stats=dict(stats),non_nested_questions=dict(nested),smoke_ids=smoke_ids,
        figure_A_question_id='00150746',all_conditions=CONDITIONS,all_three_thousand_retained=True,legacy_natural_selector_exact=True,
        budget_rule='Largest whole Question suffix token count <= floor(percent*Question_tokens), shared by every strategy. All random policies use exact whole-group subsets, not independent subword sampling.',
        question_strata='Token-index thirds; exact per-third token quotas equal to Semantic-k; whole-word groups stay intact.',
        primary_gold_nll='Official Gold appended with one ASCII space to the unchanged expanded prompt; no extra BOS or EOS; FP16 model logits, CPU FP64 log_softmax and answer-token mean. Gold contexts are independent of free generation.',
        stats_definitions='Random seeds averaged within each question, then paired whole-image cluster resampling with question weighting. Never 9000 independent questions.'))
    print('DESIGN COMPLETE',dict(stats),'confirm',len(confirm),'swap_pairs',swaps['actual'],'shards',shards['question_counts'],flush=True)

def freeze():
    cpu_record=read(OUT/'cpu_validation.json');assert cpu_record['passed'] and cpu_record['sources']==sources() and read(OUT/'input_design_audit.json')['passed']
    assert not (OUT/'protocol.json').exists();paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(ROOT/'results/layer_semantic_joint_writing_240/protocol.json')
    for n,info in old['model_files'].items():assert sha256(n)==info['sha256'],n
    index=read(OUT/'selection_index.json')
    for info in index.values():assert sha256(OUT/info['path'])==info['sha256']
    gpus=choose_gpus();names=['cpu_validation.json','sample_manifest.json','selection_index.json','gold_manifest.json','input_hashes.json','split_integrity.json','shards.json','image_swap_manifest.json','input_design_audit.json','gpu_selection.json']
    p=dict(status='FROZEN',base_commit=BASE,semantic_selector_reference=SEM_COMMIT,mask_reference='6b9467d6fb4e0f8535040809bee60397ca798ea8',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),sources=sources(),bindings={n:sha256(OUT/n) for n in names},
        model_revision=REVISION,model_path=str(MODEL),model_files=old['model_files'],transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],
        transformers='4.49.0',model_dtype='float16',attention='eager',generation=dict(do_sample=False,max_new_tokens=32,use_cache=True,logits_to_keep=1),
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),gpus=gpus,conditions=CONDITIONS,natural_conditions=NATURAL,curve_conditions=CURVE,
        budgets=BUDGETS,random_seeds=SEEDS,stop_words=sorted(STOP),main_questions=3000,logical_evaluations=3000*len(CONDITIONS),
        unique_evaluations=read(OUT/'input_design_audit.json')['stats']['unique_main_evaluations'],
        mask='Pre-softmax hard -inf only on original R->Image keys for unselected R, and original P-1 plus all real decode queries -> original576 image keys, at all32 layers. Other edges and all computations/KV native. All-Writer != native Full.',
        nll=dict(primary='Mean official Gold answer-token NLL, exclude EOS',tokenization='Whole expanded native prompt plus one ASCII space plus official Gold; exact unchanged prompt prefix; no added BOS or EOS',
            model_logits='FP16 with logits_to_keep=1 at every step',metric_reduction='CPU float64 log_softmax of saved FP16 full-vocabulary logits',
            target_positions='Original P-1 predicts first Gold token, subsequent genuine decode steps predict subsequent Gold tokens',independent_greedy_and_teacher_cache=True,first_logits_strictly_equal=True),
        statistics=dict(repetitions=20000,seed=BOOTSTRAP_SEED,unit='Image SHA256 whole clusters, paired conditions, question-weighted',
            populations=['all3000_descriptive','semantic240_image_disjoint_confirmatory'],ordinary_ci='Pointwise two-sided percentile95%, unadjusted',
            primary_family=[dict(name='semantic_selection_advantage',metric='Gold NLL',contrast='Semantic-k minus within-question mean of three Question-Random-k seeds',null='difference >= 0',alternative='difference < 0'),
                dict(name='semantic_quality_noninferiority',metric='Accuracy percentage points',contrast='Semantic-k minus All-Writer',null='difference <= -2pp',alternative='difference > -2pp')],
            multiplicity='Bonferroni over two prespecified one-sided tests, family alpha .05, each alpha .025. NLL upper97.5% <0; Accuracy lower2.5% >-2pp. These endpoints coincide numerically with ordinary two-sided95% endpoints; their inferential role is stated separately.',
            sensitivity='Also show whether the same NI bound exceeds -1pp, a sensitivity criterion, not a third primary test',random_means='Average seeds within question before statistics',
            curves='All3000 descriptive and confirmatory-subset supporting curves; all pointwise95% intervals shown; no curve or task-based hypothesis reselection'),
        swap=dict(conditions=['All-Writer','No-Writer','Semantic-k',f'Question-Random-k-{SEEDS[0]}','Question-Last-k'],
            record_source='Audit-reuse exact main3000 records for preselected endpoints and conditions; retain separate pair labels/statistics, never add endpoints as independent main samples',
            no_writer_check='Same prompt and original image layout: every effective greedy logit torch.equal, including EOS prediction. Teacher-forced contexts with different Gold answers are not falsely compared after prefixes diverge.'),
        dedup='Within each original question only, byte-identical writer positions share one independent generation and Gold teacher-forced evaluation; all44 logical labels retained.',
        no_actual_compute_saving=True,stop='No layer-token search, router, kernel, KV pruning or main edits.')
    atomic_json(OUT/'protocol.json',p);verify();print('FROZEN',p['code_commit'],'unique',p['unique_evaluations'],'logical',p['logical_evaluations'],'GPUs',gpus,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','design','freeze']);a=parser.parse_args();globals()[a.stage]()
