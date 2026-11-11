"""Freeze samples and a bounded search before inspecting candidate outputs."""
from collections import Counter, defaultdict
from common import *


def grid():
    candidates = {'K0': []}
    for lo in range(8, 15):
        candidates[f'W{lo}_{lo+13}'] = list(range(lo, lo+14))
    for lo, hi in [(11,23), (11,22), (11,21), (12,21), (14,21), (16,21),
                   (18,21), (20,21), (21,21), (13,24), (15,24), (17,24),
                   (19,24), (21,24), (23,24)]:
        candidates[f'B{lo}_{hi}'] = list(range(lo, hi+1))
    candidates['Prior4'] = [21,23,24,26]
    candidates['Prior2'] = [24,26]
    candidates['Prior1'] = [26]
    assert len(candidates) == len({tuple(v) for v in candidates.values()})
    return candidates


def main():
    from transformers import AutoProcessor
    assert not (OUT / 'protocol.json').exists()
    disk()
    planned = inventory()
    gqa = json.loads(Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official/testdev_balanced_questions.json').read_text())
    p2protocol = json.loads((P2/'p2_full_protocol.json').read_text())
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    suffix = p2protocol['suffix_exact_from_prior']
    rows = []
    for r in planned:
        if r['benchmark'] not in ['TextVQA', 'GQA', 'ScienceQA-IMG', 'MMBench']:
            continue
        r = dict(r)
        if r['benchmark'] in ['ScienceQA-IMG', 'MMBench']:
            r['body'] += '\n' + suffix
            text = processor.apply_chat_template([{'role':'user','content':[
                {'type':'image','image':r['image_path']},{'type':'text','text':r['body']}]}], tokenize=False, add_generation_prompt=True)
            r['input_length'] = len(processor.tokenizer(text, add_special_tokens=False)['input_ids']) + r['image_tokens'] - 1
            r['prompt_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            r['prompt_protocol'] = 'P2_INDEPENDENT_SHORT_ANSWER_PROTOCOL'
        else:
            r['prompt_protocol'] = 'FROZEN_PLAN_B_ORIGINAL'
        r['analysis_task'] = ('GQA-Query' if gqa[r['question_id']]['types']['structural']=='query' else 'GQA-NonQuery') if r['benchmark']=='GQA' else r['benchmark']
        rows.append(r)
    # Union image SHA and Circular membership across all tasks before splitting.
    parent = {}
    def root(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]
    for r in rows:
        image = 'I:' + r['image_sha256']
        root(image)
        if r['benchmark'] == 'MMBench':
            a,b = root(image),root('C:'+str(r['circular_group']))
            parent[max(a,b)] = min(a,b)
    prior_images = set()
    for path in [BASE/'vlm_research/results/qwen3vl_svhr_reaccess_layer_search/Discovery_manifest.json',
                 BASE/'vlm_research/results/qwen3vl_svhr_reaccess_layer_search/Validation_manifest.json',
                 PROMPT/'vlm_research/results/qwen3vl_short_answer_prompt_ablation/sample_manifest.json']:
        if path.exists():
            obj = json.loads(path.read_text())
            pool = obj if isinstance(obj,list) else obj.get('samples',[])
            prior_images.update(r['image_sha256'] for r in pool)
    assignment = {}
    for r in rows:
        cluster = root('I:'+r['image_sha256'])
        u = int(digest([SEED,'SVHR_LOCAL_BUDGET',cluster])[:16],16)/2**64
        split = 'Discovery' if u < .27 else 'Validation' if u < .62 else 'Held-out'
        if r['image_sha256'] in prior_images and split == 'Held-out':
            split = 'Discovery'
        assignment.setdefault(cluster,[]).append(split)
    assignment = {k: ('Discovery' if 'Discovery' in v else v[0]) for k,v in assignment.items()}
    targets = {'Discovery':[250,120,60,40,30], 'Validation':[500,250,100,100,100],
               'Held-out':[500,250,100,100,100]}
    selected = []
    for split, counts in targets.items():
        for task, count in zip(TASKS,counts):
            pool = [r for r in rows if r['analysis_task']==task and assignment[root('I:'+r['image_sha256'])]==split]
            groups = defaultdict(list)
            for r in pool:
                groups[r['circular_group'] if task=='MMBench' else r['key']].append(r)
            ordered = sorted(groups,key=lambda k:digest([SEED,split,task,k]))
            assert len(ordered)>=count,(split,task,len(ordered),count)
            for k in ordered[:count]:
                for r in groups[k]:
                    selected.append(dict(r,split=split,cluster=root('I:'+r['image_sha256'])))
    # Cluster-balanced device placement is fixed, never based on model outputs.
    loads = [0,0]
    groups = defaultdict(list)
    for r in selected:
        groups[r['cluster']].append(r)
    for cluster in sorted(groups,key=lambda c:(-sum(r['input_length'] for r in groups[c]),digest(c))):
        shard = min(range(2),key=lambda i:(loads[i],i))
        for r in groups[cluster]:
            r['shard']=shard
        loads[shard] += sum(r['input_length'] for r in groups[cluster])
    images = {r['image_path']:r['image_sha256'] for r in selected}
    for path, expected in images.items():
        assert sha(path)==expected
    hist = historical({r['key'] for r in selected})
    baseline = Store(OUT/'historical.sqlite')
    for r in selected:
        for cond in ['Native','SVH']:
            rec = hist.get((r['key'],cond))
            assert rec is not None,(r['key'],cond,'MISSING_BASELINE')
            assert rec['image_sha256']==r['image_sha256']
            if rec.get('status',rec.get('state','COMPLETE'))=='COMPLETE':
                score = scored(r,rec['text'],rec['truncated'])
                assert score['score']==rec.get('score',rec.get('scoring',{}).get('score'))
            else:
                r['historical_blocked_status']=rec.get('status',rec.get('state'))
            baseline.put(r['key']+'|'+cond,rec)
    baseline.conn.close()
    write(OUT/'sample_manifest.json',selected)
    write(OUT/'candidate_grid.json',grid())
    local = list(ROOT.glob('*.py'))
    inherited = [BASE/'vlm_research'/p for p in [
        'qwen3vl_c3v2_physical/engine.py','qwen3vl_svh_sqa7908_repair/engine.py',
        'qwen3vl_svh_qk_repair/engine.py','qwen3vl_svh_l28_repair/engine.py',
        'qwen3vl_svh_numerical_repair/engine.py','qwen3vl_c3v2_numerical_repair2/repair.py',
        'qwen3vl_c3v2_numerical_debug/repair.py','qwen3vl_svhr_search/physical.py',
        'qwen3vl_svh_preflight48/freeze.py','qwen3vl_svh_preflight48/score.py']]
    sources = {str(path):sha(path) for path in local+inherited}
    model = json.loads((FIRST/'protocol.json').read_text())
    p = dict(name='QWEN3VL_SVHR_LOCAL_BUDGET_SEARCH',seed=SEED,mode='B_ATE_UNIFIED_HF_NOT_AUTHOR_OFFICIAL',
        revision=MODEL.name,model=str(MODEL),dtype='bfloat16',attention='eager',visual_pixels=[65536,2097152],
        historical_strict_status='FAIL_RETAINED',sources=sources,manifest_sha256=sha(OUT/'sample_manifest.json'),
        frozen_model_files=model['model_files'],gpu_uuids=UUIDS,writer_layers=list(range(22)),exit_before=22,
        answer_skip_layers=list(range(1,7)),main_view='EXACT_FROZEN_SVH_PREFIX_DECODE_ONLY_REACCESS',
        auxiliary_prefill_layers=list(range(11,25)),auxiliary_view='FIXED_P1_L11_L24_COMPARE_DECODE_MARGINAL',
        main_candidates_max=40,auxiliary_not_ranked=True,prior_known_images=len(prior_images),
        candidate_source='Predeclared windows and budgets; bounded adaptive deletion/sparse search after Discovery only',
        discovery_ranking='TextVQA first; protection tasks reject extra point loss >1pp vs frozen SVH for efficiency',
        validation_selection='3-5 Discovery finalists frozen before Validation; no returning to search',
        final_selection='Best TextVQA; BestEfficiency within1pp of best and protection-loss<=1pp; MinK best K<=4',
        efficiency_order='Measured fixed32 E2E; if indistinguishable compare KV then peak VRAM then K',
        bootstrap=dict(repeats=20000,seed=SEED,cluster='Image SHA, complete Circular groups'),
        heldout='Used once after final configurations frozen; historical cohort, not independent external validation',
        disk_min_free_bytes=2*2**30,output_max_bytes=1500*2**20,
        notes=['P1 stays image-OFF in main view','Dense search caches real visual KV but uses frozen physical prefix',
               'Original prefix computed once per image request and forked into disjoint per-condition caches',
               'All temporary QK/AV padding and sidecar construction charged in physical timing'])
    write(OUT/'protocol.json',p)
    write(OUT/'freeze_summary.json',dict(counts={s:dict(Counter(r['analysis_task'] for r in selected if r['split']==s)) for s in targets},
        images=len(images),loads=loads,candidates=len(grid()),free_bytes=disk(),historical_records=len(hist)))
    print(json.dumps(json.loads((OUT/'freeze_summary.json').read_text()),indent=2))


if __name__=='__main__':
    main()
