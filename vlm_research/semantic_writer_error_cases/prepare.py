"""Freeze selection and all candidate/removal choices before any new model call."""
import argparse,collections,re,shutil
from PIL import Image
from transformers import AutoTokenizer
from paper_rq1.scorers import gqa_prediction
from .common import *

TAGS={
    'negation':set('not no never neither nor without none'.split()),
    'quantity':set('all any each every both either few many much more most less least some several one two three four five how'.split()),
    'comparison':set('than same different bigger smaller larger taller shorter longer fewer equal compared'.split()),
    'logical_connector':set('and or but if whether'.split()),
    'relation_connector':set('of to in on at under above below behind beside between near next from with by through left right front'.split()),
    'interrogative':set('what which who where when why how whose'.split()),
    'auxiliary_function':set('is are was were be been being am do does did have has had can could may might must shall should will would as for then there'.split()),
    'article':set('a an the'.split())}
def tags(g):
    words={w.lower() for w in g['words']};out=[k for k,v in TAGS.items() if words&v]
    if not words and g['decoded'].strip():out.append('punctuation')
    return out or ['other']
def errtype(s,allrow,semrow):
    a=s['answer'].lower();b=(semrow if allrow['correct']=='1' else allrow)['prediction'].lower()
    if {a,b}<=set(['yes','no']):return 'yes_no_reversal'
    if a.isdigit() or a in set('zero one two three four five six seven eight nine ten'.split()):return 'numerical_answer'
    if a in set('black white red green blue yellow brown gray grey orange pink purple silver gold beige tan multicolored'.split()):return 'color_choice'
    if a in set('left right behind front above below top bottom middle inside outside'.split()):return 'relation_location'
    return 'categorical_other'
def plan():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'selection_rules.json',RULES);print('RULES WRITTEN BEFORE POOL READ',sha256(OUT/'selection_rules.json'))
def candidates(spec,qid):
    selected=set(spec['writers']['Semantic-k']);missing=[g for g in spec['groups'] if not set(g['positions'])&selected]
    q=[g for g in missing if g['question']];template=[g for g in missing if not g['question']]
    qlast=spec['question_positions'][-1];res=[]
    def add(g):
        if g is not None and g not in res:res.append(g)
    add(next((g for g in q if qlast in g['positions']),None))
    add(next((g for g in template if g['positions'][0]>qlast and g['words']),None))
    add(next((g for g in reversed(template) if spec['relay_positions'][-1] in g['positions']),None))
    priority={t:i for i,t in enumerate(RULES['tag_priority'])}
    todo=sorted(q,key=lambda g:(min(priority[t] for t in tags(g)),g['positions'][0]))
    for distinct in [True,False]:
        for g in todo:
            if len(res)>=6:break
            if g in res:continue
            if distinct and any(g['decoded'].lower()==x['decoded'].lower() for x in res):continue
            add(g)
    out=[]
    for i,g in enumerate(res):
        n=len(g['positions']);pool=[v for v in template if v['words'] and v!=g and v['positions'][-1]<spec['relay_positions'][-1] and v['decoded'].lower() not in ['answer','assistant']]
        control=subsets(pool).get(n);controlpos=flatten(control) if control is not None else None
        chosen=[v for v in spec['groups'] if set(v['positions'])<=selected]
        order=sorted(chosen,key=lambda v:tie(qid,g['positions'],v['positions']))
        one=next((v for v in order if len(v['positions'])==n),None)
        removal=[one] if one is not None else subsets(order).get(n)
        out.append(dict(id=f'single_{i+1}',group=g,tags=tags(g),added=g['positions'],
            writers=sorted(selected|set(g['positions'])),control_added=controlpos,control_writers=sorted(selected|set(controlpos)) if controlpos is not None else None,
            remove=flatten(removal) if removal is not None else None,replacement_writers=sorted((selected-set(flatten(removal)))|set(g['positions'])) if removal is not None else None,
            removal_whole_groups=removal,replacement_single_removed_group=one is not None))
    assert len(out)<=6
    for v in out:
        for k in ['writers','control_writers','replacement_writers']:
            if v[k] is not None:assert intact(v[k],spec['groups']) and set(v[k])<=set(spec['relay_positions'])
        if v['replacement_writers'] is not None:assert len(v['replacement_writers'])==len(selected)
        if v['control_added'] is not None:assert len(v['control_added'])==len(v['added'])
    return out
def select_cases():
    assert read(OUT/'selection_rules.json')==RULES and not (OUT/'case_manifest.json').exists()
    complete=history_file('completion.json');assert complete['status']=='COMPLETE'
    assert history_file('accuracy_export_validation.json')['passed'];assert history_file('validation.json')['passed']
    samples=history_file('sample_manifest.json')['samples'];lookup={s['question_id']:s for s in samples}
    confirm=set(history_file('split_integrity.json')['confirmatory_keys']);index=history_file('selection_index.json')
    rows={};wanted=['All-Writer','Semantic-k','Question-Last-k']+[f'Question-Random-k-{s}' for s in RANDOM_SEEDS]
    for part in history_file('per_question_index.json')['partitions']:
        path=HIST/part['path'];assert sha256(path)==part['sha256']==complete['artifacts'][part['path']]==gitsha(path)
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['condition'] in wanted:
                    assert int(r['correct'])==int(gqa_prediction(r['text'])==r['gold']);rows.setdefault(r['question_id'],{})[r['condition']]=r
    pools={'harm':[],'rescue':[]}
    for qid,s in lookup.items():
        a=rows[qid]['All-Writer'];b=rows[qid]['Semantic-k']
        kind='harm' if a['correct']=='1' and b['correct']=='0' else 'rescue' if a['correct']=='0' and b['correct']=='1' else None
        if kind is None:continue
        sk=s['task']+'__'+qid;info=index[sk];path=HIST/info['path'];assert sha256(path)==info['sha256']==gitsha(path);spec=read(path)
        missing=[g for g in spec['groups'] if g['question'] and not set(g['positions'])&set(spec['writers']['Semantic-k'])]
        feature=set(t for g in missing for t in tags(g));binid=int(4*len(spec['writers']['Semantic-k'])/len(spec['relay_positions']))
        pools[kind].append(dict(question_id=qid,sample=s,spec=spec,rows=rows[qid],kind=kind,confirmatory=sk in confirm,
            error_type=errtype(s,a,b),feature_tags=sorted(feature),budget_bin=binid))
    assert len(pools['harm'])==184 and len(pools['rescue'])==87
    chosen=[];images=set();features=set();types=set();bins=set();tasks=set()
    def include(v):
        chosen.append(v);images.add(v['sample']['image_sha256']);features.update(v['feature_tags']);types.add(v['error_type']);bins.add(v['budget_bin']);tasks.add(v['sample']['task'])
    include(next(v for v in pools['harm'] if v['question_id']==RULES['mandatory_question_id']))
    for task in TASKS:
        while sum(v['kind']=='harm' and v['sample']['task']==task for v in chosen)<2:
            eligible=[v for v in pools['harm'] if v['sample']['task']==task and v['sample']['image_sha256'] not in images]
            assert eligible,task
            best=min(eligible,key=lambda v:(not v['confirmatory'],v['error_type'] in types,-len(set(v['feature_tags'])-features),v['budget_bin'] in bins,tie(task,v['question_id'])))
            include(best)
    for i in range(4):
        eligible=[v for v in pools['rescue'] if v['sample']['image_sha256'] not in images]
        best=min(eligible,key=lambda v:(not v['confirmatory'],v['sample']['task'] in {c['sample']['task'] for c in chosen if c['kind']=='rescue'},v['error_type'] in types,-len(set(v['feature_tags'])-features),tie('rescue',v['question_id'])))
        include(best)
    chosen.sort(key=lambda v:(v['kind']!='harm',TASKS.index(v['sample']['task']),v['question_id']))
    plans={};audit=[];tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    (OUT/'images').mkdir()
    for v in chosen:
        s=v['sample'];spec=v['spec'];qid=v['question_id'];m=s['input_mapping'];selected=spec['writers']['Semantic-k']
        assert sha256(s['image_path'])==s['image_sha256'];imagepath=OUT/'images'/(qid+Path(s['image_path']).suffix.lower());shutil.copyfile(s['image_path'],imagepath)
        v['gallery_image']=str(imagepath.relative_to(OUT));v['image_bytes_sha256']=sha256(imagepath)
        # Reconstruct and retain exact original character offsets, not guessed word positions.
        from transformers import AutoProcessor
        proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
        expanded=chat.replace('<image>','<image>'*576);enc=tok(expanded,return_offsets_mapping=True)
        assert enc['input_ids']==m['prompt_ids'] and digest(chat)==m['chat_prompt_sha256']
        start=expanded.index(s['question']);v['question_char_offset']=[start,start+len(s['question'])]
        v['tokens']=[dict(position=p,token_id=m['prompt_ids'][p],piece=tok.convert_ids_to_tokens(m['prompt_ids'][p]),decoded=tok.decode([m['prompt_ids'][p]]),offset=list(enc['offset_mapping'][p]),
            question=p in spec['question_positions'],selected=p in selected) for p in spec['relay_positions']]
        v['prompt_expanded']=expanded;v['semantic_core_dropped_groups']=[g for g in spec['groups'] if g['core'] and not set(g['positions'])&set(selected)]
        if v['kind']=='harm':
            singles=candidates(spec,qid);base=set(selected);qall=sorted(base|set(spec['question_positions']));tall=sorted(base|{p for g in spec['groups'] if not g['question'] for p in g['positions']})
            plans[qid]=dict(singles=singles,grouped=[dict(id='all_omitted_question',writers=qall,added=sorted(set(qall)-base)),dict(id='all_omitted_template',writers=tall,added=sorted(set(tall)-base))])
        for gi,g in enumerate(spec['groups']):
            audit.append(dict(question_id=qid,kind=v['kind'],task=s['task'],group_index=gi,decoded=g['decoded'],words='|'.join(g['words']),positions=json.dumps(g['positions']),subword_count=len(g['positions']),
                selected=set(g['positions'])<=set(selected),question=g['question'],core=g['core'],tags='|'.join(tags(g))))
    atomic_json(OUT/'case_manifest.json',dict(base_commit=BASE,selection_rules_sha256=sha256(OUT/'selection_rules.json'),cases=chosen))
    atomic_json(OUT/'candidate_plan.json',dict(base_commit=BASE,cases=plans,conditional_replacements_only_if_single_success=True,no_posthoc_candidate_changes=True))
    csvwrite(OUT/'token_group_audit.csv',audit)
    csvwrite(OUT/'selected_cases.csv',[dict(question_id=v['question_id'],task=v['sample']['task'],kind=v['kind'],image_sha256=v['sample']['image_sha256'],confirmatory=v['confirmatory'],error_type=v['error_type'],question=v['sample']['question'],gold=v['sample']['answer'],
        **{c+'_answer':v['rows'][c]['text'] for c in wanted},k=len(v['spec']['writers']['Semantic-k']),R=len(v['spec']['relay_positions']),k_R=len(v['spec']['writers']['Semantic-k'])/len(v['spec']['relay_positions'])) for v in chosen])
    atomic_json(OUT/'selection_validation.json',dict(passed=True,pool_harm=184,pool_rescue=87,selected_harm=12,selected_rescue=4,confirmatory_harm=sum(v['kind']=='harm' and v['confirmatory'] for v in chosen),
        confirmatory_rescue=sum(v['kind']=='rescue' and v['confirmatory'] for v in chosen),unique_images=len(images),all_offsets_and_token_groups_exact=True,original_answers_reused=True,cuda_initialized=torch.cuda.is_initialized(),candidate_plan_sha256=bind(),
        selected_cases_sha256=sha256(OUT/'case_manifest.json')))
    assert not torch.cuda.is_initialized()
    print('LOCKED CASES',[(v['question_id'],v['sample']['task'],v['kind'],v['confirmatory']) for v in chosen],flush=True)
    print('CANDIDATES',[(q,[(x['group']['decoded'],x['added']) for x in p['singles']]) for q,p in plans.items()],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','select_cases']);args=a.parse_args();globals()[args.stage]()
