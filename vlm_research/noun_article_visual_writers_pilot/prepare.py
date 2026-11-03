import collections,shutil
from transformers import AutoProcessor
from paper_rq1.scorers import gqa_prediction
from .common import *
def main():
    assert not (OUT/'token_plan.json').exists()
    parses=read(OUT/'pos_parse_all3000.json');env=read(OUT/'parser_environment.json');assert env['spacy']=='3.8.7' and env['model']=='3.8.0'
    samples=hist('sample_manifest.json')['samples'];index=hist('selection_index.json');hashes=hist('input_hashes.json');shards=hist('shards.json')['image_assignments'];assert hist('completion.json')['status']=='COMPLETE'
    harm=[v for v in read(ROOT/'results/semantic_writer_error_cases/case_manifest.json')['cases'] if v['kind']=='harm'];assert len(harm)==12
    oldimages=hist('split_integrity.json')['exploration_image_sha256']
    excluded=set(oldimages)|{v['sample']['image_sha256'] for v in harm}
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);tok=proc.tokenizer
    allplans={};audit=[];coverage=[];closures=[]
    for s in samples:
        qid=s['question_id'];sk=s['task']+'__'+qid;info=index[sk];spec=hist(info['path']);m=s['input_mapping']
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
        expanded=chat.replace('<image>','<image>'*576);enc=tok(expanded,return_offsets_mapping=True);assert enc['input_ids']==m['prompt_ids'] and digest(chat)==m['chat_prompt_sha256']
        start=expanded.index(s['question']);assert expanded.count(s['question'])==1
        parse=parses[qid];assert parse['question']==s['question'];groups=spec['groups'];sets={k:set() for k in ['N','D']}
        for t in parse['tokens']:
            raw={p for p in spec['question_positions'] if enc['offset_mapping'][p][1]>start+t['start'] and enc['offset_mapping'][p][0]<start+t['end']}
            mapped={p for g in groups if raw&set(g['positions']) for p in g['positions']};assert mapped<=set(spec['question_positions'])
            for k,flag in [('N',t['noun']),('D',t['article'])]:
                if flag:sets[k].update(mapped)
            if raw!=mapped:closures.append(dict(question_id=qid,text=t['text'],raw=sorted(raw),closed=sorted(mapped)))
            audit.append(dict(question_id=qid,task=s['task'],question=s['question'],**t,positions=json.dumps(sorted(mapped)),subwords=len(mapped)))
        S=set(spec['writers']['Semantic-k']);N=sets['N'];D=sets['D'];writers={'Semantic-k':sorted(S),'All-Writer':spec['writers']['All-Writer'],'Noun-only':sorted(N),'Noun+Article':sorted(N|D),'Semantic+Noun':sorted(S|N),'Semantic+Article':sorted(S|D),'Semantic+Noun+Article':sorted(S|N|D)};controls={}
        pool=[g for g in groups if g['question'] and g['words'] and not any(w.lower() in ['a','an','the'] for w in g['words'])]
        for label,base in [('S',S),('SN',S|N),('N',N)]:
            k=len(D-base)
            for seed in SEEDS:
                eligible=sorted([g for g in pool if not set(g['positions'])&base],key=lambda g:tie(seed,qid,label,g['positions']))
                chosen=subsets(eligible).get(k);name=f'{label}+Placebo-D-{seed}'
                writers[name]=sorted(base|set(flatten(chosen))) if chosen is not None else None
                controls[name]=dict(added_tokens=k,feasible=chosen is not None,added=flatten(chosen) if chosen is not None else None)
        if qid in {v['question_id'] for v in harm}:
            k=len(D-S);rem=subsets(sorted([g for g in groups if set(g['positions'])<=S],key=lambda g:tie('remove',qid,g['positions']))).get(k)
            writers['Article-Swap']=sorted((S-set(flatten(rem)))|D) if rem is not None else None
            for seed in SEEDS:
                add=controls[f'S+Placebo-D-{seed}']['added']
                writers[f'Random-Swap-{seed}']=sorted((S-set(flatten(rem)))|set(add)) if rem is not None and add is not None else None
        for name,w in writers.items():
            if w is not None:
                assert intact(w,groups) and w==sorted(set(w)) and set(w)<=set(spec['relay_positions'])
                if 'Swap' in name:assert len(w)==len(S)
        cov=dict(question_id=qid,task=s['task'],image_sha256=s['image_sha256'],N=len(N),D=len(D),S=len(S),R=len(spec['relay_positions']),N_selected=len(N&S),D_selected=len(D&S),eligible=bool(D) and s['image_sha256'] not in excluded)
        coverage.append(cov);allplans[qid]=dict(sample=s,spec=spec,N=sorted(N),D=sorted(D),S=sorted(S),writers=writers,controls=controls,parse=parse,input_hashes=hashes[sk],question_char_start=start)
    chosen=[];images=set()
    for task in TASKS:
        pool=sorted([c for c in coverage if c['task']==task and c['eligible']],key=lambda c:tie(task,c['question_id'],c['image_sha256']))
        for c in pool:
            if c['image_sha256'] in images:continue
            chosen.append(c['question_id']);images.add(c['image_sha256'])
            if sum(allplans[q]['sample']['task']==task for q in chosen)==20:break
        assert sum(allplans[q]['sample']['task']==task for q in chosen)==20,task
    qids=chosen+[v['question_id'] for v in harm];assert len(set(qids))==132
    selected={qid:allplans[qid] for qid in qids}
    for qid,v in selected.items():
        s=v['sample'];sk=s['task']+'__'+qid;spec=v['spec'];v['cohort']='pilot120' if qid in chosen else 'historical12';v['history']={}
        for label in ['Semantic-k','All-Writer']:
            path=HIST/f"shard{shards[s['image_sha256']]}/checkpoints"/sk/(spec['aliases'][label]+'.json');oldcp=hist(str(path.relative_to(HIST)))
            assert oldcp['writers']==v['writers'][label] and oldcp['input_hashes']==v['input_hashes'] and digest(dict(generation=oldcp['generation'],teacher_forced=oldcp['teacher_forced']))==oldcp['result_sha256']
            assert oldcp['generation']['correct']==int(gqa_prediction(oldcp['generation']['text'])==s['answer']) and sha256(s['image_path'])==s['image_sha256']
            v['history'][label]=dict(path=str(path.relative_to(HIST)),sha256=sha256(path),generation=oldcp['generation'],raw=oldcp['raw'])
        if v['cohort']=='historical12':
            image=OUT/'images'/(qid+Path(s['image_path']).suffix);image.parent.mkdir(exist_ok=True);shutil.copyfile(s['image_path'],image);v['gallery_image']=str(image.relative_to(OUT))
    csvwrite(OUT/'coverage_all3000.csv',coverage);csvwrite(OUT/'pos_token_audit.csv',audit)
    csvwrite(OUT/'sample120.csv',[c for c in coverage if c['question_id'] in chosen])
    csvwrite(OUT/'budget_matching.csv',[dict(question_id=qid,cohort=v['cohort'],condition=n,**c) for qid,v in selected.items() for n,c in v['controls'].items()])
    atomic_json(OUT/'token_plan.json',dict(cases=selected,pilot_ids=chosen,historical_ids=[v['question_id'] for v in harm]))
    atomic_json(OUT/'cpu_validation.json',dict(passed=True,parsed_questions=len(parses),native_offsets_verified=3000,whole_group_mappings_verified=3000,pilot120=len(chosen),historical12=12,pilot_unique_images=len(images),image_disjoint=not images&excluded,exclusion_images=len(excluded),per_task=dict(collections.Counter(allplans[q]['sample']['task'] for q in chosen)),closures=closures,plan_sha256=bind(),cuda_initialized=torch.cuda.is_initialized(),placebo_unavailable=sum(w is None for v in selected.values() for n,w in v['writers'].items() if 'Placebo' in n),historical_source_commit=BASE,parser_environment_sha256=sha256(OUT/'parser_environment.json')))
    assert not torch.cuda.is_initialized();print('CPU FREEZE',bind(),'132 cases',flush=True)
if __name__=='__main__':main()
