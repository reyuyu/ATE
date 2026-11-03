"""Freeze all3000 receiver sets using native offsets and historical text-only POS."""
import collections,datetime,shutil
import torch
from transformers import AutoProcessor
from paper_rq1.scorers import gqa_prediction
from .common import *

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'token_plan.json').exists()
    h=GitHistory();manifest=h.load(HIST/'sample_manifest.json',SEM);samples=manifest['samples'];index=h.load(HIST/'selection_index.json',SEM)
    hashes=h.load(HIST/'input_hashes.json',SEM);split=h.load(HIST/'split_integrity.json',SEM);shards=h.load(HIST/'shards.json',SEM)['image_assignments']
    pars=h.load(PILOT/'pos_parse_all3000.json',BASE);env=h.load(PILOT/'parser_environment.json',BASE);oldplan=h.load(PILOT/'token_plan.json',BASE)
    assert env['spacy']=='3.8.7' and env['model']=='3.8.0' and len(samples)==len(pars)==3000
    h.load(PILOT/'selection_rules.json',BASE);assert h.load(HIST/'completion.json',SEM)['status']=='COMPLETE'
    selected=h.load(ROOT/'results/semantic_writer_error_cases/case_manifest.json','425286a2566cc6b0387460492a0f99566b142a38')['cases']
    harm=[v['question_id'] for v in selected if v['kind']=='harm'];assert len(harm)==12
    # Read the committed observation case manifest only as a provenance cross-check.
    attrepo=Path('/data/lk/ATE-worktrees/semantic-omitted-token-attention-mass/vlm_research/results/semantic_omitted_token_attention_mass/case_manifest.json')
    attblob=subprocess.check_output(['git','show',ATTENTION+':vlm_research/results/semantic_omitted_token_attention_mass/case_manifest.json'],cwd=ROOT.parent)
    assert attrepo.read_bytes()==attblob;att={v['question_id']:v for v in json.loads(attblob)['cases']}
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);tok=proc.tokenizer
    oldimages=set(split['exploration_image_sha256']);oldids=set(split['exploration_question_ids']);confirm=[];plans={};boundary_audit=[];pos_audit=[];budgets=[];baselines=[];reuses=[];manifest_rows=[];controls=[];new_needed=0;closures=[]
    for i,s in enumerate(samples):
        q=s['question_id'];sk=s['task']+'__'+q;spec=h.load(HIST/index[sk]['path'],SEM);m=s['input_mapping'];I=m['visual_positions'];R=spec['relay_positions'];Q=spec['question_positions'];S=spec['writers']['Semantic-k']
        assert len(I)==576 and I==list(range(I[0],I[-1]+1)) and R==list(range(I[-1]+1,m['p_minus_one'])) and m['p_minus_one']==len(m['prompt_ids'])-1
        assert sha256(Path(s['image_path']))==s['image_sha256']
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
        expanded=chat.replace('<image>','<image>'*576);enc=tok(expanded,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];assert ids==m['prompt_ids'] and digest(chat)==m['chat_prompt_sha256']
        assert expanded.count(s['question'])==1;start=expanded.index(s['question']);assert flatten(spec['groups'])==R
        B=[I[-1]+1,I[-1]+2];groups=spec['groups'];bypos={p:g for g in groups for p in g['positions']}
        assert R[:2]==B and Q[0]==B[-1]+1 and all(p in R and p not in I and p!=m['p_minus_one'] and p not in Q for p in B)
        assert [ids[p] for p in B]==[29871,13] and tok.convert_ids_to_tokens([ids[p] for p in B])==['\u2581','<0x0A>']
        assert [tok.decode([ids[p]],skip_special_tokens=False,clean_up_tokenization_spaces=False) for p in B]==['','\n']
        assert tok.decode([ids[p] for p in B],skip_special_tokens=False,clean_up_tokenization_spaces=False)=='\n'
        assert all(bypos[p]['positions']==[p] and not bypos[p]['question'] for p in B)
        image_end=expanded.rfind('<image>')+len('<image>');assert expanded[image_end:start]=='\n'
        assert all(offsets[p]==(image_end,start) for p in B),'Unexpected actual SPACE/NL offsets; stop'
        for label,p in zip(['SPACE','NL'],B):
            boundary_audit.append(dict(question_id=q,task=s['task'],label=label,position=p,token_id=ids[p],piece=tok.convert_ids_to_tokens(ids[p]),decoded=json.dumps(tok.decode([ids[p]],skip_special_tokens=False,clean_up_tokenization_spaces=False)),offset=json.dumps(offsets[p]),raw_span=json.dumps(expanded[offsets[p][0]:offsets[p][1]]),question_start=start,in_R=p in R,in_Question=p in Q,in_Image=p in I,is_P_minus_one=p==m['p_minus_one'],inserted_text=False))
        parse=pars[q];assert parse['question']==s['question'];sets={'N':set(),'D':set()}
        for t in parse['tokens']:
            raw={p for p in Q if offsets[p][1]>start+t['start'] and offsets[p][0]<start+t['end']};mapped={p for g in groups if raw&set(g['positions']) for p in g['positions']};assert mapped<=set(Q)
            for k,flag in [('N',t['noun']),('D',t['article'])]:
                if flag:sets[k].update(mapped)
            if raw!=mapped:closures.append(dict(question_id=q,text=t['text'],raw=sorted(raw),whole_group=sorted(mapped)))
            pos_audit.append(dict(question_id=q,task=s['task'],**t,positions=json.dumps(sorted(mapped)),subwords=len(mapped)))
        N=sets['N'];D=sets['D'];ND=sorted(N|D);instruction=s['prompt_text'][len(s['question']):].strip();istart=expanded.index(instruction,start+len(s['question']));T=control_two(groups,Q,B,(istart,istart+len(instruction)),offsets)
        assert not set(S)&set(B) and set(S)<=set(Q) and set(ND)<=set(Q)
        if T is not None:assert len(T)==2 and not set(T)&(set(B)|set(Q)|set(S)|set(ND)) and intact(T,groups)
        writers={'All-Writer':R,'Semantic-k':S,'S+SPACE':sorted(set(S)|{B[0]}),'S+NL':sorted(set(S)|{B[1]}),'S+SPACE+NL':sorted(set(S)|set(B)),'Noun+Article':ND,'Noun+Article+SPACE+NL':sorted(set(ND)|set(B)),'S+Template-Control-2':sorted(set(S)|set(T)) if T is not None else None,'N+D+Template-Control-2':sorted(set(ND)|set(T)) if T is not None else None}
        if q in oldplan['cases']:
            old=oldplan['cases'][q];assert old['S']==S and old['N']==sorted(N) and old['D']==sorted(D) and old['input_hashes']==hashes[sk]
        if q in att:assert att[q]['spec']['writers']['Semantic-k']==S and att[q]['input_hashes']==hashes[sk]
        history={};available={}
        for label in ['All-Writer','Semantic-k']:
            path=HIST/f"shard{shards[s['image_sha256']]}/checkpoints"/sk/(spec['aliases'][label]+'.json');cp=h.load(path,SEM);g=cp['generation']
            assert cp['writers']==writers[label] and cp['input_hashes']==hashes[sk] and cp['image_sha256']==s['image_sha256'] and cp['sample_sha256']==digest(s)
            assert cp['selection_binding']==spec['selection_binding'] and cp['result_sha256']==digest(dict(generation=g,teacher_forced=cp['teacher_forced']))
            assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer']) and g['audit']['passed'] and g['writer_positions']==writers[label]
            assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['token_ids'] and g['length']==len(g['token_ids'])
            ref=dict(path=str(path.relative_to(ROOT)),commit=SEM,sha256=sha256(path),condition=label);history[label]=ref;available[tuple(writers[label])]=ref
            baselines.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],gold=s['answer'],condition=label,answer=g['text'],prediction=g['prediction'],correct=g['correct'],token_ids=json.dumps(g['token_ids']),first_token=g['token_ids'][0],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],k=len(writers[label]),R=len(R),writer_ratio=len(writers[label])/len(R),source_path=ref['path'],source_commit=SEM,source_sha256=ref['sha256'],status='AUDITED_HISTORICAL_GENERATION'))
        if q in oldplan['cases']:
            # N+D pilot reuse must bind the exact old token plan and inputs, not just its label.
            oldpath=PILOT/'records'/q/'Noun+Article.json';cp=h.load(oldpath,BASE);assert cp['writers']==ND
            g=cp['generation'];assert g['writer_positions']==ND and g['correct']==int(gqa_prediction(g['text'])==s['answer']) and g['audit']['passed']
            if not cp['reused']:assert cp['input_hashes']==hashes[sk] and cp['plan_sha256']==sha256(PILOT/'token_plan.json')
            available[tuple(ND)]=dict(path=str(oldpath.relative_to(ROOT)),commit=BASE,sha256=sha256(oldpath),condition='Noun+Article')
        unique={};aliases={}
        for label,w in writers.items():
            if w is None:aliases[label]=None;continue
            assert w==sorted(set(w)) and intact(w,groups) and set(w)<=set(R)
            uid=digest(w);aliases[label]=uid
            if uid not in unique:unique[uid]=dict(positions=w,labels=[],reuse=available.get(tuple(w)))
            assert unique[uid]['positions']==w;unique[uid]['labels'].append(label)
            budgets.append(dict(question_id=q,task=s['task'],condition=label,k=len(w),R=len(R),ratio=len(w)/len(R),added_to_S=len(set(w)-set(S)),added_to_ND=len(set(w)-set(ND)),canonical_id=uid,status='HISTORICAL_REUSE_AUDITED' if unique[uid]['reuse'] else 'PENDING_GPU_GENERATION'))
        new_needed+=sum(u['reuse'] is None for u in unique.values())
        for uid,u in unique.items():
            if u['reuse']:reuses.append(dict(question_id=q,canonical_id=uid,labels='|'.join(u['labels']),**u['reuse']))
        is_confirm=q not in oldids and s['image_sha256'] not in oldimages
        if is_confirm:confirm.append(sk)
        plans[q]=dict(task=s['task'],sample_key=sk,image_sha256=s['image_sha256'],S=S,N=sorted(N),D=sorted(D),ND=ND,B=B,template_control=T,R=R,Q=Q,input_hashes=hashes[sk],prompt_ids_sha256=digest(ids),prompt_chat_sha256=digest(chat),question_char_start=start,groups=[dict(**g,offsets=[list(offsets[p]) for p in g['positions']]) for g in groups],writers=writers,aliases=aliases,unique=unique,history=history,confirmatory2640=is_confirm,pilot120=q in oldplan['pilot_ids'],historical12=q in harm)
        controls.append(dict(question_id=q,feasible=T is not None,positions=json.dumps(T),decoded=json.dumps([bypos[p]['decoded'] for p in T]) if T else '',added_to_S=2 if T else '',added_to_ND=2 if T else '',same_budget_as_B=T is not None))
        manifest_rows.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],confirmatory2640=is_confirm,pilot120=q in oldplan['pilot_ids'],historical12=q in harm))
        if (i+1)%500==0:print('CPU AUDITED',i+1,'pending canonical',new_needed,flush=True)
    assert confirm==split['confirmatory_keys'] and len(confirm)==2640
    assert sum(v['pilot120'] for v in plans.values())==120 and sum(v['historical12'] for v in plans.values())==12
    image_ids=sorted({s['image_sha256'] for s in samples});assign={im:i%2 for i,im in enumerate(image_ids)}
    assert len({assign[v['image_sha256']] for v in plans.values()})==2
    h.close();save('source_audit.json',dict(files=h.files,attention_manifest_sha256=hashlib.sha256(attblob).hexdigest(),attention_commit=ATTENTION))
    shutil.copyfile(HIST/'sample_manifest.json',OUT/'sample_manifest.json');shutil.copyfile(PILOT/'parser_environment.json',OUT/'parser_environment.json')
    csvwrite('boundary_token_audit.csv',boundary_audit);csvwrite('pos_token_mapping.csv',pos_audit);csvwrite('token_budgets.csv',budgets);csvwrite('audited_historical_baselines.csv',baselines);csvwrite('historical_reuse_index.csv',reuses);csvwrite('template_control_audit.csv',controls);csvwrite('cohort_membership.csv',manifest_rows)
    save('token_plan.json',dict(cases=plans,conditions=CONDITIONS,planned_new_canonical_generations=new_needed,logical_new_labels=7*3000))
    save('cohorts.json',dict(main_ids=[s['question_id'] for s in samples],image_disjoint2640=confirm,pilot120=oldplan['pilot_ids'],historical12=harm,old240_image_sha256=sorted(oldimages),image_disjoint_not_external_validation=True))
    save('prospective_shards.json',dict(mode='prospective; no GPU selected; all conditions for one image stay together',one_worker={im:0 for im in image_ids},two_workers=assign,two_worker_questions=dict(collections.Counter(assign[s['image_sha256']] for s in samples)),two_worker_task_counts={str(rank):dict(collections.Counter(s['task'] for s in samples if assign[s['image_sha256']]==rank)) for rank in [0,1]}))
    assert not torch.cuda.is_initialized()
    save('cpu_input_validation.json',dict(passed=True,questions=3000,unique_images=len(image_ids),task_counts=dict(collections.Counter(s['task'] for s in samples)),all3000_native_ids_offsets_and_prompts_exact=True,all6000_boundary_tokens_verified=True,SPACE=dict(id=29871,piece='\u2581',individual_decode='',role='native boundary, not literal inserted SPACE text'),NL=dict(id=13,piece='<0x0A>',individual_decode='\n'),shared_offset='Both native tokens may map to the same single newline character span; positions are distinct.',no_prompt_text_inserted=True,all_original_groups_preserved=True,all3000_N_D_from_original_fixed_parser=True,original_pilot132_mapping_exact=True,history_baselines_audited=6000,old240_image_disjoint_questions=len(confirm),template_control_unavailable=sum(v['template_control'] is None for v in plans.values()),planned_new_canonical_generations=new_needed,logical_new_condition_labels=21000,word_group_closures=closures,cuda_initialized=False))
    print('CPU INPUT FREEZE COMPLETE',new_needed,'new canonical generations pending',flush=True)

if __name__=='__main__':main()
