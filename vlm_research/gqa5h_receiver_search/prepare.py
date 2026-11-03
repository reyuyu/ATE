"""Blind data custodian: split only by input metadata, seal all Gold separately."""
import collections,io,shutil,unittest
from .common import *
from semantic_visual_writers.groups import token_groups
from transformers import AutoProcessor
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus
from relay_source_causality.common import hardware

def main():
    guard(reserve=1800);started=time.monotonic();PRIVATE.mkdir(parents=True,exist_ok=True)
    original=read(HIST/'sample_manifest.json');pool=original['samples'];assert len(pool)==3000 and all(sum(s['task']==t for s in pool)==500 for t in TASKS)
    # No answer or output value participates in this metadata table or split.
    metadata={s['question_id']:{k:s[k] for k in ['question_id','task','image_sha256']} for s in pool}
    cohorts=read(BASE/'results/gqa3000_space_nl_visual_writers/cohorts.json');dev=[r['question_id'] for r in csv.DictReader((BASE/'results/gqa360_structural_receiver_selection/sample_manifest.csv').open())]
    devimages={metadata[q]['image_sha256'] for q in dev};exclude=set(cohorts['old240_image_sha256'])
    exclusion_ids=set(cohorts['pilot120'])|set(cohorts['historical12'])|set(SMOKE_IDS)
    for name in ['failure_probe144/sample_plan.json','quick24/sample_plan.json']:
        exclusion_ids.update(read(BASE/'results/gqa3000_space_nl_visual_writers'/name)['cases'])
    exclusion_ids.update(c['question_id'] for c in read(BASE/'results/semantic_writer_error_cases/case_manifest.json')['cases'])
    exclude.update(metadata[q]['image_sha256'] for q in exclusion_ids if q in metadata)
    used=set(devimages);ids={'DEV':dev};refill=[]
    for split,n in [('FINAL',100),('TUNE',40)]:
        chosen=[]
        for t in TASKS:
            candidates=sorted((r for r in metadata.values() if r['task']==t),key=lambda r:digest([20261104,'gqa5h-v1',split,t,r['question_id'],r['image_sha256']]))
            selected=[]
            for allow_exposed in [False,True]:
                for r in candidates:
                    q=r['question_id'];im=r['image_sha256']
                    if im in used or (im in exclude and not allow_exposed):continue
                    selected.append(q);used.add(im)
                    if im in exclude:refill.append(dict(split=split,question_id=q,image_sha256=im,reason='Deterministic refill after clean input-only pool exhausted'))
                    if len(selected)==n:break
                if len(selected)==n:break
            assert len(selected)==n,(split,t,len(selected));chosen+=selected
        ids[split]=chosen
    assert len(ids['DEV'])==360 and len(ids['TUNE'])==240 and len(ids['FINAL'])==600
    images={k:{metadata[q]['image_sha256'] for q in vv} for k,vv in ids.items()}
    assert all(not images[a]&images[b] for a,b in [('DEV','TUNE'),('DEV','FINAL'),('TUNE','FINAL')])
    loo=[];dev_stages={}
    for n in [60,120,180,360]:
        dev_stages[str(n)]=[q for t in TASKS for q in sorted((q for q in dev if metadata[q]['task']==t),key=lambda q:digest(['dev-stage-v1',q,metadata[q]['image_sha256']]))[:n//6]]
    for t in TASKS:loo += sorted((q for q in dev if metadata[q]['task']==t),key=lambda q:digest(['LOO-96-v1',q,metadata[q]['image_sha256']]))[:16]
    split=dict(ids=ids,metadata={q:metadata[q] for values in ids.values() for q in values},unique_image_counts={k:len(v) for k,v in images.items()},task_counts={k:collections.Counter(metadata[q]['task'] for q in v) for k,v in ids.items()},dev_stages=dev_stages,loo96=loo,exclusion_images=sorted(exclude),refill=refill,seed=20261104,holdout_sealed=True,holdout_is_internal_historical_pool=True,selection_uses_only=['task','question_id','image_sha256'],Gold_access_policy='Custodian serializes blind Gold files; analysis/worker callers cannot read FINAL Gold or historical outputs until winner_freeze exists.')
    atomic_json(P/'split_manifest.json',split)
    writecsv(P/'split_membership.csv',[dict(split=k,**metadata[q]) for k,values in ids.items() for q in values])
    # The custodian writes Gold to a separate resource directory; no Gold is
    # copied into public input specs, input manifests or selector arguments.
    wanted={q for v in ids.values() for q in v}|set(SMOKE_IDS);samples={s['question_id']:s for s in pool if s['question_id'] in wanted}
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);pos=read(BASE/'results/noun_article_visual_writers_pilot/pos_parse_all3000.json');inputhash=read(HIST/'input_hashes.json');history=History();spec_index=[]
    for q,s in samples.items():
        original_hash=digest(s);clean={k:v for k,v in s.items() if k!='answer'};clean['input_mapping']={k:v for k,v in s['input_mapping'].items() if k!='gold_continuation_ids'}
        atomic_json(PRIVATE/'samples'/f'{q}.json',clean);atomic_json(PRIVATE/'gold'/f'{q}.json',dict(answer=s['answer']))
        file=HIST/'token_manifests'/(s['task']+'__'+q+'.json');history.committed(file);old=read(file)
        chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True);assert digest(chat)==s['input_mapping']['chat_prompt_sha256']
        groups,Q,words=token_groups(processor.tokenizer,s['question'],chat,s['input_mapping']['prompt_ids'],s['input_mapping']['visual_positions'],old['relay_positions']);assert groups==old['groups'] and Q==old['question_positions'] and words==old['words']
        start=chat.replace('<image>','<image>'*576).index(s['question'])
        for g in groups:
            matches=[t for t in pos[q]['tokens'] if any(w['in_question'] and set(w['positions'])&set(g['positions']) and t['start']<w['end']-start and t['end']>w['start']-start for w in words)]
            g['pos']=sorted({t['pos'] for t in matches});g['dep']=sorted({t['dep'] for t in matches})
        info=dict(question=s['question'],task=s['task'],Q=Q,R=old['relay_positions'],groups=groups,words=words,k80=old['budgets']['80']['actual_k'],historical_writers=old['writers'],historical_aliases=old['aliases'],input_hashes=inputhash[s['task']+'__'+q],original_sample_sha256=original_hash,prompt_ids_sha256=old['prompt_ids_sha256'],historical_token_manifest_sha256=sha(file),input_sample_sha256=digest(clean))
        atomic_json(P/'input_specs'/f'{q}.json',info);spec_index.append(dict(question_id=q,path=f'input_specs/{q}.json',sha256=sha(P/'input_specs'/f'{q}.json'),input_sample_sha256=digest(clean),image_sha256=s['image_sha256'],original_sample_sha256=original_hash))
    history.close();writecsv(P/'input_spec_index.csv',spec_index)
    hp=read(HIST/'protocol.json');assert hp['model_revision']==REVISION and hp['transformers']=='4.49.0'
    for n,h in {**hp['transformers_sources'],**hp['scorer_sources']}.items():assert sha(n)==h
    for n,info in hp['model_files'].items():assert sha(n)==info['sha256']
    state=hardware();gpus=idle_gpus(state)[:2];disk={n:shutil.disk_usage(n)._asdict() for n in [str(P),str(PRIVATE),'/data']};assert shutil.disk_usage(P).free>2*1024**3
    shared={str(BASE/n):h for n,h in read(BASE/'results/gqa360_structural_receiver_selection/protocol.json')['sources'].items()} if False else {f:h for f,h in source_hashes().items() if str(BASE) in f}
    atomic_json(P/'protocol.json',dict(status='FROZEN',model_revision=REVISION,precision='float16',attention='eager',transformers='4.49.0',max_new_tokens=32,logits_to_keep=1,scope='Only receiver pre-softmax Q-to-original-I hard masks; all32 P-1/decode Iblocked; dense all other computation and independent true KV.',original_manifest_sha256=sha(HIST/'sample_manifest.json'),split_sha256=sha(P/'split_manifest.json'),input_spec_index_sha256=sha(P/'input_spec_index.csv'),gpus=gpus,model_files=hp['model_files'],transformers_sources=hp['transformers_sources'],scorer_sources=hp['scorer_sources'],shared_sources=shared,initial_source_hashes=source_hashes(),bootstrap_seed=20261104,bootstrap_repetitions=20000,deadline=read(P/'clock.json'),budget_update=read(P/'budget_authorization_update.json'),custodian_no_holdout_outcomes_exposed=True))
    atomic_json(P/'resource_preflight.json',dict(**state,selected=gpus,disk=disk,utc=utc(),elapsed_seconds=time.monotonic()-read(P/'clock.json')['start_monotonic']))
    from .tests import SearchTests
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SearchTests));(P/'cpu_tests.log').write_text(stream.getvalue());assert result.wasSuccessful();assert not torch.cuda.is_initialized()
    atomic_json(P/'cpu_validation.json',dict(passed=True,tests=result.testsRun,holdout_sealed=True,source_hashes=source_hashes(),seconds=time.monotonic()-started));event('PREPARE_COMPLETE',seconds=time.monotonic()-started,cases=len(wanted),gpus=gpus,refill_count=len(refill));print(json.dumps(dict(frozen=True,sizes={k:len(v) for k,v in ids.items()},refill_count=len(refill),gpus=gpus,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
