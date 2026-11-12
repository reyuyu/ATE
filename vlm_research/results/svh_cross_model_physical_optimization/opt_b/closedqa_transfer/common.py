import collections
import datetime
import json
from pathlib import Path
import subprocess

from ..accuracy_autoresearch.common import config, conditional_writers, source_hashes as inherited_sources
from ..physical_visual_handoff.common import atomic, digest, read, sha, write_csv, TOLERANCES
from ..benchmarks.protocols import prompt_text, protocols, MODEL_PATH
from ..benchmarks.scoring import generated_score, pope_answer
from ..benchmarks.dataset_registry import digest as phase_digest
from ..fastp2_bypass_writer.common import hardware

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/ate_closedqa_frozen_transfer'
PHASE='7ccb44941cb48e901e6e23e96bae6da173fe2ce5'
BASE='60d262f83a981e49a614ef8213e5311c3c7c6469'
TASKS=('scienceqa_img','pope_random','pope_popular','pope_adversarial')
SEED=20261105
CONFIGS={
    'Native Full':config('native',head=1),
    'Fast-P2':config(head=1),
    'C3-V2':config(answer='both',head=1),
    'Full-QSA':config('qsa','decode',head=1),
    'Historical C4-V2':config('conditional15','both',head=1)}
DEVICES=[dict(index=2,uuid='GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6'),
         dict(index=3,uuid='GPU-96bc853f-ceea-2404-c704-c48a3d1ca185')]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def git_bytes(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths=[p for p in paths if p.name in ('__init__.py','common.py','engine.py','run.py','tests.py')]
    paths += [ROOT/'vlm_research/benchmarks'/p for p in ('protocols.py','scoring.py','dataset_registry.py')]
    return dict(inherited_sources(),**{str(p.relative_to(ROOT)):sha(p) for p in paths})


def offsets_for(encoded,start,end,text):
    positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>start and a<end]
    assert positions and all(encoded['input_ids'][i]!=32000 for i in positions)
    for i in positions:
        a,b=encoded['offset_mapping'][i]
        assert not text[a:min(b,start)].strip() and not text[max(a,end):b].strip(),'Substantive token crosses component boundary'
    return positions


def mapping(processor,s):
    payload=prompt_text(s)
    chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=payload)])],add_generation_prompt=True)
    expanded=chat.replace('<image>','<image>'*576)
    e=processor.tokenizer(expanded,return_offsets_mapping=True)
    payload_start=expanded.index(payload)
    context=('Context: '+s['metadata']['raw_row']['hint']+'\n') if s['dataset']=='scienceqa_img' and s['metadata']['raw_row'].get('hint') else ''
    q_start=payload_start+len(context); q_end=q_start+len(s['question'])
    assert expanded[q_start:q_end]==s['question']
    q=offsets_for(e,q_start,q_end,expanded)
    assistant_start=expanded.rindex('ASSISTANT')
    assistant=offsets_for(e,assistant_start,assistant_start+9,expanded)
    assert [e['input_ids'][i] for i in assistant]==[319,1799,9047,13566]
    image=[i for i,v in enumerate(e['input_ids']) if v==32000]
    r=list(range(image[-1]+1,len(e['input_ids'])-1))
    parts={'Q':q,'ASSISTANT':assistant,'Hint':[],'Options':[]}
    if context:
        a=payload_start+len('Context: '); b=payload_start+len(context)-1
        parts['Hint']=offsets_for(e,a,b,expanded)
    if s['dataset']=='scienceqa_img':
        choices='\n'.join(f'{letter}. {text}' for letter,text in zip(s['candidates'],s['metadata']['raw_row']['choices']))
        a=q_end+1
        assert expanded[a:a+len(choices)]==choices
        parts['Options']=offsets_for(e,a,a+len(choices),expanded)
    assert set(q).isdisjoint(parts['Hint']+parts['Options']+assistant)
    assert len(image)==576 and set(q+assistant)<=set(r) and len(e['input_ids'])-1 not in q+assistant
    previous=s['metadata']['input_mapping']
    assert payload==previous['text_payload'] and chat==previous['rendered_prompt']
    assert e['input_ids']==previous['prompt_ids']
    return dict(prompt_ids=e['input_ids'],prompt_text=payload,rendered_prompt=chat,
                image=image,R=r,Pminus1=len(e['input_ids'])-1,parts=parts,
                Q_char_span=[q_start,q_end],offsets=e['offset_mapping'])


def freeze():
    from PIL import Image
    from transformers import AutoProcessor
    assert not (OUT/'protocol.json').exists(),'Already frozen; do not overwrite'
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
    samples={}; provenance=[]; seen_images={}; original_manifests={}
    lock=read('/data/lk/ATE-resources/benchmarks/phase4_0/selection_lock.json')
    for task in TASKS:
        path='vlm_research/results/phase4_0_dataset_suite/manifests/'+task+'.json'
        raw=git_bytes(PHASE,path); m=json.loads(raw); original_manifests[task]=m
        assert len(m['samples'])==(500 if task=='scienceqa_img' else 300)
        assert set(x['question_id'] for x in m['samples'])==set(lock['question_ids'][task])
        for src in m['sources']:
            assert sha(src['path'])==src['sha256']; provenance.append(dict(task=task,**src))
        for s in m['samples']:
            assert phase_digest(s['metadata']['raw_row'])==s['source_row_sha256']
            h=seen_images.get(s['image_path'])
            if h is None:
                h=sha(s['image_path']); seen_images[s['image_path']]=h
                with Image.open(s['image_path']) as im: im.verify()
            assert h==s['image_sha256']
            if s['dataset']=='scienceqa_img':
                assert s['candidates']==list('ABCDE'[:len(s['metadata']['raw_row']['choices'])])
                assert s['answer']==s['candidates'][s['metadata']['gold_answer_index']]
            spec=mapping(processor,s); key=task+'__'+s['question_id']
            samples[key]=dict(task=task,dataset=s['dataset'],question_id=s['question_id'],image_id=s['image_id'],
                image_path=s['image_path'],image_sha256=h,question=s['question'],answer=s['answer'],
                candidates=s['candidates'],subject=s['metadata'].get('subject'),input_mapping=spec,
                prompt_text=spec['prompt_text'],source_row_sha256=s['source_row_sha256'],
                gold_index=s['metadata'].get('gold_answer_index'),choice_count=len(s['candidates']),
                phase_manifest_sha256=__import__('hashlib').sha256(raw).hexdigest(),
                allowed_prompt_fields=['question','hint','choices','fixed_instruction'] if s['dataset']=='scienceqa_img' else ['question','fixed_instruction'])
    # Verify the selected raw rows against the actual official annotation files.
    science=read(original_manifests['scienceqa_img']['sources'][0]['path'])
    for s in original_manifests['scienceqa_img']['samples']:
        assert science[s['question_id']]==s['metadata']['raw_row']
    for task in TASKS[1:]:
        src=Path(original_manifests[task]['sources'][0]['path'])
        official={str(r['question_id']):r for r in map(json.loads,src.read_text().splitlines()) if r}
        selected=original_manifests[task]['samples']; images=collections.defaultdict(list)
        for s in selected:
            assert official[s['question_id']]==s['metadata']['raw_row']; images[s['image_sha256']].append(s)
        assert all(len(g)==6 for g in images.values()) and len(images)==50
    groups=collections.defaultdict(list)
    for k,s in samples.items(): groups[s['image_sha256']].append(k)
    canonical={}; aliases={}
    for k,s in sorted(samples.items()):
        m=s['input_mapping']
        identity=digest([s['image_sha256'],m['prompt_ids'],m['parts'],
                         conditional_writers(s['question'],m['parts']['Q'],m['parts']['ASSISTANT'])[0]])
        aliases[k]=canonical.setdefault(identity,k)
    loads=[0,0]; shard={}
    for h,keys in sorted(groups.items(),key=lambda x:(-len(x[1]),digest([SEED,x[0]]))):
        rank=min(range(2),key=lambda r:(loads[r],r)); shard[h]=rank; loads[rank]+=len(keys)
    smoke={}; performance={}
    for task in TASKS:
        keys=[k for k,s in samples.items() if s['task']==task]
        ordered=sorted(keys,key=lambda k:digest([SEED,samples[k]['image_sha256'],samples[k]['input_mapping']['prompt_ids']]))
        smoke[task]=ordered[:3]
        chosen=[]; images=set()
        for k in ordered:
            if samples[k]['image_sha256'] not in images:
                chosen.append(k); images.add(samples[k]['image_sha256'])
            if len(chosen)==20: break
        assert len(chosen)==20; performance[task]=chosen
    weights=read('/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h/vlm_research/results/accuracy_autoresearch_6h/execution_freeze.json')['weights']
    assert all(sha(p)==h for p,h in weights.items())
    atomic(OUT/'sample_manifest.json',samples)
    atomic(OUT/'partition.json',dict(image_shard=shard,loads=loads,smoke=smoke,performance=performance,
        canonical_input=aliases,unique_actual_inputs=len(canonical)))
    atomic(OUT/'data_audit.json',dict(status='PASS',at=now(),sources=provenance,questions=len(samples),
        unique_image_paths=len(seen_images),image_SHA_clusters=len(groups),same_frozen_selection=True,
        no_lecture_solution_gold_in_prompt=True,all_phase_prompt_ids_exact=True,POPE_complete_six_question_groups=True))
    atomic(OUT/'protocol.json',dict(at=now(),base=BASE,phase4=PHASE,configs=CONFIGS,model=protocols()['common'],
        logits_to_keep=1,max_new_tokens=protocols()['common']['generation']['max_new_tokens'],
        no_search_training_or_adaptation=True,Q_definition='Only actual original question stem; excludes Hint/options/instruction',
        primary='Full-QSA minus Native: Accuracy for four datasets; question weighted paired Image-SHA cluster bootstrap',
        noninferiority=dict(family=list(TASKS),size=4,margin_pp=-1,alpha=.05,method='Bonferroni4 one-sided lower quantile .05/4; lower > -1'),
        bootstrap_replicates=20000,seed=SEED,other_comparisons='exploratory paired intervals vs Native and C3-V2; no NI claims',
        performance=dict(images_each_task=20,warmup=3,repeats=5,interleaved=True,cold_KV=True,vision_recomputed=True,
                         excludes='Processor, I/O and layout localization; includes model/cache setup, Vision, Prefill and Decode'),
        stop_after_frozen_matrix=True))
    atomic(OUT/'execution_freeze.json',dict(at=now(),sources=source_hashes(),weights=weights,devices=DEVICES,
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),partition_sha256=sha(OUT/'partition.json'),
        protocol_sha256=sha(OUT/'protocol.json'),tolerances=TOLERANCES))
    print('Frozen',len(samples),'questions;',len(groups),'image clusters; shards',loads,flush=True)


def scored(sample,text):
    proxy=dict(dataset=sample['dataset'],answer=sample['answer'],candidates=sample['candidates'],
               metadata=dict(gold_answer_index=sample['gold_index']))
    r=generated_score(proxy,text)
    return dict(score=float(r['official_generation_correct']),parsed=pope_answer(text) if sample['dataset']=='pope' else r['parsed_index'],
                parse_failed=sample['dataset']=='scienceqa_img' and r['parsed_index']==-1)


def checkpoint(key,label):
    return OUT/'checkpoints'/key/(label.replace(' ','_')+'.json')


def record(key,label,sample):
    p=checkpoint(key,label)
    if not p.exists(): return None
    b=read(p); r=b['payload']
    assert b['sha256']==digest(r) and r['sample_sha256']==digest(sample)
    assert r['config']==CONFIGS[label] and r['sources']==source_hashes()
    assert scored(sample,r['generation']['text'])==r['scoring']
    return r
