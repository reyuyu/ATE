"""Conservative historical exclusions and input-only independent cohort freeze."""
import argparse
import json
from pathlib import Path
import subprocess
from collections import Counter
from PIL import Image
from transformers import AutoProcessor
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.protocols import prompt_text
from paper_rq1_single_layer_gqa6.data import order as old_order
from .protocol import ROOT, OUT, SEED, TASKS, ANNOTATION, IMAGES, MODEL_PATH, PROTOCOL


def rows():
    command='import json,pyarrow.parquet as pq,sys;json.dump(pq.read_table(sys.argv[1]).to_pylist(),sys.stdout)'
    return json.loads(subprocess.check_output(['/home/likai/anaconda3/envs/AFAC/bin/python','-c',command,str(ANNOTATION)]))


def historical():
    ids, hashes, qids, sources, seen = set(), set(), set(), {}, set()
    # Inspect experiment manifests, not enormous per-condition checkpoint trees.
    roots=[p/'vlm_research/results' for p in Path('/data/lk/ATE-worktrees').iterdir() if p.is_dir()]
    roots.append(Path('/data/lk/深度学习/ATE/vlm_research/results'))
    def walk(value):
        if isinstance(value,dict):
            for key in ('image_id','imageId'):
                if key in value and isinstance(value[key],(str,int)):ids.add(str(value[key]))
            identity=value.get('image_identity','')
            if isinstance(identity,str) and identity.startswith('vg:'):ids.add(identity.split(':',1)[1])
            path=value.get('image_path','')
            if isinstance(path,str) and '/MMEB/GQA/image_' in path:ids.add(Path(path).stem.removeprefix('image_'))
            if isinstance(value.get('image_sha256'),str):hashes.add(value['image_sha256'])
            if 'question_id' in value:qids.add(str(value['question_id']))
            for item in value.values():
                if isinstance(item,(dict,list)):walk(item)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(dict,list)):walk(item)
    for root in sorted(roots):
        paths=set(root.glob('*/*manifest*.json'))|set(root.glob('*/manifests/*.json'))|set(root.glob('*/config.json'))
        for path in sorted(paths):
            if 'paper_rq1_early_joint_attention' in path.parts:continue
            h=sha256(path)
            if h in seen:continue
            seen.add(h)
            value=json.loads(path.read_text());before=len(ids);walk(value)
            sources[str(path)]=dict(sha256=h,additional_image_ids=len(ids)-before)
    return ids,hashes,qids,sources


def eligible(source,used,used_q):
    return {task:[r for r in source if r['types']['detailed'] in detail and str(r['imageId']) not in used
        and str(r['id']) not in used_q and r['isBalanced'] is True and r['answer']] for task,detail in TASKS.items()}


def census():
    source=rows();used,hashes,qids,sources=historical();pools=eligible(source,used,qids)
    return source,pools,used,hashes,qids,sources


def prepare():
    if (OUT/'execution_manifest.json').exists():raise ValueError('Already frozen for GPU')
    source,pools,used,used_hash,used_q,sources=census()
    OUT.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default'
    selected=[];smoke=[];image_hash={};rejected=[];counts={}
    for task,pool in pools.items():
        ordered=sorted(pool,key=lambda r:digest([SEED,'early-joint-v1',task,str(r['id'])]))
        accepted=[]
        for r in ordered:
            iid=str(r['imageId']);path=IMAGES/f'image_{iid}.jpg'
            if not path.exists():
                raise ValueError(f'Input-only candidate missing: {path}; no silent replacement')
            if iid not in image_hash:image_hash[iid]=sha256(path)
            if image_hash[iid] in used_hash:
                rejected.append(dict(task=task,question_id=r['id'],reason='historical_image_sha256'));continue
            accepted.append(r)
        chosen=accepted[:500]
        counts[task]=dict(official_questions=sum(r['types']['detailed'] in TASKS[task] for r in source),
            unseen_id_questions=len(pool),eligible_questions=len(accepted),eligible_images=len({r['imageId'] for r in accepted}),
            selected=len(chosen),shortfall=500-len(chosen))
        for r in chosen:
            iid=str(r['imageId']);path=IMAGES/f'image_{iid}.jpg'
            s=dict(dataset='gqa_balanced',task=task,question_id=str(r['id']),image_id=iid,image_identity='vg:'+iid,
                image_path=str(path),image_sha256=image_hash[iid],question=r['question'],answer=r['answer'],
                type_detailed=r['types']['detailed'],source_row_sha256=digest(r),source='Official GQA Balanced validation pinned parquet',
                prior_image_overlap=False,rank=int(digest([SEED,'worker',iid]),16)%3)
            selected.append(s)
        print(task,counts[task],flush=True)
    # Freeze separate deterministic smoke questions; never select by Full output.
    selected_ids={s['image_id'] for s in selected};selected_q={s['question_id'] for s in selected}
    spare=[r for task in TASKS for r in sorted(pools[task],key=lambda r:digest([SEED,'smoke',str(r['id'])]))
        if str(r['imageId']) not in selected_ids and str(r['id']) not in selected_q
        and image_hash.get(str(r['imageId'])) not in used_hash]
    for r in spare:
        iid=str(r['imageId'])
        if iid in {s['image_id'] for s in smoke}:continue
        task=next(t for t,d in TASKS.items() if r['types']['detailed'] in d)
        smoke.append(dict(dataset='gqa_balanced',task=task,question_id=str(r['id']),image_id=iid,image_identity='vg:'+iid,
            image_path=str(IMAGES/f'image_{iid}.jpg'),image_sha256=image_hash[iid],question=r['question'],answer=r['answer'],
            source_row_sha256=digest(r),source='Official GQA Balanced validation separate smoke'))
        if len(smoke)==2:break
    if len(smoke)!=2:raise ValueError('Need two disjoint smoke images')
    for i,s in enumerate(selected+smoke):
        with Image.open(s['image_path']) as original:original.load();image=original.convert('RGB');s['image_size']=list(image.size)
        s['prompt_text']=prompt_text(dict(dataset='gqa_balanced',question=s['question']))
        chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=s['prompt_text'])])],tokenize=False,add_generation_prompt=True)
        ids=processor(images=image,text=chat,return_tensors='pt')['input_ids'][0].tolist()
        if ids.count(32000)!=576:raise ValueError('Wrong visual input mapping')
        s['input_mapping']=dict(prompt_ids=ids,p_minus_one=len(ids)-1,chat_prompt_sha256=digest(chat))
        if (i+1)%500==0:print('Mapped',i+1,flush=True)
    if len({s['question_id'] for s in selected})!=len(selected):raise ValueError('Duplicate question')
    if {s['image_id'] for s in selected}&used or {s['image_sha256'] for s in selected}&used_hash:raise ValueError('Historical image leakage')
    from paper_rq1.scorers import GQA_EVALUATOR_SHA,GQA_CONVERTER_SHA
    official=ANNOTATION.parent/'gqa_eval.py'
    converter=Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/llava/scripts/convert_gqa_for_eval.py')
    if sha256(official)!=GQA_EVALUATOR_SHA or sha256(converter)!=GQA_CONVERTER_SHA:raise ValueError('Official scorer changed')
    freeze_json(OUT/'experiment_config.json',PROTOCOL)
    freeze_json(OUT/'sample_manifest.json',dict(samples=selected,smoke=smoke,counts=counts,seed=SEED,
        annotation_sha256=sha256(ANNOTATION),scorer_bindings={str(official):GQA_EVALUATOR_SHA,str(converter):GQA_CONVERTER_SHA}))
    freeze_json(OUT/'selection_audit.json',dict(total=len(selected),images=len({s['image_id'] for s in selected}),
        intended_total=3000,counts=counts,excluded_image_ids=len(used),excluded_question_ids=len(used_q),
        historical_sources=sources,sha_exclusions=rejected,historical_id_and_sha_overlap=0,
        caveat='Conservatively excludes every image_id recorded in prior manifests/configs, including reserved images. '
        'Not a perceptual-near-duplicate proof. Actual unbalanced available cohort is not all six balanced500/task.',
        shard_counts=dict(Counter(s['rank'] for s in selected))))
    print('FROZEN',len(selected),'questions',len({s['image_id'] for s in selected}),'images',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--census',action='store_true');args=parser.parse_args()
    if args.census:
        source,pools,used,hashes,qids,sources=census()
        print(json.dumps(dict(official_rows=len(source),excluded_images=len(used),sources=len(sources),
            pools={t:dict(questions=len(v),images=len({r['imageId'] for r in v})) for t,v in pools.items()}),indent=2))
    else:prepare()
