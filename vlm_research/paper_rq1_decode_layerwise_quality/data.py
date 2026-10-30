"""Input-only selection, historical-overlap inventory and exact processor binding."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from attention_position_probe import locate_question_tokens
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.protocols import SHORT
from paper_rq1.data import sample
from paper_rq1.inventory import RESOURCES, OLD
from .protocol import ROOT, OUT, MODEL_PATH, SEED, PROTOCOL


def rgb_sha(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        return digest([image.size,hashlib.sha256(image.tobytes()).hexdigest()])


def unique_input_order(rows, seed=SEED):
    ordered=sorted(rows,key=lambda r:digest([seed,r['image_identity'],r['question_id']]))
    result=[]; seen=set()
    for row in ordered:
        if row['image_identity'] not in seen:
            seen.add(row['image_identity']); result.append(row)
    return result


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    old=ROOT/'results/paper_rq1_attention_time'; bindings={}; excluded=[]
    shards=[]
    for p in sorted((old/'manifests').glob('shard*.json')):
        bindings[str(p)]=sha256(p); shards.extend(json.loads(p.read_text())['samples'])
    prior_coco_ids={json.loads(p.read_text())['question_id'] for p in (old/'discovery').glob('shard*/checkpoints/coco_caption__*.json')}
    for s in shards:
        if s['dataset']=='coco_caption' and s['question_id'] in prior_coco_ids:excluded.append(s)
    for name in ['paper_rq1_token_role_decode','paper_rq1_late_attention_sources']:
        p=ROOT/'results'/name/'sample_manifest.json';bindings[str(p)]=sha256(p)
        m=json.loads(p.read_text());excluded.extend(m['samples']+[m['smoke']])
    p=ROOT/'results/paper_rq1_function_attention_sparsify/sample_manifest.json';bindings[str(p)]=sha256(p)
    m=json.loads(p.read_text());excluded.extend(m['development']+m['holdout']+[m['smoke']])
    used_ids={s['image_identity'] for s in excluded};used_sha={s['image_sha256'] for s in excluded};used_rgb={s['rgb_sha256'] for s in excluded}
    coco_pool=[s for s in shards if s['dataset']=='coco_caption']
    coco=[]
    for s in unique_input_order(coco_pool):
        if s['image_identity'] in used_ids or s['image_sha256'] in used_sha or s['rgb_sha256'] in used_rgb:continue
        coco.append(dict(s,prior_experiment_overlap=False))
        if len(coco)==201:break
    if len(coco)<201:raise ValueError('Need200 independent COCO images and separate smoke; no silent reuse')
    # Previous TextVQA protocols are not hidden confirmation evidence for this pilot.
    prior_text={s['image_identity'] for s in shards if s['dataset']=='textvqa'}
    prior_text_sha={s['image_sha256'] for s in shards if s['dataset']=='textvqa'}
    for p in sorted((ROOT/'results').glob('*/config.json')):
        try: config=json.loads(p.read_text())
        except (ValueError,OSError):continue
        jobs=config.get('jobs',[])
        tv=[j for j in jobs if j.get('unit')=='textvqa']
        if tv:
            bindings[str(p)]=sha256(p);prior_text_sha.update(j['image_sha256'] for j in tv)
    for p in (ROOT/'results').glob('*/per_sample.csv'):
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get('dataset',r.get('unit'))=='textvqa' and r.get('image_sha256'):prior_text_sha.add(r['image_sha256'])
    annotation=OLD/'sources/textvqa/TextVQA_0.5.1_val.json';bindings[str(annotation)]=sha256(annotation)
    originals=json.loads(annotation.read_text())['data']; candidates=[]
    for r in originals:
        iid=r['image_id'];paths=[RESOURCES/'images/openimages'/f'{iid}.jpg',OLD/'images/textvqa'/f'{iid}.jpg']
        path=next((p for p in paths if p.is_file()),None)
        if path is None:continue
        s=sample('textvqa',r['question_id'],'openimages:'+iid,r['question'],r['answers'],r,'official TextVQA0.5.1 validation',path)
        s.update(image_sha256=sha256(path),rgb_sha256=rgb_sha(path))
        s['prior_experiment_overlap']=s['image_identity'] in prior_text or s['image_sha256'] in prior_text_sha
        candidates.append(s)
    candidates=unique_input_order(candidates)
    candidates.sort(key=lambda s:s['prior_experiment_overlap'])
    picked=[];seen={s['image_sha256'] for s in coco};seen_rgb={s['rgb_sha256'] for s in coco}
    for s in candidates:
        if s['image_sha256'] in seen or s['rgb_sha256'] in seen_rgb:continue
        picked.append(s);seen.add(s['image_sha256']);seen_rgb.add(s['rgb_sha256'])
        if len(picked)==201:break
    if len(picked)!=201:raise ValueError('Insufficient unique official TextVQA images')
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default'
    processor.tokenizer.padding_side='left'
    samples=[];smoke=[]
    for unit,chosen in [('coco_caption',coco),('textvqa',picked)]:
        for i,s in enumerate(chosen):
            s=dict(s);s['rank']=i%3;s['split']='exploration';s.pop('shard',None)
            if sha256(s['image_path'])!=s['image_sha256'] or rgb_sha(s['image_path'])!=s['rgb_sha256']:raise ValueError('Image changed')
            text=PROTOCOL['coco_prompt'] if unit=='coco_caption' else s['question']+'\n'+SHORT
            conversation=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
            rendered=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
            with Image.open(s['image_path']) as image:inputs=processor(images=image.convert('RGB'),text=rendered,return_tensors='pt')
            ids=inputs['input_ids'][0].tolist()
            if ids.count(32000)!=576:raise ValueError('Wrong visual token layout')
            positions,_=locate_question_tokens(processor,text,ids,32000)
            s.update(prompt_text=text,input_mapping=dict(prompt_ids=ids,p_minus_1=len(ids)-1,question_positions=[r['sequence_index'] for r in positions]))
            (smoke if i==200 else samples).append(s)
    refs=json.loads((RESOURCES/'annotations/captions_val2014.json').read_text());byid=defaultdict(list)
    for row in refs['annotations']:byid[str(row['image_id'])].append(row['caption'])
    for s in coco:
        if s['answer']!=byid[s['image_identity'].split(':')[-1]]:raise ValueError('COCO original references differ')
    bindings[str(RESOURCES/'annotations/captions_val2014.json')]=sha256(RESOURCES/'annotations/captions_val2014.json')
    freeze_json(OUT/'experiment_config.json',PROTOCOL)
    freeze_json(OUT/'sample_manifest.json',dict(samples=samples,smoke=smoke,source_bindings=bindings,
        selection=dict(seed=SEED,coco_historical_excluded_images=len(used_ids),coco_unused_eligible_at_least=len(coco),
            textvqa_local_unique_candidates=len(candidates),textvqa_unused_local_candidates=sum(not s['prior_experiment_overlap'] for s in candidates),
            textvqa_selected_historical_overlap=sum(s['prior_experiment_overlap'] for s in samples if s['dataset']=='textvqa'),
            caveat='Input-only exploratory selection, not independent confirmation; exact bytes/RGB checked, not perceptual-near-duplicate proof.'),
        counts={unit:sum(s['dataset']==unit for s in samples) for unit in ['coco_caption','textvqa']}))
    print('Frozen',len(samples),'samples;TextVQA prior overlap',sum(s['prior_experiment_overlap'] for s in samples),flush=True)


if __name__=='__main__':prepare()
