import json
from pathlib import Path
from PIL import Image
import hashlib
from benchmarks.dataset_registry import digest, freeze_json, sha256
from .protocol import ROOT,OUT,OLD,SEED,PROTOCOL


def prepare():
    development=[];excluded=[];bindings={}
    for cohort,base in enumerate(OLD):
        manifest=base/'sample_manifest.json';targets=base/'selected_token_manifest.json'
        bindings[str(manifest)]=sha256(manifest);bindings[str(targets)]=sha256(targets)
        source=json.loads(manifest.read_text());selected=json.loads(targets.read_text())['images']
        excluded.extend(source['samples']+[source['smoke']])
        for sample in source['samples']:
            path=base/f"full/shard{sample['rank']}/checkpoints/{sample['image_identity'].replace(':','_')}.json"
            generation=json.loads(path.read_text())
            if sha256(generation['raw_path'])!=generation['raw_sha256']:raise ValueError('Historical raw changed')
            development.append(dict(sample,cohort=cohort,old_checkpoint=str(path),old_sha256=sha256(path),
                selected=selected[sample['image_identity']]['selected']))
    development.sort(key=lambda s:digest([SEED,'function-predictor-split',s['image_sha256']]))
    for i,s in enumerate(development):s.update(split='train' if i<300 else 'calibration',rank=i%3)
    pool=ROOT/'results/paper_rq1_attention_time';eligible=[]
    prior_ids={json.loads(p.read_text())['question_id'] for p in pool.glob('discovery/shard*/checkpoints/coco_caption__*.json')}
    for path in sorted((pool/'manifests').glob('shard*.json')):
        bindings[str(path)]=sha256(path)
        for s in json.loads(path.read_text())['samples']:
            if s['dataset']!='coco_caption':continue
            if s['question_id'] in prior_ids:excluded.append(s)
            elif s['split']=='confirmation':eligible.append(s)
    ids={s['image_identity'] for s in excluded};sha={s['image_sha256'] for s in excluded};rgb={s['rgb_sha256'] for s in excluded}
    eligible.sort(key=lambda s:digest([SEED,'function-sparsify-holdout',s['image_sha256']]))
    holdout=[]
    for s in eligible:
        if s['image_identity'] in ids or s['image_sha256'] in sha or s['rgb_sha256'] in rgb:continue
        holdout.append(dict(s,rank=len(holdout)%3,split='holdout'))
        ids.add(s['image_identity']);sha.add(s['image_sha256']);rgb.add(s['rgb_sha256'])
        if len(holdout)==200:break
    if len(development)!=400 or len(holdout)!=200:raise ValueError('Insufficient frozen image identities')
    for s in development+holdout:
        if sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Image bytes changed')
        with Image.open(s['image_path']) as im:
            im=im.convert('RGB');value=digest([im.size,hashlib.sha256(im.tobytes()).hexdigest()])
        if value!=s['rgb_sha256']:raise ValueError('Decoded image changed')
    smoke=json.loads((OLD[1]/'sample_manifest.json').read_text())['smoke']
    OUT.mkdir(parents=True,exist_ok=True)
    freeze_json(OUT/'sample_manifest.json',dict(development=development,holdout=holdout,smoke=smoke,
        protocol=PROTOCOL,source_bindings=bindings,holdout_outputs_read=False))
    print('Frozen development400 (300train/100calibration), independent holdout200, disjoint smoke.',flush=True)


if __name__=='__main__':prepare()
