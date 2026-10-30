"""Reuse official frozen caption resources, excluding earlier observed images."""
import hashlib
import json
from collections import defaultdict
from PIL import Image
from benchmarks.dataset_registry import digest, freeze_json, sha256
from .protocol import ROOT, OUT, OLD, POOL, SEED, PROTOCOL


def main():
    prior = json.loads((OLD/'sample_manifest.json').read_text())
    exclude = prior['samples'] + [prior['smoke']]
    historical = json.loads((POOL/'data_manifest.json').read_text())
    annotation = '/data/lk/ATE-resources/paper_rq1/annotations/captions_val2014.json'
    if sha256(annotation) != historical['annotation_sha256'][annotation]:
        raise ValueError('Changed official COCO annotations')
    official = json.loads(open(annotation).read())
    image_rows = {str(r['id']):r for r in official['images']}
    refs = defaultdict(list)
    for r in official['annotations']: refs[str(r['image_id'])].append(r['caption'])
    bounds = {str(OLD/'sample_manifest.json'):sha256(OLD/'sample_manifest.json'),
              str(POOL/'data_manifest.json'):sha256(POOL/'data_manifest.json'),annotation:sha256(annotation)}
    discovery = set()
    for p in POOL.glob('discovery/shard*/checkpoints/coco_caption__*.json'):
        r = json.loads(p.read_text()); discovery.add(r['question_id'])
    pool = []
    for p in sorted((POOL/'manifests').glob('shard*.json')):
        if sha256(p) != historical['shard_sha256'][str(p.relative_to(POOL))]:
            raise ValueError('Changed historical input manifest')
        bounds[str(p)] = sha256(p)
        for r in json.loads(p.read_text())['samples']:
            if r['dataset'] != 'coco_caption': continue
            if r['question_id'] in discovery: exclude.append(r)
            elif r['split']=='confirmation': pool.append(r)
    excluded_ids = {r['image_identity'] for r in exclude}
    excluded_sha = {r['image_sha256'] for r in exclude}
    excluded_rgb = {r['rgb_sha256'] for r in exclude}
    eligible = [r for r in pool if r['image_identity'] not in excluded_ids and
        r['image_sha256'] not in excluded_sha and r['rgb_sha256'] not in excluded_rgb]
    eligible.sort(key=lambda r:digest([SEED,'late-source-images-v1',r['image_identity'],r['image_sha256']]))
    chosen = []; ids=set(); shas=set(); rgbs=set()
    for r in eligible:
        if r['image_identity'] in ids or r['image_sha256'] in shas or r['rgb_sha256'] in rgbs: continue
        iid = r['image_identity'].split(':')[-1]
        if r['raw_row'] != image_rows[iid] or r['answer'] != refs[iid] or r['prompt_text'] != PROTOCOL['prompt']:
            raise ValueError('Changed official source/prompt')
        if sha256(r['image_path']) != r['image_sha256']: raise ValueError('Changed image bytes')
        with Image.open(r['image_path']) as image:
            image.load(); image=image.convert('RGB')
            rgb=digest([image.size,hashlib.sha256(image.tobytes()).hexdigest()])
        if rgb != r['rgb_sha256']: raise ValueError('Changed decoded RGB identity')
        ids.add(r['image_identity']);shas.add(r['image_sha256']);rgbs.add(rgb)
        chosen.append(dict(r,rank=len(chosen)%4))
        if len(chosen)==201:break
    if len(chosen)!=201:raise ValueError(f'Insufficient independent images: {len(chosen)}, no replacement allowed')
    OUT.mkdir(parents=True,exist_ok=True)
    freeze_json(OUT/'sample_manifest.json',dict(samples=chosen[:200],smoke=chosen[200],protocol_sha256=digest(PROTOCOL)))
    freeze_json(OUT/'data_manifest.json',dict(status='ready',images=200,smoke_images=1,source_sha256=bounds,
        sample_sha256=sha256(OUT/'sample_manifest.json'),prior_observation2_images=200,prior_discovery_images=len(discovery),
        overlap_previous_ids=0,overlap_previous_bytes=0,overlap_previous_rgb=0,eligible_pool=len(eligible),
        selection_uses_outputs=False,rank_counts={str(i):sum(r['rank']==i for r in chosen[:200]) for i in range(4)}))
    print('Frozen200 independent new images plus disjoint smoke; four shards of50.',flush=True)


if __name__=='__main__':main()
