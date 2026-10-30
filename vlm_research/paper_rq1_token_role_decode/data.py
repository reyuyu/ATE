import json
from collections import defaultdict
from PIL import Image
from benchmarks.dataset_registry import digest, freeze_json, sha256
from .protocol import OUT, OLD, SEED, PROTOCOL


def main():
    old = json.loads((OLD / 'data_manifest.json').read_text())
    annotation_path = '/data/lk/ATE-resources/paper_rq1/annotations/captions_val2014.json'
    if sha256(annotation_path) != old['annotation_sha256'][annotation_path]:
        raise ValueError('Official COCO annotation hash changed')
    annotations = json.loads(open(annotation_path).read())
    official = {str(r['id']): r for r in annotations['images']}
    refs = defaultdict(list)
    for r in annotations['annotations']:
        refs[str(r['image_id'])].append(r['caption'])
    used = set()
    for path in OLD.glob('discovery/shard*/checkpoints/coco_caption__*.json'):
        row = json.loads(path.read_text())
        used.add(row.get('question_id', path.stem.split('__')[1].replace('caption_', 'caption:')))
    pool = []
    bound = {str(OLD / 'data_manifest.json'): sha256(OLD / 'data_manifest.json'), annotation_path: sha256(annotation_path)}
    for path in sorted((OLD / 'manifests').glob('shard*.json')):
        if sha256(path) != old['shard_sha256'][str(path.relative_to(OLD))]:
            raise ValueError('Changed historical input manifest')
        bound[str(path)] = sha256(path)
        for row in json.loads(path.read_text())['samples']:
            if row['dataset'] != 'coco_caption' or row['question_id'] in used or row['split'] != 'confirmation':
                continue
            iid = row['image_identity'].split(':')[-1]
            if row['raw_row'] != official[iid] or row['answer'] != refs[iid] or sha256(row['image_path']) != row['image_sha256']:
                raise ValueError('COCO source or image mismatch')
            if row['prompt_text'] != PROTOCOL['prompt']:
                raise ValueError('Changed native caption prompt')
            pool.append(row)
    pool.sort(key=lambda r: digest([SEED, 'token-role-images-v1', r['image_identity'], r['image_sha256']]))
    selected = []; seen = set(); ids = set()
    for row in pool:
        if row['image_sha256'] in seen or row['image_identity'] in ids:
            continue
        with Image.open(row['image_path']) as im:
            im.load(); im = im.convert('RGB')
            import hashlib
            rgb = digest([im.size, hashlib.sha256(im.tobytes()).hexdigest()])
        if rgb != row['rgb_sha256']:
            raise ValueError('Changed decoded image')
        seen.add(row['image_sha256']); ids.add(row['image_identity'])
        selected.append(dict(row, rank=len(selected) % 4))
        if len(selected) == 201:
            break
    if len(selected) != 201:
        raise ValueError('Need200 unique images plus disjoint Smoke')
    manifest = dict(samples=selected[:200], smoke=selected[200], protocol_sha256=digest(PROTOCOL))
    freeze_json(OUT / 'sample_manifest.json', manifest)
    freeze_json(OUT / 'data_manifest.json', dict(status='ready', images=200, smoke_images=1,
        source_sha256=bound, sample_manifest_sha256=sha256(OUT / 'sample_manifest.json'),
        previously_observed_coco_images=len(used), overlaps_with_prior_coco_discovery=0,
        source_pool=len(pool), candidate_selection_uses_model_outputs=False,
        note='Reuse previously reserved but unexecuted COCO confirmation resources. Exclude earlier400 Discovery images; no claim of general grounding or all historical image-alias detection beyond frozen prior deduplication.'))
    print('Frozen200 images, four shards of50; disjoint Smoke image; no earlier COCO Discovery overlap.', flush=True)


if __name__ == '__main__':
    main()
