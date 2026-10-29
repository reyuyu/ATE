"""Resolve published photo aliases before freezing any GPU sample or outcome."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import zipfile

from PIL import Image

from benchmarks.dataset_registry import digest, freeze_json, sha256
from .acquire import GQA_REV
from .data import order, sample, fetch_coco, fetch_openimages
from .inventory import OLD, OUTPUT, RESOURCES
from .protocol import COUNTS, PROTOCOL


class PhotoAliases:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def link(self, keys):
        roots = sorted({self.find(k) for k in keys})
        for root in roots[1:]:
            self.parent[root] = roots[0]


def flickr_id(url):
    if not url:
        return None
    parsed = urlparse(url)
    if not (parsed.hostname or '').endswith('staticflickr.com'):
        return None
    value = Path(parsed.path).name.split('_')[0]
    return 'flickr:'+value if value.isdigit() else None


def build_pools(initial):
    import pyarrow.parquet as pq
    annotations = RESOURCES/'annotations'
    pools = {}
    gqa = []
    for batch in pq.ParquetFile(annotations/'gqa_val_balanced.parquet').iter_batches(batch_size=512):
        for r in batch.to_pylist():
            if r['isBalanced'] is not True or not r['answer']:
                raise ValueError('Invalid official GQA balanced row')
            iid = str(r['imageId'])
            gqa.append(sample('gqa_balanced', r['id'], 'vg:'+iid, r['question'], r['answer'], r,
                f'lmms-lab/GQA@{GQA_REV}:val_balanced_instructions',
                Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')/f'image_{iid}.jpg'))
    pools['gqa_balanced'] = gqa
    pools['scienceqa_img'] = initial['samples']['scienceqa_img']
    pope = [json.loads(line) for line in (OLD/'sources/pope/output/coco/coco_pope_adversarial.json').read_text().splitlines() if line]
    pools['pope_adversarial'] = [sample('pope_adversarial', r['question_id'],
        'coco2014:'+str(int(Path(r['image']).stem.split('_')[-1])), r['text'], r['label'], r,
        'AoiDragon/POPE@08d957b917e5a378a2f99d35b6293c536a66298b:adversarial') for r in pope]
    coco = json.loads((annotations/'captions_val2014.json').read_text())
    refs = defaultdict(list)
    for r in coco['annotations']:
        refs[r['image_id']].append(r['caption'])
    pools['coco_caption'] = [sample('coco_caption', 'caption:'+str(r['id']), 'coco2014:'+str(r['id']),
        PROTOCOL['caption_prompt'], refs[r['id']], r, 'official COCO2014 validation captions') for r in coco['images']]
    tc = json.loads((annotations/'textcaps_val.json').read_text())['data']
    pools['textcaps'] = [sample('textcaps', 'caption:'+r['image_id'], 'openimages:'+r['image_id'],
        PROTOCOL['caption_prompt'], r['reference_strs'], r, 'official TextCaps0.1 validation (five publisher reference_strs)') for r in tc]
    tv = json.loads((OLD/'sources/textvqa/TextVQA_0.5.1_val.json').read_text())['data']
    pools['textvqa'] = [sample('textvqa', r['question_id'], 'openimages:'+r['image_id'],
        r['question'], r['answers'], r, 'official TextVQA0.5.1 validation') for r in tv]
    aliases = PhotoAliases()
    vg_path = annotations/'vg_image_data.zip'
    with zipfile.ZipFile(vg_path) as archive:
        vg = json.loads(archive.read('image_data.json'))
    vg_ids = set()
    for r in vg:
        key = 'vg:'+str(r['image_id'])
        vg_ids.add(key)
        aliases.link([key]+(['coco2014:'+str(r['coco_id'])] if r['coco_id'] else [])+
                     (['flickr:'+str(r['flickr_id'])] if r['flickr_id'] else []))
    for r in coco['images']:
        alias = flickr_id(r.get('flickr_url'))
        aliases.link(['coco2014:'+str(r['id'])]+([alias] if alias else []))
    for r in tc+tv:
        keys = ['openimages:'+r['image_id']]
        for field in ('flickr_original_url', 'flickr_300k_url'):
            alias = flickr_id(r.get(field))
            if alias:
                keys.append(alias)
        aliases.link(keys)
    return pools, aliases, vg_ids, vg_path


def main():
    initial_path = OUTPUT/'selection_plan.json'
    initial = json.loads(initial_path.read_text())
    if (OUTPUT/'execution_manifest.json').exists():
        raise ValueError('Input refinement prohibited after execution freeze')
    pools, aliases, vg_ids, vg_path = build_pools(initial)
    # Input-only priority: preserve GQA/ScienceQA, then POPE/COCO, then TextCaps/TextVQA.
    priority = ['gqa_balanced', 'scienceqa_img', 'pope_adversarial', 'coco_caption', 'textcaps', 'textvqa']
    selected, rejected, coverage = {}, [], {}
    owners, bytes_seen, rgb_seen = {}, {}, {}
    for unit in priority:
        groups = defaultdict(list)
        for row in pools[unit]:
            groups[row['image_identity']].append(row)
        preferred = {r['image_identity'] for r in initial['samples'][unit]}
        candidates = sorted(groups, key=lambda i: (i not in preferred, order(i)))
        kept, selected_ids = [], set()
        for iid in candidates:
            canonical = aliases.find(iid)
            if canonical in owners:
                rejected.append(dict(dataset=unit, image_identity=iid, reason='published_photo_alias',
                                     canonical=canonical, owner=owners[canonical], in_draft=iid in preferred))
                continue
            if unit == 'gqa_balanced' and iid not in vg_ids:
                raise ValueError('No published VG identity metadata for selected GQA photo')
            rows = groups[iid]
            rows = sorted(rows, key=lambda r: order(r['question_id']))
            rows = rows if unit == 'pope_adversarial' else rows[:1]
            if unit == 'pope_adversarial' and len(rows) != 6:
                raise ValueError('Incomplete official POPE image group')
            path = Path(rows[0]['image_path'])
            if not path.is_file():
                if iid.startswith('coco2014:'):
                    fetch_coco(rows[0])
                elif iid.startswith('openimages:'):
                    fetch_openimages(rows[:1])
                else:
                    rejected.append(dict(dataset=unit, image_identity=iid, reason='missing_local_image', in_draft=iid in preferred))
                    continue
            byte = sha256(path)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                rgb_hash = digest([rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()])
            if byte in bytes_seen or rgb_hash in rgb_seen:
                rejected.append(dict(dataset=unit, image_identity=iid, reason='duplicate_bytes_or_RGB', in_draft=iid in preferred))
                continue
            owners[canonical] = dict(dataset=unit, identity=iid)
            bytes_seen[byte], rgb_seen[rgb_hash] = iid, iid
            kept.extend(dict(row, image_cluster=canonical) for row in rows)
            selected_ids.add(iid)
            if len(kept) == sum(COUNTS[unit]):
                break
        selected[unit] = kept
        coverage[unit] = dict(questions=len(kept), images=len(selected_ids), retained_draft_images=len(preferred & selected_ids),
                              added_images=sorted(selected_ids-preferred), removed_images=sorted(preferred-selected_ids))
        print(unit, coverage[unit]['questions'], 'new_images', len(coverage[unit]['added_images']), flush=True)
    if any(len(selected[k]) != sum(COUNTS[k]) for k in COUNTS):
        raise ValueError('Input pools insufficient after alias exclusion; no GPU work permitted')
    result = dict(status='final_input_selection_not_GPU_ready', samples=selected, protocol=PROTOCOL,
        annotation_sha256=dict(initial['annotation_sha256'], **{str(vg_path): sha256(vg_path)}),
        initial_draft_sha256=sha256(initial_path), annotation_counts=initial['annotation_counts'],
        refinement=dict(before_any_GPU_experiment=True, priority=priority,
            rule='Retain draft images in fixed seed-hash order where eligible, then next official-pool hash-ranked image; '
                 'exclude published VG/COCO/Flickr/OpenImages aliases and exact byte/RGB duplicates globally. '
                 'No outcome or label filtering; no perceptual-near-duplicate guarantee. ScienceQA has no publisher '
                 'photo-ID aliases, so byte/RGB checks only for its cross-source identity.',
            coverage=coverage, exclusions=rejected))
    freeze_json(OUTPUT/'selection_plan_final.json', result)


if __name__ == '__main__':
    main()
