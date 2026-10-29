"""Input-only selection plan, source-aware image identities and disjoint frozen shards."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor

from attention_position_probe import locate_question_tokens
from benchmarks.acquire import download
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.fetch_resources import RangeReader
from benchmarks.protocols import MODEL_PATH, SHORT, prompt_text
from benchmarks.prepare_datasets import token_mapping
from benchmarks.scienceqa_adapter import verify_publisher_order
from .acquire import GQA_REV
from .inventory import RESOURCES, OUTPUT, OLD
from .protocol import COUNTS, PROTOCOL, SEED


def order(value):
    return digest([SEED, value])


def sample(unit, qid, identity, question, answer, raw, source, image_path=None):
    return dict(dataset=unit, question_id=str(qid), image_identity=identity, question=question, answer=answer,
                source=source, source_row_sha256=digest(raw), raw_row=raw,
                image_path=str(image_path or RESOURCES/'images'/(identity.replace(':', '/')+'.jpg')))


def one_per_image(rows, count, excluded=()):
    by_image = defaultdict(list)
    for row in rows:
        if row['image_identity'] not in excluded:
            by_image[row['image_identity']].append(row)
    ids = sorted(by_image, key=order)[:count]
    return [min(by_image[i], key=lambda r: order(r['question_id'])) for i in ids]


def make_plan():
    import pyarrow.parquet as pq
    sources = RESOURCES/'annotations'
    gqa = []
    for batch in pq.ParquetFile(sources/'gqa_val_balanced.parquet').iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row['isBalanced'] is not True or not row['answer']:
                raise ValueError('Not official balanced answered validation data')
            iid = str(row['imageId'])
            gqa.append(sample('gqa_balanced', row['id'], 'vg:'+iid, row['question'], row['answer'], row,
                f'lmms-lab/GQA@{GQA_REV}:val_balanced_instructions',
                Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')/f'image_{iid}.jpg'))
    gqa_selected = one_per_image(gqa, 1300)
    # ScienceQA image bytes and every public content field must match the official PID order.
    problems = json.loads((OLD/'sources/scienceqa/data/scienceqa/problems.json').read_text())
    ids = json.loads((OLD/'sources/scienceqa/data/scienceqa/pid_splits.json').read_text())['test']
    mirror = pq.read_table(OLD/'sources/scienceqa_hf/test-00000-of-00001-f0e719df791966ff.parquet').to_pylist()
    matched = verify_publisher_order(problems, ids, mirror)
    science, seen = [], set()
    for qid in sorted(ids, key=order):
        row = problems[qid]
        if not row['image']:
            continue
        data = matched[qid]['image']['bytes']
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert('RGB')
            key = digest([rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()])
        if key in seen:
            continue
        seen.add(key)
        path = RESOURCES/'images/scienceqa'/qid/row['image']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError('ScienceQA existing bytes differ')
        if not path.exists():
            with path.open('xb') as f:
                f.write(data)
        s = sample('scienceqa_img', qid, 'scienceqa:'+key, row['question'], 'ABCDE'[row['answer']], row,
                   'lupantech/ScienceQA@2cbf8318e07b9ece895bb2ae605e71e38d623264:test', path)
        s['candidates'] = list('ABCDE'[:len(row['choices'])])
        s['metadata'] = dict(raw_row=row, gold_answer_index=row['answer'])
        science.append(s)
        if len(science) == 1100:
            break
    tc = json.loads((sources/'textcaps_val.json').read_text())['data']
    tc_rows = [sample('textcaps', 'caption:'+r['image_id'], 'openimages:'+r['image_id'],
                     PROTOCOL['caption_prompt'], r['reference_strs'], r,
                     'official TextCaps0.1 validation (five publisher reference_strs)') for r in tc]
    textcaps = one_per_image(tc_rows, 1400)
    reserved = {s['image_identity'] for s in textcaps}
    tv = json.loads((OLD/'sources/textvqa/TextVQA_0.5.1_val.json').read_text())['data']
    textvqa = one_per_image([sample('textvqa', r['question_id'], 'openimages:'+r['image_id'],
        r['question'], r['answers'], r, 'official TextVQA0.5.1 validation') for r in tv], 1300, reserved)
    pope = [json.loads(line) for line in (OLD/'sources/pope/output/coco/coco_pope_adversarial.json').read_text().splitlines() if line]
    groups = defaultdict(list)
    for r in pope:
        iid = int(Path(r['image']).stem.split('_')[-1])
        groups['coco2014:'+str(iid)].append(sample('pope_adversarial', r['question_id'], 'coco2014:'+str(iid),
            r['text'], r['label'], r, 'AoiDragon/POPE@08d957b917e5a378a2f99d35b6293c536a66298b:adversarial'))
    chosen = sorted(groups, key=order)[:200]
    if any(len(groups[k]) != 6 for k in chosen):
        raise ValueError('POPE requires frozen six-question image groups')
    pope_selected = [r for k in chosen for r in sorted(groups[k], key=lambda r: r['question_id'])]
    annotations = json.loads((sources/'captions_val2014.json').read_text())
    refs = defaultdict(list)
    for r in annotations['annotations']:
        refs[r['image_id']].append(r['caption'])
    coco = one_per_image([sample('coco_caption', 'caption:'+str(r['id']), 'coco2014:'+str(r['id']),
        PROTOCOL['caption_prompt'], refs[r['id']], r, 'official COCO2014 validation captions')
        for r in annotations['images']], 1400, set(chosen))
    selections = dict(gqa_balanced=gqa_selected, scienceqa_img=science, pope_adversarial=pope_selected,
                      textvqa=textvqa, coco_caption=coco, textcaps=textcaps)
    if any(len(selections[k]) != sum(COUNTS[k]) for k in COUNTS):
        raise ValueError('Annotation pool insufficient for target; inspect counts, do not duplicate or substitute')
    files = [sources/'gqa_val_balanced.parquet', sources/'gqa_README.md', sources/'textcaps_val.json',
             sources/'captions_val2014.json', OLD/'sources/textvqa/TextVQA_0.5.1_val.json',
             OLD/'sources/scienceqa/data/scienceqa/problems.json', OLD/'sources/scienceqa/data/scienceqa/pid_splits.json',
             OLD/'sources/pope/output/coco/coco_pope_adversarial.json']
    value = dict(status='selection_plan_not_GPU_ready', protocol=PROTOCOL, samples=selections,
                 annotation_sha256={str(p): sha256(p) for p in files},
                 annotation_counts=dict(gqa_balanced=len(gqa), textcaps_images=len({r['image_id'] for r in tc}),
                                        textvqa_questions=len(tv), pope_questions=len(pope), coco_images=len(annotations['images'])))
    freeze_json(OUTPUT/'selection_plan.json', value)
    print('Selected input-only plan', {k: len(v) for k, v in selections.items()}, flush=True)


def fetch_coco(row):
    identity = row['image_identity']
    iid = int(identity.split(':')[1])
    filename = f'COCO_val2014_{iid:012d}.jpg'
    path = Path(row['image_path'])
    old = OLD/'images/coco/val2014'/filename
    if old.exists() and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old, path)
    s = requests.Session()
    s.trust_env = False
    result = download(s, f'https://s3.amazonaws.com/images.cocodataset.org/val2014/{filename}', path, cap=10*1024**2)
    return dict(image_identity=identity, **result)


class BufferedMemberReader(RangeReader):
    """Prefetch one ZIP member range; zipfile remains responsible for headers/CRC."""
    def member(self, info):
        self.buffer_start = info.header_offset
        self.seek(self.buffer_start)
        self.buffer = super().read(min(info.compress_size+65536, self.size-self.buffer_start))

    def read(self, size=-1):
        buffer = getattr(self, 'buffer', b'')
        offset = self.position-getattr(self, 'buffer_start', 0)
        if size >= 0 and 0 <= offset and offset+size <= len(buffer):
            self.position += size
            return buffer[offset:offset+size]
        return super().read(size)


def fetch_openimages(rows):
    s = requests.Session()
    s.trust_env = False
    reader = BufferedMemberReader(s, 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip')
    records = []
    with zipfile.ZipFile(reader) as archive:
        names = {Path(i.filename).name: i for i in archive.infolist() if not i.is_dir()}
        for index, row in enumerate(rows):
            iid = row['image_identity'].split(':')[1]
            path = Path(row['image_path'])
            old = OLD/'images/textvqa'/f'{iid}.jpg'
            if old.exists() and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, path)
            if not path.exists():
                info = names[f'{iid}.jpg']
                reader.member(info)
                data = archive.read(info)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    f.write(data)
            records.append(dict(image_identity=row['image_identity'], status='verified', sha256=sha256(path)))
            if index % 100 == 0:
                print('OpenImages worker', len(rows), index+1, flush=True)
    return dict(images=records, zip_etag=reader.etag, transferred_bytes=reader.transferred_bytes, ranges=reader.records)


def fetch_images():
    plan = json.loads((OUTPUT/'selection_plan.json').read_text())
    rows = {r['image_identity']: r for group in plan['samples'].values() for r in group}
    coco = [r for k, r in rows.items() if k.startswith('coco2014:')]
    openimages = [r for k, r in rows.items() if k.startswith('openimages:')]
    # Four bounded network workers. Completed image files are never overwritten.
    records = list(ThreadPoolExecutor(4).map(fetch_coco, coco))
    (OUTPUT/'coco_images_acquisition.json').write_text(json.dumps(records, indent=2)+'\n')
    records = list(ThreadPoolExecutor(4).map(fetch_openimages, [openimages[i::4] for i in range(4)]))
    (OUTPUT/'openimages_acquisition.json').write_text(json.dumps(records, indent=2)+'\n')


def freeze_data():
    plan_path = OUTPUT/'selection_plan_final.json'
    plan = json.loads(plan_path.read_text())
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.patch_size, processor.num_additional_image_tokens = 14, 1
    processor.vision_feature_select_strategy = 'default'
    byte_owners, rgb_owners, identity_owners, shards, exclusions = {}, {}, {}, [[], [], [], []], []
    actual = {}
    for unit in COUNTS:
        grouped = defaultdict(list)
        for row in plan['samples'][unit]:
            grouped[row['image_cluster']].append(row)
        accepted = []
        for iid in sorted(grouped, key=order):
            rows = grouped[iid]
            path = Path(rows[0]['image_path'])
            if not path.is_file():
                exclusions.append(dict(identity=iid, reason='missing_image', dataset=unit))
                continue
            byte = sha256(path)
            with Image.open(path) as image:
                image = image.convert('RGB')
                rgb = digest([image.size, hashlib.sha256(image.tobytes()).hexdigest()])
                if iid in identity_owners or byte in byte_owners or rgb in rgb_owners:
                    exclusions.append(dict(identity=iid, reason='duplicate_image_identity_bytes_or_RGB', dataset=unit))
                    continue
                identity_owners[iid], byte_owners[byte], rgb_owners[rgb] = unit, unit, unit
                accepted.append((iid, rows, byte, rgb))
        discovery_target, confirmation_target = COUNTS[unit]
        count = dict(discovery=0, confirmation=0)
        for iid, rows, byte, rgb in accepted:
            split = 'discovery' if count['discovery'] < discovery_target else 'confirmation'
            for original in rows:
                row = dict(original, split=split, image_sha256=byte, rgb_sha256=rgb)
                if count[split] >= (discovery_target if split == 'discovery' else confirmation_target):
                    continue
                if unit == 'scienceqa_img':
                    text = prompt_text(row)
                elif unit in ('textcaps', 'coco_caption'):
                    text = PROTOCOL['caption_prompt']
                else:
                    text = row['question']+'\n'+SHORT
                conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
                rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                with Image.open(row['image_path']) as image:
                    inputs = processor(images=image.convert('RGB'), text=rendered, return_tensors='pt')
                ids = inputs['input_ids'][0].tolist()
                if unit in ('textcaps', 'coco_caption'):
                    question, _ = locate_question_tokens(processor, text, ids, 32000)
                    mapping = dict(prompt_ids=ids, p_minus_1=len(ids)-1,
                                   question_positions=[r['sequence_index'] for r in question])
                else:
                    mapping = token_mapping(processor, dict(row, candidates=row.get('candidates')))
                    if mapping['prompt_ids'] != ids or mapping['text_payload'] != text:
                        raise ValueError('Actual Processor input differs from frozen benchmark mapping')
                if ids.count(32000) != 576:
                    raise ValueError('Not576 visual patches')
                row.update(prompt_text=text, input_mapping=mapping)
                shard = int(order(iid), 16) % 4
                row['shard'] = shard
                shards[shard].append(row)
                count[split] += 1
        actual[unit] = count
    files = {}
    for i, rows in enumerate(shards):
        path = OUTPUT/f'manifests/shard{i}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = dict(samples=rows, protocol_sha256=digest(PROTOCOL))
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError('Frozen shard differs')
        if not path.exists():
            with path.open('x') as f:
                json.dump(content, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        files[str(path.relative_to(OUTPUT))] = sha256(path)
    complete = all(actual[k] == dict(zip(('discovery', 'confirmation'), COUNTS[k])) for k in COUNTS)
    freeze_json(OUTPUT/'data_manifest.json', dict(status='ready' if complete else 'partial', counts=actual,
        shard_sha256=files, protocol_sha256=digest(PROTOCOL), selection_plan_sha256=sha256(plan_path),
        annotation_sha256=plan['annotation_sha256'], exclusions=exclusions,
        deduplication='Global published VG/COCO/Flickr/OpenImages photo aliases, byte SHA256 and decoded RGB SHA256. '
                      'See final input selection refinement for priority, exclusions and coverage limits. '
                      'No result-based filtering. No perceptual near-duplicate guarantee.'))
    print('FROZEN', actual, 'complete=', complete, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'fetch', 'freeze'])
    args = parser.parse_args()
    {'plan': make_plan, 'fetch': fetch_images, 'freeze': freeze_data}[args.stage]()


if __name__ == '__main__':
    main()
