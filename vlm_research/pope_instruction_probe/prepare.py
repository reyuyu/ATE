"""Freeze exact inputs and obtain only the official COCO val2014 annotation member."""
import argparse
import datetime
import io
import sys
import urllib.request
import zipfile

from PIL import Image
from transformers import AutoProcessor

from .common import *

URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'


class RemoteZip(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as r:
            self.length = int(r.headers['Content-Length'])
            self.etag = r.headers['ETag']
        self.pos, self.requests, self.bytes = 0, [], 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.length if whence == 2 else 0)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, count=-1):
        count = self.length-self.pos if count < 0 else min(count, self.length-self.pos)
        if count <= 0:
            return b''
        start, end = self.pos, self.pos+count-1
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 206, 'Server ignored byte range; do not download entire archive'
            assert response.headers['ETag'] == self.etag
            data = response.read()
        assert len(data) == count
        self.requests.append([start, end]); self.bytes += len(data); self.pos += len(data)
        return data


def annotations():
    path = RESOURCES/'official_coco_roi_subset.json'
    if path.exists():
        return read(path)
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as z:
        info = z.getinfo('annotations/instances_val2014.json')
        raw = z.read(info)  # ZipFile checks the official ZIP member CRC.
    data = json.loads(raw)
    ids = {25972, 149592, 515904}
    categories = {c['id']: c['name'] for c in data['categories']}
    subset = dict(images=[i for i in data['images'] if i['id'] in ids],
        annotations=[dict(a, category_name=categories[a['category_id']]) for a in data['annotations'] if a['image_id'] in ids],
        categories=data['categories'], source=dict(url=URL, archive_etag=remote.etag,
        member='annotations/instances_val2014.json', member_sha256=hashlib.sha256(raw).hexdigest(),
        member_crc32=info.CRC, member_compressed_bytes=info.compress_size,
        network_bytes=remote.bytes, byte_ranges=remote.requests, acquired_at=datetime.datetime.now().astimezone().isoformat()))
    atomic(path, subset)
    return subset


def prepare():
    assert not (OUT/'freeze.json').exists(), 'Frozen experiment already exists'
    sys.path.insert(0, str(ROOT/'vlm_research'))
    from ..roi_sink_common import geometry, map_box
    p = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    all_samples = read(HISTORY/'sample_manifest.json')
    ann = annotations(); samples = {}; mappings = []; roi_audit = []
    for key in KEYS:
        s = all_samples[key]; m = s['input_mapping']; qid = str(s['question_id'])
        assert sha(s['image_path']) == s['image_sha256']
        assert len(m['parts']['Instruction']) == 10
        samples[key] = s
        directory = OUT/'images'/qid; directory.mkdir(parents=True, exist_ok=True)
        with Image.open(s['image_path']) as image:
            g, crop = geometry(image, p.image_processor)
            Image.fromarray(crop).save(directory/'processor_crop.png')
        image_id = int(Path(s['image_id']).stem.split('_')[-1])
        same_class = [a for a in ann['annotations'] if a['image_id'] == image_id and a['category_name'] == TARGETS[qid]]
        objects = [a for a in same_class if a['id'] in TARGET_ANNOTATIONS[qid]]
        assert sorted(a['id'] for a in objects) == sorted(TARGET_ANNOTATIONS[qid])
        boxes = [dict(annotation_id=a['id'], category=a['category_name'], iscrowd=a['iscrowd'],
                      **map_box(a['bbox'], g)) for a in objects]
        other_boxes = [dict(annotation_id=a['id'],category=a['category_name'],**map_box(a['bbox'],g))
            for a in same_class if a['id'] not in TARGET_ANNOTATIONS[qid]]
        assert objects, f'Official target bbox missing: {qid}; do not invent ROI'
        fractions = np.zeros((24,24))
        # Official boxes are unioned geometrically per patch, not double-counted.
        for r in range(24):
            for c in range(24):
                rects=[]
                for b in boxes:
                    x0,y0,x1,y1=b['crop_xyxy']
                    rect=[max(c*14,x0),max(r*14,y0),min((c+1)*14,x1),min((r+1)*14,y1)]
                    if rect[0]<rect[2] and rect[1]<rect[3]: rects.append(rect)
                edges=sorted({v for rect in rects for v in (rect[0],rect[2])})
                area=0.
                for left,right in zip(edges,edges[1:]):
                    intervals=sorted((a[1],a[3]) for a in rects if a[0] <= left and a[2] >= right)
                    hi=-float('inf'); length=0.
                    for lo,end in intervals:
                        length+=max(0.,end-max(lo,hi)); hi=max(hi,end)
                    area+=(right-left)*length
                fractions[r,c]=area/196
        visible = bool(fractions.sum()>0)
        roi = dict(question_id=qid, target=TARGETS[qid], geometry=g, boxes=boxes,other_same_class_boxes=other_boxes,
            target_annotation_ids=TARGET_ANNOTATIONS[qid],
            status='OFFICIAL_VISIBLE' if visible else 'OFFICIAL_FULLY_CROPPED_OUT',
            roi_patch_fractions=fractions.reshape(-1).tolist(), roi_area_fraction=float(fractions.mean()),
            positive_overlap_patch_count=int((fractions>0).sum()),
            note='Official COCO instance bounding boxes, not segmentation; cropped-out ROI ratios/enrichment are NA, not evidence of absence.')
        atomic(directory/'roi.json', roi); roi_audit.append({k:v for k,v in roi.items() if k not in ('boxes','roi_patch_fractions')})
        for i,pos in enumerate(m['R']):
            groups=[name for name,ls in m['parts'].items() if pos in ls]
            mappings.append(dict(question_id=qid, r_index=i, position=pos, token_id=m['prompt_ids'][pos],
                subword=p.tokenizer.convert_ids_to_tokens(m['prompt_ids'][pos]), decoded=p.tokenizer.decode([m['prompt_ids'][pos]]),
                region=groups[0] if groups else 'Separator', E1_writer=pos in writer_set(m,'E1')))
    atomic(OUT/'sample_manifest.json', samples)
    atomic(OUT/'roi_audit.json', roi_audit)
    write_csv(OUT/'token_mapping.csv', mappings)
    variants={k:{form:candidate_ids(p.tokenizer,s['input_mapping'],form) for form in FORMS} for k,s in samples.items()}
    protocol=dict(base_commit=SOURCE_COMMIT, gpu=GPU, model_revision=Path(MODEL_PATH).name,
        dtype='float16', attention='eager', transformers='4.49.0', config=CFG,
        samples=list(KEYS), conditions=conditions(10), new_full_generations=3*len(conditions(10)),
        target_annotation_ids=TARGET_ANNOTATIONS,
        scoring=dict(primary='sum logP of exact canonical suffix " Yes" minus " No", no EOS; full vocabulary FP32 log_softmax',
            variants=variants, multiple_tokens='Teacher-force every candidate token with an independent condition cache; no extra BOS/EOS',
            lowercase_and_no_space='Separate sensitivity audits, never silently mix or choose favorable form'),
        attention_capture='Actual kernel-produced post-softmax R->576 Image rows, L0-L19, all 32 heads; observer does not edit tensors',
        edge='Outer two 24x24 patch rings; 176/576 patches. Ratio is conditional on Image mass; separately record absolute mass.',
        roi='Official COCO boxes mapped by actual integer resize + center crop; bbox-overlap fractions; outside-crop targets remain NA',
        plots='Head mean for layer galleries; all-head atlases and exact NPZ retained. Fixed log2(1+576*p_visual) scale for all-head conditional maps.',
        addback='Restore the selected native instruction subwords in all L0-L19; all contiguous windows of length 1,2,4; C3 is all10 positive control',
        scope='Three selected historical failure cases; descriptive association and local add-back effects, no generalization/independence claim',
        no_training=True, no_input_edits=True, no_new_conditions_after_outputs=True)
    atomic(OUT/'protocol.json', protocol)
    frozen=dict(sources=sources(), sample_manifest_sha256=sha(OUT/'sample_manifest.json'),
        protocol_sha256=sha(OUT/'protocol.json'), history_manifest_sha256=sha(HISTORY/'sample_manifest.json'),
        official_roi_subset_sha256=sha(RESOURCES/'official_coco_roi_subset.json'),
        weights=read(HISTORY/'execution_freeze.json')['weights'])
    atomic(OUT/'freeze.json', frozen)
    print('FROZEN',len(samples),len(conditions(10)), 'conditions', 'ROIs',roi_audit,flush=True)


if __name__ == '__main__':
    prepare()
