"""Authorized same-image/same-object-ID scene-graph supplementation."""
from collections import Counter
import csv
import json
import math
import re
import zipfile

from .preflight import ANNOTATION, HISTORY, OUT, SOURCES, canonical, dump, sha

ARCHIVE = ANNOTATION.parent / 'gqa_sceneGraphs.zip'
URL = 'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip'


def target_ids(semantic, index=None):
    index = len(semantic)-1 if index is None else index
    step = semantic[index]
    assert all(0 <= d < index for d in step['dependencies'])
    found = re.search(r'\(([\d, -]+)\)\s*$', step['argument'])
    own = set(re.findall(r'\d+', found[1])) if found else set()
    if step['operation'] in ('select','relate'):
        # A relate step names its output objects, not all earlier reference objects.
        assert found is not None, 'Unannotated semantic target'
        return own
    assert step['operation'].startswith(('choose','query','filter','exist','common','or'))
    return own.union(*(target_ids(semantic,d) for d in step['dependencies']))


def clip_box(xywh, size):
    x,y,w,h = map(float,xywh)
    width,height = map(float,size)
    assert all(math.isfinite(v) for v in (x,y,w,h,width,height)) and min(w,h,width,height)>0
    x0,y0,x1,y1 = max(0.,x),max(0.,y),min(width,x+w),min(height,y+h)
    assert x1>x0 and y1>y0, 'Target box does not intersect image'
    clipped=[x0,y0,x1-x0,y1-y0]
    return clipped,clipped != list(map(float,xywh))


def resolve():
    import pyarrow.parquet as pq
    from PIL import Image
    samples=json.loads((HISTORY/'sample_manifest.json').read_text())['samples']
    originals={canonical(r['id']):r for r in pq.read_table(ANNOTATION).to_pylist()}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        raw=z.read('val_sceneGraphs.json')
    graphs=json.loads(raw)
    csv_rows={}
    for task,(name,_) in SOURCES.items():
        for row in csv.DictReader((OUT/'sources'/name).open()):
            csv_rows[(task,canonical(row['question_id']))]=row
    all_boxes,semantic_records,box_rows={},[],[]
    csv_matches=0
    for s in samples:
        original=originals[canonical(s['question_id'])]
        assert original['question']==s['question'] and original['answer']==s['answer']
        ids=target_ids(original['semantic'])
        assert ids, 'No positive existing target ROI; stop rather than fabricate'
        graph=graphs[s['image_id']]
        with Image.open(s['image_path']) as image:size=list(image.size)
        assert size==[graph['width'],graph['height']]
        row=csv_rows.get((s['task'],canonical(s['question_id'])))
        official={}
        if row:
            assert row['question']==s['question'] and row['answer']==s['answer']
            assert canonical(row['img_id'])==canonical(s['image_id'])
            for p in ('central object','object1','object2'):
                if p+' id' in row and row[p+' id']!='-':
                    official[canonical(row[p+' id'])]=[float(row[p+' '+c]) for c in ('x','y','w','h')]
            assert set(official)==ids, 'Semantic target definition differs from official CSV'
            csv_matches+=1
        boxes=[]
        for oid in sorted(ids,key=int):
            obj=graph['objects'][oid]
            sg_box=[float(obj[c]) for c in ('x','y','w','h')]
            if row:
                assert official[oid]==sg_box, 'CSV vs scene graph coordinates differ'
            original_box=official.get(oid,sg_box)
            clipped,changed=clip_box(original_box,size)
            source='official_cross_modal_csv' if row else 'official_gqa_scene_graph'
            b=dict(object_id=oid,name=obj['name'],xywh=clipped,original_xywh=original_box,
                   image_size=size,source=source,clipped=changed,
                   target_id_source='Frozen original GQA annotated semantic program; terminal dependencies',
                   csv_scene_graph_exact=bool(row))
            boxes.append(b)
            box_rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],object_id=oid,source=source,
                original_xywh=json.dumps(original_box),xywh=json.dumps(clipped),clipped=int(changed),
                csv_scene_graph_exact=int(bool(row))))
        key=f'{s["task"]}__{s["question_id"]}'
        all_boxes[key]=boxes
        semantic_records.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
            semantic=original['semantic'],target_ids=sorted(ids,key=int),input_binding=s['input_binding']))
    assert len(all_boxes)==240 and csv_matches==120
    dump(OUT/'resolved_boxes.json',all_boxes)
    dump(OUT/'target_semantics.json',semantic_records)
    with (OUT/'bbox_provenance.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(box_rows[0]),lineterminator='\n');w.writeheader();w.writerows(box_rows)
    # Store only the 240-image subset, not the unrelated complete scene-graph corpus.
    dump(OUT/'sources/gqa_val_scene_graphs_selected.json',{s['image_id']:graphs[s['image_id']] for s in samples})
    import hashlib
    dump(OUT/'annotation_resolution.json',dict(status='complete',samples=240,
        user_authorized='Same-object official GQA scene graph supplementation and explicit boundary clipping',
        scene_graph_url=URL,archive_path=str(ARCHIVE),archive_sha256=sha(ARCHIVE),
        val_scene_graphs_sha256=hashlib.sha256(raw).hexdigest(),
        csv_questions_matched=120,csv_vs_semantic_target_sets_exact=120,csv_vs_scene_graph_coordinates_exact=120,
        supplemented_questions=120,total_target_boxes=len(box_rows),
        clipped_boxes=[r for r in box_rows if r['clipped']],
        clipped_questions=len({r['question_id'] for r in box_rows if r['clipped']}),
        tasks=dict(Counter(s['task'] for s in samples))))
    print(f'Resolved all 240 questions: 120 CSV + 120 same-ID scene graphs; {sum(r["clipped"] for r in box_rows)} clipped boxes.',flush=True)


if __name__=='__main__':
    resolve()
