"""Post-collection feature portability, scorer and immutable-artifact review."""
import argparse
import inspect
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image
from benchmarks.dataset_registry import sha256,digest
from paper_rq1.run import preflight
from paper_rq1.scorers import caption_scores,DEPS
from paper_rq1_single_layer_gqa6.run import atomic_json
from .protocol import OUT,ROOT
from .predictor import OnlinePredictor,prefix_features,hidden_features


def feature_portability():
    if json.loads((OUT/'pipeline_status.json').read_text())['status']!='complete':raise ValueError('Wait for all GPU workers before independent tiny feature audit')
    resource=preflight(1)
    frozen=json.loads((OUT/'predictor_freeze.json').read_text());selected=frozen['selected']
    if selected is None:
        atomic_json(OUT/'feature_portability_audit.json',dict(status='not_applicable',reason='No deployable predictor'));return
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    with np.load(OUT/'predictor_parameters.npz') as f:parameters={k:f[k] for k in f.files}
    cpu=OnlinePredictor(parameters,selected['predictor'],selected['threshold'],'cpu')
    gpu=OnlinePredictor(parameters,selected['predictor'],selected['threshold'],'cuda:0')
    count=0;max_difference=0.;mismatches=[]
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    with torch.inference_mode():
        for sample in manifest['holdout']:
            path=OUT/f"quality/shard{sample['rank']}/checkpoints/{sample['image_identity'].replace(':','_')}.json";record=json.loads(path.read_text())
            with np.load(record['raw_path']) as f:h=f['h31_in']
            ids=record['runs']['S0']['token_ids']
            for t in range(1,len(ids)):
                # The audit deliberately receives no target id, logits or labels.
                consumed=ids[:t];hidden=torch.from_numpy(h[t]);a=float(cpu.probability(consumed,hidden));b=float(gpu.probability(consumed,hidden.cuda()))
                max_difference=max(max_difference,abs(a-b));count+=1
                if (a>=selected['threshold'])!=(b>=selected['threshold']):mismatches.append(dict(image_id=sample['image_identity'],t=t,cpu=a,gpu=b))
    torch.cuda.synchronize()
    atomic_json(OUT/'feature_portability_audit.json',dict(status='complete',resource=resource,model_loaded=False,
        tokens=count,selected=selected['predictor'],threshold=selected['threshold'],max_cpu_gpu_probability_difference=max_difference,
        routing_mismatches=mismatches,threshold_unchanged=True,decision_function_arguments=list(inspect.signature(OnlinePredictor.probability).parameters),
        features_source_sha256=sha256(ROOT/'paper_rq1_function_attention_sparsify/predictor.py'),
        interpretation='CPU evaluation vs actual GPU arithmetic. All discrepancies recorded, no adjusted threshold/tolerance. No future id/logits/labels supplied to either feature computation.'))
    if mismatches:raise ValueError('CPU/GPU routing mismatch needs explicit supplementary native-risk evaluation, not silent reuse')
    print('Feature portability',count,'tokens, max probability difference',max_difference,'routing mismatches',len(mismatches),flush=True)


def scorer():
    refs={'coco2014:1':['A red car is on the road.'],'coco2014:2':['A dog is beside a tree.']}
    predictions={k:v[0] for k,v in refs.items()};total,per=caption_scores(refs,predictions)
    if not np.isclose(total['CIDEr'],10.) or total['SPICE']!=1.:raise AssertionError('Official identical-caption fixture failed')
    paths=[ROOT/'paper_rq1/scorers.py',DEPS/'pycocoevalcap/cider/cider.py',DEPS/'pycocoevalcap/cider/cider_scorer.py',
        DEPS/'pycocoevalcap/tokenizer/ptbtokenizer.py',DEPS/'pycocoevalcap/spice/spice.py',DEPS/'pycocoevalcap/spice/spice-1.0.jar',
        DEPS/'pycocoevalcap/spice/lib/stanford-corenlp-3.6.0-models.jar']
    atomic_json(OUT/'scorer_validation.json',dict(passed=True,official_identical_caption_fixture=total,per_image=per,source_sha256={str(p):sha256(p) for p in paths}))


def data():
    path=Path('/data/lk/ATE-resources/paper_rq1/annotations/captions_val2014.json')
    historical=json.loads((ROOT/'results/paper_rq1_attention_time/data_manifest.json').read_text())
    if sha256(path)!=historical['annotation_sha256'][str(path)]:raise ValueError('Official frozen annotations changed')
    official=json.loads(path.read_text());images={str(r['id']):r for r in official['images']};refs={}
    for r in official['annotations']:refs.setdefault(str(r['image_id']),[]).append(r['caption'])
    manifest=json.loads((OUT/'sample_manifest.json').read_text());samples=manifest['development']+manifest['holdout']
    for s in samples:
        iid=s['image_identity'].split(':')[-1]
        if s['raw_row']!=images[iid] or s['answer']!=refs[iid]:raise ValueError('Annotation identity/answer mismatch')
        if sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Final image bytes changed')
        with Image.open(s['image_path']) as im:
            im=im.convert('RGB');rgb=digest([im.size,hashlib.sha256(im.tobytes()).hexdigest()])
        if rgb!=s['rgb_sha256']:raise ValueError('Final decoded RGB changed')
    atomic_json(OUT/'data_integrity_audit.json',dict(passed=True,images=len(samples),official_annotation_sha256=sha256(path),
        raw_image_rows_exact=True,official_reference_captions_exact=True,image_bytes_and_rgb_exact=True,
        sample_manifest_sha256=sha256(OUT/'sample_manifest.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','scorer','data']);a=p.parse_args()
    {'features':feature_portability,'scorer':scorer,'data':data}[a.stage]()
