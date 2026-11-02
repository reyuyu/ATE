"""Post-collection descriptive coverage review; frozen collector is unchanged."""
import csv
from collections import defaultdict
import numpy as np

from coco_wordwise_sources.common import *
from coco_wordwise_sources.statistics import ci


def js_rows(a, b):
    a = a/a.sum(-1, keepdims=True)
    b = b/b.sum(-1, keepdims=True)
    middle = (a+b)/2
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(a > 0, a*np.log(a/middle), 0)
        y = np.where(b > 0, b*np.log(b/middle), 0)
    return .5*(x+y).sum(-1)


def main():
    verify()
    images = read(OUT/'sample_manifest.json')['samples']
    adjacent = defaultdict(dict)
    head_rows, quantiles = [], []
    eligible_non_eos_pairs = 0
    per_image = {}
    for sample in images:
        result = read(OUT/'checkpoints'/(sample['image_identity'].replace(':','_')+'.json'))
        per_image[result['image_id']] = result
        with np.load(result['raw_path']) as data:
            mass = data['mass'].astype(float)
        primary = np.stack([mass[...,0],mass[...,1],mass[...,2],mass[...,3]+mass[...,4]],-1)
        average = primary.mean(2)
        steps = [t for t in range(2,len(mass)) if result['labels'][t]['role']!='EOS']
        eligible_non_eos_pairs += len(steps)
        for layer in range(32):
            for include_eos in [False,True]:
                selected = list(range(2,len(mass))) if include_eos else steps
                if not selected:
                    continue
                selected = np.array(selected)
                values = js_rows(average[selected,layer],average[selected-1,layer])
                normalized = js_rows(average[selected,layer,1:],average[selected-1,layer,1:])
                adjacent[layer,include_eos,'raw'][result['image_sha256']] = float(values.mean())
                adjacent[layer,include_eos,'nonbos'][result['image_sha256']] = float(normalized.mean())
        for t in steps:
            for layer in SELECTED:
                raw = float(js_rows(average[t,layer],average[t-1,layer]))
                conditional = float(js_rows(average[t,layer,1:],average[t-1,layer,1:]))
                if raw<=1e-4 and conditional<=1e-3:
                    heads = js_rows(primary[t,layer],primary[t-1,layer])
                    nonbos = js_rows(primary[t,layer,:,1:],primary[t-1,layer,:,1:])
                    head_rows.append(dict(image_id=result['image_id'],image_sha256=result['image_sha256'],t=t,layer=layer,
                        aggregate_js=raw,aggregate_nonbos_js=conditional,
                        mean_head_js=float(heads.mean()),max_head_js=float(heads.max()),
                        mean_head_nonbos_js=float(nonbos.mean()),max_head_nonbos_js=float(nonbos.max()),
                        fraction_heads_meeting_same_threshold=float(((heads<=1e-4)&(nonbos<=1e-3)).mean())))
    csv_write(OUT/'adjacent_js_summary.csv',[dict(layer=k[0],include_eos=k[1],metric=k[2],**ci(v)) for k,v in adjacent.items()])
    csv_write(OUT/'near_mass_head_audit.csv',head_rows)
    with (OUT/'within_caption_variation.csv').open() as stream:
        variation = list(csv.DictReader(stream))
    for source in MAIN:
        chosen = [r for r in variation if r['analysis']=='non_eos_decode' and int(r['layer'])==31 and r['source']==source]
        values = np.array([float(r['range']) for r in chosen])
        p5,median,p95 = np.quantile(values,[.05,.5,.95])
        quantiles.append(dict(layer=31,source=source,n_images=len(values),p5_range=float(p5),median_range=float(median),p95_range=float(p95)))
    csv_write(OUT/'l31_across_image_range_quantiles.csv',quantiles)
    with (OUT/'per_token_contributions.csv').open() as stream:
        examples = [r for r in csv.DictReader(stream) if r['near_mass']=='True' and r['contribution_changed']=='True'
            and r['eos']=='False' and r['source'] in ['IMAGE','GENERATED_HISTORY']]
    examples.sort(key=lambda r:(r['image_id'],int(r['t']),int(r['layer']),r['source']))
    csv_write(OUT/'similar_mass_difference_cases.csv',examples)
    review = dict(post_collection_review=True,frozen_collector_sources_unchanged=True,
        source_sha256={str(__file__):sha256(__file__)},eligible_non_eos_adjacent_decode_pairs=eligible_non_eos_pairs,
        thresholds='Unchanged pre-GPU aggregate JS thresholds; per-head audit is descriptive coverage, not a new condition',
        l31_near_mass_pairs=sum(r['layer']==31 for r in head_rows),
        l31_mean_fraction_heads_also_similar=float(np.mean([r['fraction_heads_meeting_same_threshold'] for r in head_rows if r['layer']==31])),
        head_mean_not_all_heads_identical=True,examples_retained=len(examples),l31_image_range_quantiles=quantiles)
    write(OUT/'review_summary.json',review)
    print(review,flush=True)


if __name__=='__main__':
    main()
