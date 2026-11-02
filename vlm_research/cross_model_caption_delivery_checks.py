"""CPU-only delivery checks; does not alter the frozen GPU implementation."""
import csv
import hashlib
import json

import numpy as np

from .cross_model_caption_regions.common import *
from .cross_model_caption_regions.analyze import interval


def main():
    verify()
    samples=read(OUT/'sample_manifest.json')['samples']
    refs={s['question_id']:s['answer'] for s in samples}
    assert read(LLAVA/'sample_manifest.json')['samples']==samples
    p=read(LLAVA/'protocol.json')
    assert p['generation']['max_new_tokens']==96 and p['generation']['use_cache']
    with (LLAVA/'per_image.csv').open() as f: rows=list(csv.DictReader(f))
    checked=[]
    for key in ['Full','Decode None','Early','Middle','Late']:
        predictions={r['question_id']:r['caption'] for r in rows if r['condition']==key}
        assert len(predictions)==60
        binding=dict(refs=refs,predictions=predictions,scorer=p['scorer_source'])
        original=hashlib.sha256(json.dumps(binding,sort_keys=True,ensure_ascii=False,
                             allow_nan=False,separators=(',',':')).encode()).hexdigest()
        assert read(LLAVA/'scores'/(key+'.json'))['binding']==original
        checked.append(key)
    idx=np.random.default_rng(20261102).integers(0,60,(20000,60));a=np.linspace(0,1,60)
    assert interval(a-a,idx)==[0.,0.]
    assert np.allclose(interval(-a,idx),-np.array(interval(a,idx))[::-1])
    layout_summary={}
    for kind in MODELS:
        layout=list(read(OUT/f'{kind}_input_layouts.json').values())
        layout_summary[kind]=dict(prompt_length_min=min(x['input_length'] for x in layout),
            prompt_length_max=max(x['input_length'] for x in layout),
            image_token_counts=sorted({x['image_tokens'] for x in layout}))
    write(OUT/'delivery_checks.json',dict(passed=True,historical_conditions=checked,
        historical_canonical_json_digest_checked=True,historical_images_refs_predictions_scorer_bound=True,
        bootstrap_zero_and_sign_fixtures=True,bootstrap_repetitions=20000,
        input_layouts=layout_summary,source_sha256=sha(__file__)))
    if all((OUT/m/'complete.json').exists() for m in MODELS):
        import sys
        sys.path.insert(0,'/data/lk/ATE-resources/paper_rq1/metric_deps')
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        captions={'references':{s['question_id']:s['answer'] for s in samples}}
        for kind in MODELS:
            captions[kind]={s['question_id']:[read(OUT/kind/'checkpoints'/
                s['question_id'].replace(':','_')/'Full.json')['generation']['text']] for s in samples}
        pred={r['question_id']:[r['caption']] for r in rows if r['condition']=='Full'}
        captions['llava']=pred
        lengths={}
        for kind,values in captions.items():
            tokenized=PTBTokenizer().tokenize({q:[dict(caption=t) for t in strings] for q,strings in values.items()})
            counts=[len(t.split()) for strings in tokenized.values() for t in strings]
            lengths[kind]=dict(n=len(counts),ptb_words_mean=float(np.mean(counts)),
                ptb_words_median=float(np.median(counts)),min=min(counts),max=max(counts))
        write(OUT/'native_caption_length_review.json',dict(official_PTB_tokenizer=True,
            length_summary=lengths,cider_native_sigma=6.0,
            interpretation='Descriptive length comparison only. Native CIDEr includes its unchanged Gaussian length penalty; do not attribute all score differences to length.',
            source_sha256=sha(__file__)))
    comparison_path=OUT/'paired_comparisons.csv'
    if comparison_path.exists():
        with comparison_path.open() as f: comparisons=list(csv.DictReader(f))
        for row in comparisons:
            c=conditions(row['model'])
            a=len(c[row['condition']])
            b=(len(c['Middle']) if row['reference']=='Matched Random mean' else len(c[row['reference']]))
            row['count_matched']=str(a==b)
            row['retained_count_condition']=a
            row['retained_count_reference']=b
        with comparison_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
            writer.writeheader();writer.writerows(comparisons)
        write(OUT/'comparison_metadata_review.json',dict(passed=True,rows=len(comparisons),
            corrected='Explicit retained counts and count_matched for every pair, including comparisons to Full.',
            scores_intervals_and_predeclared_pairs_unchanged=True,source_sha256=sha(__file__)))
    print('Historical reuse bindings, paired-bootstrap fixtures, and input layout coverage passed.')


if __name__=='__main__':main()
