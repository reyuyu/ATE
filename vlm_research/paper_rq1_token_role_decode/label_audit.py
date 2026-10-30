"""Frozen rule/offset coverage audit; does not relabel or inspect interventions."""
import json
from collections import Counter
from benchmarks.dataset_registry import digest,freeze_json,sha256
from paper_rq1.analyze import write_csv
from .protocol import OUT,SEED
from .run import verify_targets


def main():
    verify_targets(); manifest=json.loads((OUT/'selected_token_manifest.json').read_text())
    ordered=sorted(manifest['images'].items(),key=lambda item:digest([SEED,'classification-audit20',item[0]]))[:20]
    rows=[]
    for image,data in ordered:
        for word in data['selected']:
            t=word['generation_step'];token=data['annotation']['tokens'][t]
            rows.append(dict(image_id=image,caption=data['caption'],word=word['word'],role=word['word_type'],
                t=t,token_id=word['token_id'],token_piece=token['token_piece'],char_start=word['char_start'],char_end=word['char_end'],
                char_substring=data['annotation']['text'][word['char_start']:word['char_end']],
                exact_single_token=word['exact_complete_single_token'],rule=word['rule_source']))
    write_csv(OUT/'classification_audit_sample.csv',rows)
    words=[w for data in manifest['images'].values() for w in data['annotation']['words']]
    selected=[w for data in manifest['images'].values() for w in data['selected']]
    if any(w['token_length']!=1 or not w['exact_complete_single_token'] or w['generation_step']<1 for w in selected):
        raise ValueError('Invalid main lexical alignment')
    freeze_json(OUT/'classification_audit.json',dict(
        frozen_manifest_sha256=sha256(OUT/'selected_token_manifest.json'),images=200,manual_review_input_images=20,
        words=len(words),word_roles=dict(Counter(w['word_type'] for w in words)),
        multi_token_words=sum(w['token_length']>1 for w in words),
        selected_words_by_role={role:dict(Counter(w['word'].lower() for w in selected if w['word_type']==role)) for role in ['Function','Visual','Relation']},
        main_selected_all_exact_single_complete_words=True,main_selected_all_decode_t_ge1=True,
        role_label_changes=0,intervention_outputs_used=False,
        limitation='Explicit conservative lexicon labels, not gold POS/grounding; excludes ambiguous/unknown and all multi-subword words from main. Audit examples allow contextual review but do not change frozen labels.'))


if __name__=='__main__':main()
