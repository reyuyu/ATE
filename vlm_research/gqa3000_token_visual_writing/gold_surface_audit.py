"""Post-hoc CPU diagnostic of exact Gold spelling versus normalized GQA scoring.

No alternate endpoint, tokenization, model call, or primary hypothesis is added.
"""
import collections,csv,json
import numpy as np
from transformers import AutoTokenizer
from .common import *
from paper_rq1.scorers import gqa_prediction

def main():
    p=verify();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    counts={c:collections.Counter() for c in CONDITIONS};examples=[];n=0
    for part in read(OUT/'per_question_index.json')['partitions']:
        # The index stores output-relative file paths and exact hashes.
        path=OUT/part['path'];assert sha256(path)==part['sha256']
        with path.open() as f:
            for r in csv.DictReader(f):
                c=counts[r['condition']];ids=json.loads(r['token_ids']);gold=json.loads(r['gold_token_ids'])
                assert int(r['correct'])==int(gqa_prediction(r['text'])==r['gold'])
                c['questions']+=1;c['normalized_correct']+=int(r['correct']);c['exact_token_sequence_without_EOS_equal']+=int([v for v in ids if v!=2]==gold)
                c['correct_but_exact_Gold_token_sequence_differs']+=int(int(r['correct']) and [v for v in ids if v!=2]!=gold)
                c['text_casefold_equals_official_gold']+=int(r['text'].strip().casefold()==r['gold'].strip().casefold())
                if r['question_id']=='001020372' and r['condition'] in ['All-Writer','No-Writer','Semantic-k']:
                    with np.load(r['raw_logits_path']) as z:
                        first=z['first_logits'].astype(float);target=gold[0];chosen=int(first.argmax())
                        assert chosen==ids[0];rank=int((first>first[target]).sum())+1
                    examples.append(dict(question_id=r['question_id'],condition=r['condition'],official_gold=r['gold'],generated_text=r['text'],correct=int(r['correct']),
                        gold_token_ids=gold,gold_token_pieces=tok.convert_ids_to_tokens(gold),generated_token_ids=ids,generated_token_pieces=tok.convert_ids_to_tokens(ids),
                        exact_gold_first_token_rank=rank,exact_gold_first_token_logit=float(first[target]),greedy_first_token_logit=float(first[chosen]),gold_nll=float(r['gold_nll'])))
                n+=1
    assert n==132000 and all(c['questions']==3000 for c in counts.values()) and not torch.cuda.is_initialized()
    atomic_json(OUT/'gold_surface_form_audit.json',dict(passed=True,posthoc_diagnostic_only=True,primary_endpoints_unchanged=True,
        logical_answers=n,conditions={k:dict(v) for k,v in counts.items()},fixed_first_manifest_question_examples=examples,
        exact_gold_NLL_is_not_case_or_surface_invariant=True,source_sha256=sha256(__file__),binding=digest(p)))
    print('GOLD SURFACE AUDIT',n,json.dumps(examples,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
