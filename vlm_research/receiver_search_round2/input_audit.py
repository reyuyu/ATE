"""Actual native IDs and offsets for structural receivers, CPU only."""
from .common import *

def main():
    counts={};examples=[];splits=read(P/'split_manifest.json');example_ids=set(splits['dev_stages']['60'][:6])
    from .cpu_cache import enable
    enable()
    for q in splits['metadata']:
        s=spec(q);Q=s['Q'];groups=s['groups'];after=[g for g in groups[:-1] if min(g['positions'])>Q[-1]];punct=[g for g in after if not g['words']];word=[g for g in after if g['words']]
        anchors=dict(assistant=groups[-1],instruction_period=punct[-1],postQ=after[0],qtail1=[g for g in groups if g['question']][-1],instruction_word=word[-1]);assert anchors['assistant']['decoded']=='ASSISTANT' and anchors['instruction_period']['decoded']=='.' and anchors['postQ']['decoded']=='\n'
        ids=sample(q)['input_mapping']['prompt_ids']
        for name,g in anchors.items():
            counts[(name,g['decoded'],len(g['positions']))]=counts.get((name,g['decoded'],len(g['positions'])),0)+1
            if q in example_ids:examples.append(dict(question_id=q,anchor=name,decoded=g['decoded'],words=g['words'],positions=g['positions'],token_ids=[ids[p] for p in g['positions']],inside_R=set(g['positions'])<=set(s['R']),Pminus1=len(ids)-1))
    atomic_json(P/'native_anchor_audit.json',dict(passed=True,count_summary=[dict(anchor=k[0],decoded=k[1],subwords=k[2],n=v) for k,v in counts.items()],examples=examples,Prompt_modified=False,UTC=utc()));print(json.dumps(dict(passed=True,counts=[dict(anchor=k[0],decoded=k[1],subwords=k[2],n=v) for k,v in counts.items()])),flush=True)

if __name__=='__main__':main()
