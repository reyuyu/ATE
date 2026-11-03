"""Audit historical baselines on the exact authorized split, never cross-split totals."""
import argparse
from .common import *
def main(split):
    start=time.monotonic();ids=read(P/'split_manifest.json')['ids'][split];h=History();rows=[]
    for q in ids:
        authorized(q);s=sample(q);sp=spec(q)
        for c in BASELINES:
            r=h.get(q,c);g=r['generation'];rows.append(dict(split=split,question_id=q,task=s['task'],image_sha256=s['image_sha256'],condition=c,correct=g['correct'],answer=g['text'],gold=gold(q),token_ids=json.dumps(g['token_ids']),writers=json.dumps(r['writers']),k=len(r['writers']),R=len(sp['R']),R_ratio=len(r['writers'])/len(sp['R']),source_path=r['source']['path'],source_sha256=r['source']['sha256'],raw_sha256=r['source']['raw_sha256']))
    h.close();d=P/'baselines'/split;d.mkdir(parents=True,exist_ok=True);writecsv(d/'per_question.csv',rows);atomic_json(d/'history_audit.json',h.files)
    summary=[]
    for c in BASELINES:
        rr=[r for r in rows if r['condition']==c];summary.append(dict(condition=c,n=len(rr),correct=sum(r['correct'] for r in rr),ACC=100*np.mean([r['correct'] for r in rr]),mean_R_ratio=np.mean([r['R_ratio'] for r in rr])))
    writecsv(d/'summary.csv',summary);event('BASELINE_AUDITED',split=split,seconds=time.monotonic()-start,summary=summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['DEV','TUNE','FINAL'],required=True);main(p.parse_args().split)
