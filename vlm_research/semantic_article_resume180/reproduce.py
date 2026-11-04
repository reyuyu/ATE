"""Rebuild the frozen winner's receivers on CPU; never load model or outcomes.

Run with the inference PYTHONPATH: python -m semantic_article_resume180.reproduce
"""
from .common import P,read,spec,sha,digest,writecsv
from .pipeline import load

def main():
    f=read(P/'winner_freeze.json');d,m=load(f['candidate_id']);out=[]
    for r in f['positions']:
        q=r['question_id'];w=m.select(spec(q),d['config']);assert w==r['writers'];out.append(dict(question_id=q,candidate=f['candidate_id'],writers=w,writers_sha256=digest(w)))
    assert len(out)==3000
    print('PASS: all 3000 frozen receiver sets reproduced exactly; source SHA256='+d['source_sha256'])

if __name__=='__main__':main()
