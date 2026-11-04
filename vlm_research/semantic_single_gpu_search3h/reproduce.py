"""CPU-only exact reproduction of the final frozen receiver manifest."""
from .common import *
from .pipeline import load

if __name__=='__main__':
    f=read(P/'winner_freeze.json');d,m=load(f['candidate_id'])
    for r in f['positions']:assert m.select(spec(r['question_id']),d['config'])==r['writers']
    assert len(f['positions'])==3000
    print('PASS: all 3000 receiver sets exactly reproduced without reading answers.')
