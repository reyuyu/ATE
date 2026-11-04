"""Protect original S; complete Q and add low-cost native relay groups."""
from semantic_visual_writers.groups import flatten,intact
from receiver_search_round2.selection import selected as original

def select(sp,c):
    if 'old' in c:return original(sp,c['old'])[0]
    Q=[g for g in sp['groups'] if g['question']];S=set(sp['historical_writers']['Semantic-k']);words=[s.lower() for g in Q for s in g['words']];after=[g for g in sp['groups'][:-1] if min(g['positions'])>max(sp['Q'])];iw=[g for g in after if g['words']];punct=[g for g in after if not g['words']];w=set(S)
    mode=c.get('base','Q');hit=True
    if mode=='conditional':
        rule=c['rule'];hit=(words[0] in ['what','which'] if rule=='whatwhich' else len(sp['Q'])>=20 if rule=='long20' else len(sp['Q'])<=10 if rule=='short10' else words[0] not in ['is','are','do','does'])
        if not hit:w.update(sp['groups'][-1]['positions'])
    if hit:w.update(sp['Q'])
    for anchor in c.get('anchors',[]):
        if anchor=='period':gg=punct[-1:]
        elif anchor=='postQ':gg=after[:1]
        elif anchor=='space':gg=sp['groups'][:1]
        elif anchor=='phrase':gg=iw[-1:]
        elif anchor=='or':gg=iw[-2:-1]
        elif anchor=='word':gg=iw[-3:-2]
        elif anchor=='single':gg=iw[-4:-3]
        else:raise ValueError(anchor)
        if hit:w.update(flatten(gg))
    out=sorted(w);assert S<=w<=set(sp['R']) and intact(out,sp['groups']);return out
