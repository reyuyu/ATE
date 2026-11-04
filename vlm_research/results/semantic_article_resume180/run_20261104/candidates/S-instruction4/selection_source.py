"""Preserve original S; add complete groups by input structure only."""
import math
from semantic_visual_writers.groups import flatten,intact
from gqa5h_receiver_search.selection import exact_knapsack,features,FEATURES

LOGIC=set('not no neither nor both either all any other more less fewer than and or same different'.split())
REL=set('left right above below behind front under over beside between next near middle center top bottom across inside outside on in to of'.split())

def selected(sp,c):
    groups=sp['groups'];Q=sp['Q'];S=set(sp['historical_writers']['Semantic-k']);w=set(S);last=groups[-1];assert last['words']==['ASSISTANT'] and len(last['positions'])==4
    qgroups=[g for g in groups if g['question']];after=[g for g in groups[:-1] if min(g['positions'])>Q[-1]];iw=[g for g in after if g['words']];punct=[g for g in after if not g['words']]
    for name in c.get('anchors',[]):
        if name=='assistant':gg=[last]
        elif name=='qtail1':gg=qgroups[-1:]
        elif name=='qtail2':gg=qgroups[-2:]
        elif name=='qtail3':gg=qgroups[-3:]
        elif name=='postQ':gg=after[:1]
        elif name=='instruction_period':gg=punct[-1:]
        elif name=='instruction_word':gg=iw[-1:]
        elif name=='instruction_pair':gg=iw[-2:]
        elif name=='instruction4':gg=iw[-4:];assert len(flatten(gg))==4
        else:raise ValueError(name)
        w.update(flatten(gg))
    for category in c.get('categories',[]):
        for g in qgroups:
            words=set(v.lower() for v in g['words']);pos=set(g.get('pos',[]));dep=set(g.get('dep',[]))
            hit=category=='allQ' or category=='article' and bool(words&{'a','an','the'}) and 'det' in dep or category=='noun' and bool(pos&{'NOUN','PROPN'}) or category=='logic' and bool(words&LOGIC) or category=='relation' and bool(words&REL)
            if hit:w.update(g['positions'])
    available=[g for g in qgroups if not set(g['positions'])<=w];added=0;target=None
    if 'extra' in c or 'Rcap' in c:
        target=c['extra'] if 'extra' in c else max(0,int(c['Rcap']*len(sp['R']))-len(w))
        maxk=min(target,len(flatten(available)));possible={0}
        for g in available:possible.update(k+len(g['positions']) for k in list(possible) if k+len(g['positions'])<=maxk)
        added=max(possible);order=c.get('order','late')
        scores=[(i if order=='late' else -i) if order!='role' else (3*bool(set(g.get('pos',[]))&{'NOUN','PROPN'})+2*bool(set(v.lower() for v in g['words'])&(LOGIC|REL))+i/100) for i,g in enumerate(available)]
        if added:w.update(exact_knapsack(available,added,scores))
    answer=sorted(w);assert S<=w and w<=set(sp['R']) and intact(answer,groups)
    return answer,dict(original_k=len(S),final_k=len(answer),R=len(sp['R']),extra_target=target,question_added_by_fill=added,exact_extra_match=target is None or added==target,selected_groups=[dict(words=g['words'],positions=g['positions']) for g in groups if set(g['positions'])<=w])

def select(sp,c):return selected(sp,c)[0]
