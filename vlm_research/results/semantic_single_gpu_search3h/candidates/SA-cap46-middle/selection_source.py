"""Input-only augmentation of the exact original S, with whole-group budgets."""
import math
from semantic_visual_writers.groups import flatten,intact
from gqa5h_receiver_search.selection import exact_knapsack
from receiver_search_round2.selection import selected as original

REL=set('left right above below behind front under over beside between next near middle center top bottom across inside outside on in to of'.split())
LOG=set('not no neither nor both either all any other more less fewer than and or same different'.split())

def select(sp,c):
    if 'old' in c:return original(sp,c['old'])[0]
    S=set(sp['historical_writers']['Semantic-k']);w=set(S);groups=sp['groups'];Q=[g for g in groups if g['question']];assistant=groups[-1]
    assert assistant['words']==['ASSISTANT'] and len(assistant['positions'])==4
    if c.get('assistant',True):w.update(assistant['positions'])
    if c.get('anchor')=='tail':w.update(Q[-1]['positions'])
    if c.get('anchor')=='space':w.update(groups[0]['positions'])
    # Completion decisions depend only on native question structure, never task labels or outputs.
    mode=c.get('mode','cap');words=[s.lower() for g in Q for s in g['words']];hasrel=bool(set(words)&REL)
    if mode=='conditional':
        rule=c['rule'];hit=(hasrel if rule=='relation' else words[0] in ['what','which'] if rule=='whatwhich' else len(sp['Q'])>=c.get('length',15) if rule=='long' else len(sp['Q'])<=c.get('length',15) if rule=='short' else any(set(g.get('pos',[]))&{'AUX','VERB'} for g in Q) if rule=='verb' else words[0] in ['is','are','does','do'])
        if hit:w.update(flatten(Q))
    else:
        target=max(0,math.floor(c.get('cap',.49)*len(sp['R']))-len(w));available=[g for g in Q if not set(g['positions'])<=w];possible={0}
        for g in available:possible.update(k+len(g['positions']) for k in list(possible) if k+len(g['positions'])<=target)
        k=max(possible);order=c.get('order','late');scores=[]
        for i,g in enumerate(available):
            gs={x.lower() for x in g['words']};pos=set(g.get('pos',[]));dep=set(g.get('dep',[]));location=min(g['positions'])/max(sp['Q'])
            base={'late':location,'early':-location,'relation':5*bool(gs&REL)+2*bool(gs&LOG)+location,'logic':5*bool(gs&LOG)+2*bool(gs&REL)+location,'noun':5*bool(pos&{'NOUN','PROPN'})+location,'function':5*bool(pos&{'DET','ADP','AUX','PRON','SCONJ'})+location,'attribute':5*bool(pos&{'ADJ','NUM','ADV'})+location,'syntax':5*bool(dep&{'ROOT','nsubj','dobj','pobj','amod','nummod'})+location,'middle':-abs(location-.5)}[order];scores.append(base)
        if k:w.update(exact_knapsack(available,k,scores))
    out=sorted(w);assert S<=w<=set(sp['R']) and intact(out,groups);return out
