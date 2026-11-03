"""Deployable text-only selectors: no question ID, image, answer or Gold input."""
import hashlib,math
import numpy as np
from semantic_visual_writers.groups import flatten,intact

POS=['NOUN','PROPN','ADJ','NUM','VERB','AUX','DET','ADP','PRON','ADV','PART','CCONJ','SCONJ','PUNCT']
DEPS=['ROOT','dobj','pobj','nsubj','attr','amod','prep','neg','det','conj','compound','nummod','advmod']
SPATIAL=set('left right above below behind front under over beside between next near middle center top bottom across inside outside on in to of'.split())
ATTR=set('color colour shape size material texture tall short small large big long wide narrow same different common taller shorter larger smaller black white red green blue yellow brown gray grey orange purple pink'.split())
LOGIC=set('not no neither nor both either all any other more less fewer than and or same different'.split())
WH=set('what which where who how many number'.split())
FEATURES=['bias','position','position2','tail','head','tokens','Q_tokens','core','noun','attribute','spatial','logic','wh','article','punctuation','relational_context','can_container','adjacent_content']+['pos_'+p for p in POS]+['dep_'+p for p in DEPS]

def features(groups,question):
    out=[];flat=flatten(groups);lo=flat[0];n=len(flat)
    for j,g in enumerate(groups):
        words=[w.lower() for w in g['words']];ww=set(words);pos=set(g.get('pos',[]));deps=set(g.get('dep',[]));context=set(w.lower() for h in groups[max(0,j-2):j+3] for w in h['words']);rel=bool(context&{'left','right','front','behind','above','below','next'});container='can' in ww and bool(context&{'garbage','soda','trash','beer','tin','metal'})
        x=[1,(np.mean(g['positions'])-lo)/max(n-1,1),((np.mean(g['positions'])-lo)/max(n-1,1))**2,float(j>=len(groups)-3),float(j<3),len(g['positions'])/4,n/32,float(g['core']),float(bool(pos&{'NOUN','PROPN'}) or container),float(bool(ww&ATTR)),float(bool(ww&SPATIAL)),float(bool(ww&LOGIC)),float(bool(ww&WH)),float(bool(ww&{'a','an','the'})),float(not words),float(rel),float(container),float(any(h.get('core') for h in groups[max(0,j-1):j]+groups[j+1:j+2]))]+[float(p in pos) for p in POS]+[float(d in deps) for d in DEPS]
        assert len(x)==len(FEATURES);out.append(x)
    return np.asarray(out,dtype=np.float64)

def exact_knapsack(groups,k,scores):
    """Maximize whole-group total score at exactly k tokenizer positions."""
    found={0:(0.,())}
    for i,g in enumerate(groups):
        size=len(g['positions'])
        for n,(value,chosen) in list(found.items()):
            if n+size>k:continue
            candidate=(value+float(scores[i])*size,chosen+(i,));previous=found.get(n+size)
            if previous is None or candidate[0]>previous[0]+1e-12 or abs(candidate[0]-previous[0])<=1e-12 and candidate[1]<previous[1]:found[n+size]=candidate
    assert k in found,'Historical feasible budget unexpectedly unreachable'
    answer=flatten([groups[i] for i in found[k][1]]);assert len(answer)==k and intact(answer,groups);return answer

def feasible_k(sp,fraction):
    q=sp['Q'];target=int(len(q)*fraction)
    return max(k for k in range(target+1) if intact(q[-k:] if k else [],sp['groups']))

ROUTE_FEATURES=['bias','Q_length','noun','attribute','spatial','logic','wh','article','relational_context','core','question_is','question_what','question_where','question_which','question_does','contains_or','contains_than','contains_not','distinct_content_fraction']
def route_features(sp):
    groups=[g for g in sp['groups'] if g['question']];x=features(groups,sp['question']);words=[w.lower() for g in groups for w in g['words']];first=words[0] if words else '';core=[w for g in groups if g['core'] for w in g['words']]
    return np.asarray([1,len(sp['Q'])/32]+[x[:,FEATURES.index(n)].mean() for n in ['noun','attribute','spatial','logic','wh','article','relational_context','core']]+[float(first==w) for w in ['is','what','where','which','does']]+[float(w in words) for w in ['or','than','not']]+[len(set(core))/max(len(core),1)],dtype=np.float64)

def select(sp,config):
    groups=[g for g in sp['groups'] if g['question']];x=features(groups,sp['question']);family=config['family'];fraction=config.get('fraction',.8)
    if family=='historical':return list(sp['historical_writers'][config['label']])
    if family=='structural_prune':
        removed=set(p for g in groups[:config.get('remove_first_groups',0)] for p in g['positions'])
        if config.get('remove_punctuation',False):removed.update(p for g in groups if not g['words'] for p in g['positions'])
        return [p for p in sp['Q'] if p not in removed]
    if family=='late_relay':
        if config['anchor']=='last_R_group':anchor=[sp['groups'][-1]]
        else:
            instruction=[g for g in sp['groups'][:-1] if not g['question'] and g['words'] and min(g['positions'])>sp['Q'][-1]]
            assert instruction;anchor=instruction[-config.get('instruction_groups',1):]
        p=flatten(anchor);assert p and min(p)>sp['Q'][-1] and set(p)<=set(sp['R'])
        b=config.get('question_budget',0);base=sp['historical_writers'][f'Semantic-first-Q{b}'] if b else []
        return sorted(set(base)|set(p))
    if family=='router':
        m=config['model'];z=(route_features(sp)-np.asarray(m['mean']))/np.asarray(m['scale']);z[0]=1;value=float(z@np.asarray(m['weights']));return select(sp,config['high'] if value>=config.get('threshold',0.) else config['low'])
    if family=='adaptive_historical':
        words=set(w.lower() for g in groups for w in g['words']);complexity=sum(w in words for w in ['left','right','behind','front','than','both','or','and','not'])+int(len(sp['Q'])>=18)
        budget=config['high'] if complexity>=config['threshold'] else config['low'];return list(sp['historical_writers'][f'Semantic-first-Q{budget}'])
    k=sp['k80'] if fraction==.8 else feasible_k(sp,fraction)
    if family=='role':
        weights=np.zeros(len(FEATURES));mapping=config['weights']
        for name,w in mapping.items():weights[FEATURES.index(name)]=w
        scores=x@weights
        if config.get('lexicographic_core',False):scores=scores/max(float(np.max(np.abs(scores))),1.)+(2*len(sp['Q'])+1)*x[:,FEATURES.index('core')]
    elif family=='risk':
        model=config['model'];xx=(x-np.asarray(model['mean']))/np.asarray(model['scale']);xx[:,0]=1
        scores=xx@np.asarray(model['weights'])
        # Data-driven deletion risk plus a preregistered weak lexical prior.
        scores+=config.get('prior',0.)*x[:,FEATURES.index('core')]
        if config.get('lexicographic_core',False):
            # Keep the original maximum lexical-core token coverage first;
            # causal scores only resolve feasible ties. The bounded residual
            # cannot compensate for losing even one core tokenizer position.
            scores=scores/max(float(np.max(np.abs(scores))),1.)+(2*len(sp['Q'])+1)*x[:,FEATURES.index('core')]
    else:raise ValueError(family)
    answer=exact_knapsack(groups,k,scores);assert set(answer)<=set(sp['Q']);return answer
