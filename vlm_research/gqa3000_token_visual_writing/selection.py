"""Input-only whole-word budgets. Never receives images, Gold or model outputs."""
import hashlib,random
from semantic_visual_writers.groups import token_groups,select,subsets,flatten,intact,STOP
from .common import SEEDS,BUDGETS,CONDITIONS,digest

def priority_random(groups,seed,question,salt=''):
    ordered=list(groups);local=int(hashlib.sha256(f'{seed}|{question}{salt}'.encode()).hexdigest(),16)
    random.Random(local).shuffle(ordered);return ordered
def semantic_first(groups,k):
    # Maximize core tokenizer-token count subject to exact whole-group k;
    # ties prefer earlier whole groups in original question order.
    found={0:(0,())}
    for i,g in enumerate(groups):
        size=len(g['positions']);value=size if g['core'] else 0
        for n,(score,chosen) in list(found.items()):
            if n+size>k:continue
            cand=(score+value,chosen+(i,));old=found.get(n+size)
            if old is None or cand[0]>old[0] or cand[0]==old[0] and cand[1]<old[1]:found[n+size]=cand
    assert k in found
    return flatten([groups[i] for i in found[k][1]])
def stratum_counts(positions,qpos):
    index={p:i for i,p in enumerate(qpos)};counts=[0,0,0]
    for p in positions:counts[min(2,3*index[p]//len(qpos))]+=1
    return tuple(counts)
def stratified(groups,k,semantic,qpos,seed,question):
    target=stratum_counts(semantic,qpos);found={(0,0,0):[]}
    for g in priority_random(groups,seed,question,'|strata'):
        v=stratum_counts(g['positions'],qpos)
        for counts,chosen in list(found.items()):
            new=tuple(a+b for a,b in zip(counts,v))
            if all(a<=b for a,b in zip(new,target)):found.setdefault(new,chosen+[g])
    if target not in found:return None,target
    result=flatten(found[target]);assert len(result)==k and stratum_counts(result,qpos)==target
    return result,target

def make_selection(tokenizer,question,chat,prompt_ids,image,relay):
    groups,qpos,words=token_groups(tokenizer,question,chat,prompt_ids,image,relay)
    natural=select(groups,qpos,relay,question)  # Unmodified historical selector.
    qgroups=[g for g in groups if g['question']];assert flatten(qgroups)==qpos
    k=natural['k'];legacy=natural['writers'];writers={n:legacy[n] for n in ['All-Writer','No-Writer','Semantic-k','Question-Last-k','R-Last-k','Drop-Semantic']}
    for seed in SEEDS:
        writers[f'Question-Random-k-{seed}']=flatten(subsets(priority_random(qgroups,seed,question))[k])
        writers[f'R-Random-k-{seed}']=legacy[f'Random-k-{seed}']
    strata={}
    for seed in SEEDS[:2]:
        positions,target=stratified(qgroups,k,legacy['Semantic-k'],qpos,seed,question)
        assert positions is not None,'An exact semantic-matched stratum set should itself be feasible'
        writers[f'Question-Stratified-k-{seed}']=positions;strata[str(seed)]=dict(feasible=True,target=list(target),actual=list(stratum_counts(positions,qpos)))
    # All strategies admit exactly the same k: use the largest whole-question
    # suffix budget <= floor(f * actual Question token count). Other policies
    # can realize this suffix as a whole-group subset, proving common feasibility.
    feasible=[n for n in range(len(qpos)+1) if intact(qpos[-n:] if n else [],groups)]
    qr={seed:subsets(priority_random(qgroups,seed,question)) for seed in SEEDS[:2]}
    rr={seed:subsets(priority_random(groups,seed,question)) for seed in SEEDS[:2]}
    budgets={}
    for b in BUDGETS:
        target=b*len(qpos)//100;actual=max(n for n in feasible if n<=target)
        writers[f'Semantic-first-Q{b}']=semantic_first(qgroups,actual)
        writers[f'Question-Last-Q{b}']=qpos[-actual:] if actual else []
        for seed in SEEDS[:2]:
            writers[f'Question-Random-{seed}-Q{b}']=flatten(qr[seed][actual])
            writers[f'R-Random-{seed}-Q{b}']=flatten(rr[seed][actual])
        budgets[str(b)]=dict(question_tokens=len(qpos),nominal_percent=b,target_floor_tokens=target,actual_k=actual,
            token_shortfall=target-actual,actual_question_fraction=actual/len(qpos),exact_target_matched=actual==target,feasible=True)
    assert set(writers)==set(CONDITIONS) and len(CONDITIONS)==44
    for name,positions in writers.items():
        assert positions==sorted(set(positions)) and set(positions)<=set(relay) and intact(positions,groups),(name,positions)
        if name.startswith(('Question-','Semantic-first')):assert set(positions)<=set(qpos)
        if name.endswith(tuple('-Q'+str(b) for b in BUDGETS)):
            b=name.rsplit('-Q',1)[1];assert len(positions)==budgets[b]['actual_k']
    for b in BUDGETS:assert len(writers[f'Semantic-first-Q{b}'])<=len(qpos)
    assert writers['Semantic-first-Q100']==writers['Question-Last-Q100']==qpos and qpos!=relay
    aliases={};unique={}
    for name in CONDITIONS:
        uid='set_'+digest(writers[name])[:20];aliases[name]=uid
        if uid in unique:assert unique[uid]['positions']==writers[name]
        else:unique[uid]=dict(positions=writers[name],labels=[])
        unique[uid]['labels'].append(name)
    policies=['Semantic-first','Question-Last']+[f'{source}-Random-{seed}' for source in ['Question','R'] for seed in SEEDS[:2]]
    nested={name:all(set(writers[f'{name}-Q{a}'])<=set(writers[f'{name}-Q{b}']) for a,b in zip(BUDGETS,BUDGETS[1:])) for name in policies}
    return dict(question=question,groups=groups,question_positions=qpos,relay_positions=relay,words=words,natural=natural,
        writers=writers,aliases=aliases,unique=unique,budgets=budgets,nested=nested,stratification=strata,
        prompt_ids_sha256=digest(prompt_ids),selection_binding=digest(writers),legacy_selection_binding=digest(natural))
