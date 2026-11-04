import re,random,collections
from .common import *
def firstword(q):return (re.findall('[a-z]+',q.lower()) or [''])[0]
def trigger(q,n,threshold):return firstword(q) in {'what','which'} or n>=threshold
def masks(spec,random_trigger):
 q=spec['Q'];a=spec['ASSISTANT'];both=sorted(q+a)
 assert len(a)==4 and not set(q)&set(a) and set(q+a)<=set(spec['R'])
 return dict(zip(CONDITIONS,[spec['R'],q,both if trigger(spec['question'],len(q),15) else q,both if trigger(spec['question'],len(q),18) else q,both if random_trigger else q]))
def random_labels(specs):
 strata=collections.defaultdict(list)
 for r in specs:strata[(len(r['Q']),len(r['R']))].append(r)
 labels={};records=[];rng=random.Random(SEED)
 for (n,rlen),rows in sorted(strata.items()):
  rows=sorted(rows,key=lambda r:r['question_id']);original=[trigger(r['question'],n,15) for r in rows];shuffled=original.copy();rng.shuffle(shuffled)
  for r,t in zip(rows,shuffled):labels[r['question_id']]=t
  records.append(dict(Q=n,R=rlen,n=len(rows),triggered=sum(original),changed=sum(a!=b for a,b in zip(original,shuffled))))
 base=sum(trigger(r['question'],len(r['Q']),15) for r in specs)
 assert sum(labels.values())==base
 original_mean=sum(4*trigger(r['question'],len(r['Q']),15)/len(r['R']) for r in specs)
 random_mean=sum(4*labels[r['question_id']]/len(r['R']) for r in specs)
 assert abs(original_mean-random_mean)<1e-10
 return labels,dict(seed=SEED,strata=records,trigger_count=base,changed_labels=sum(x['changed'] for x in records),exact_trigger_count=True,exact_added_tokens=True,exact_mean_and_pooled_R_budget=True,mean_added_R_ratio=original_mean/len(specs),rule='Permute Threshold-15 labels within exact native Question-token-count and R-token-count strata; stable qid order. Singleton/all-trigger strata remain unchanged.',labels=labels)
