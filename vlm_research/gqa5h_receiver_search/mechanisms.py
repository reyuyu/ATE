"""Aggregate conditional interventions without treating them as token necessity."""
import collections
from .common import *
from .analyze import rows,bootstrap,exact_p

def main():
    rr=rows(P/'jobs/r02-loo-dev96/results.csv');words=collections.defaultdict(list);kinds=collections.defaultdict(list)
    for r in rr:
        g=json.loads(r['removed_group']);label=int(r['causal_harm_label']);name=' '.join(g['words']).lower() or '[punctuation]';words[name].append(label)
        kinds['first_question_group' if int(r['group_index'])==0 else 'other_groups'].append(label)
    summary=[dict(word=w,interventions=len(v),harm=sum(x==1 for x in v),fix=sum(x==-1 for x in v),unchanged=sum(x==0 for x in v),interpretation='Conditional removal from Question-Only; repeated questions are not independent samples.') for w,v in sorted(words.items())];writecsv(P/'loo_word_summary.csv',summary)
    writecsv(P/'loo_position_summary.csv',[dict(group=k,n=len(v),harm=sum(x==1 for x in v),fix=sum(x==-1 for x in v)) for k,v in kinds.items()])
    a={r['question_id']:r for r in rows(P/'jobs/r12-late-relay-dev180/results.csv') if r['candidate_id']=='q80-late-R-group'};b={r['question_id']:r for r in rows(P/'jobs/r14-late-relay-followup-dev180/results.csv') if r['candidate_id']=='q80-instruction-last4'};assert set(a)==set(b) and len(a)==180
    compare=[]
    for q in sorted(a):
        x,y=a[q],b[q];assert x['k']==y['k'] and x['R_ratio']==y['R_ratio'];compare.append(dict(question_id=q,image_sha256=x['image_sha256'],task=x['task'],question=x['question'],gold=x['gold'],late_ASSISTANT_answer=x['answer'],instruction4_answer=y['answer'],late_ASSISTANT_correct=int(x['correct']),instruction4_correct=int(y['correct']),k=x['k'],R_ratio=x['R_ratio'],late_positions=x['writers'],control_positions=y['writers']))
    writecsv(P/'matched_template_budget_control.csv',compare);d=np.array([r['late_ASSISTANT_correct']-r['instruction4_correct'] for r in compare]);lo,hi=bootstrap(d,[r['image_sha256'] for r in compare]);atomic_json(P/'mechanism_summary.json',dict(LOO_interventions=len(rr),LOO_harm=sum(int(r['causal_harm_label'])==1 for r in rr),LOO_fix=sum(int(r['causal_harm_label'])==-1 for r in rr),matched_template_control=dict(n=180,exact_actual_per_question_k=True,late_ASSISTANT_correct=int(sum(int(x['correct']) for x in a.values())),instruction4_correct=int(sum(int(x['correct']) for x in b.values())),delta_pp=float(100*d.mean()),CI95=[lo,hi],harm=int(sum(d==-1)),fix=int(sum(d==1)),exact_p=exact_p(int(sum(d==-1)),int(sum(d==1))),inference='DEV post-design exploratory control only; 4-token ASSISTANT versus four original instruction words, no Prompt edit, no proof of a unique visual store.'),scope='Only interventions at the frozen conditions and these DEV questions; no global per-token necessity claim.'));print(json.dumps(read(P/'mechanism_summary.json')),flush=True)

if __name__=='__main__':main()
