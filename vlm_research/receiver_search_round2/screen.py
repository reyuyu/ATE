"""Freeze S-preserving hypotheses before GPU outputs; pre-audit all budgets."""
from .common import *
from .selection import selected
from .experiments import register,candidate_job

def configurations():
    cfg={'S':{},'S-assistant':dict(anchors=['assistant'])}
    for a in ['qtail1','qtail2','qtail3','postQ','instruction_period','instruction_word','instruction_pair','instruction4']:cfg['S-'+a]=dict(anchors=[a])
    for c in ['article','noun','logic','relation']:
        cfg['S-'+c]=dict(categories=[c]);cfg['S-assistant-'+c]=dict(anchors=['assistant'],categories=[c])
    cfg['S-noun-article']=dict(categories=['noun','article'])
    cfg['S-assistant-noun-article']=dict(anchors=['assistant'],categories=['noun','article'])
    cfg['S-assistant-qtail1']=dict(anchors=['assistant','qtail1'])
    cfg['S-assistant-qtail2']=dict(anchors=['assistant','qtail2'])
    cfg['S-article-qtail1']=dict(anchors=['qtail1'],categories=['article'])
    cfg['S-article-period']=dict(anchors=['instruction_period'],categories=['article'])
    cfg['S-question-extra4']=dict(extra=4,order='late')
    cfg['S-question-extra4-role']=dict(extra=4,order='role')
    for a in ['assistant','instruction_period']:
        for cap in [.46,.49]:cfg[f'S-{a}-fill{round(cap*100)}']=dict(anchors=[a],Rcap=cap,order='late')
    cfg['S-assistant-fill49-early']=dict(anchors=['assistant'],Rcap=.49,order='early')
    cfg['S-assistant-fill49-role']=dict(anchors=['assistant'],Rcap=.49,order='role')
    cfg['S-allQ-period']=dict(anchors=['instruction_period'],categories=['allQ'])
    cfg['S-allQ-word']=dict(anchors=['instruction_word'],categories=['allQ'])
    return {cid:dict(family='S-augmentation',**c) for cid,c in cfg.items()}

def main():
    guard(reserve=1800);assert (P/'CPU_GATE.json').exists();specs={q:spec(q) for q in read(P/'split_manifest.json')['metadata']};audit=[];accepted=[]
    for cid,c in configurations().items():
        rr=[]
        for q,s in specs.items():
            w,extra=selected(s,c);rr.append(dict(question_id=q,R_ratio=len(w)/len(s['R']),k=len(w),extra_target=extra['extra_target'],matched=extra['exact_extra_match']))
        mean=float(np.mean([r['R_ratio'] for r in rr]));passed=mean<.5;audit.append(dict(candidate_id=cid,mean_R_ratio_3000=mean,admitted=passed,extra_budget_unmatched=sum(not r['matched'] for r in rr),source_sha256=sha(Path(__file__).with_name('selection.py')),rule=json.dumps(c),selection_uses_only_inputs=True))
        if passed:register(cid,c,'Preserve exact original S; add frozen complete existing groups or input-only Question completion. No Prompt edits.');accepted.append(cid)
    writecsv(P/'pre_GPU_budget_audit.csv',audit);atomic_json(P/'screen_plan.json',dict(candidates=accepted,rejected_over_budget=[r for r in audit if not r['admitted']],stage='DEV60',new_data=False,all3000_previously_seen=True,UTC=utc(),S_always_preserved=True,budget_audit_before_outputs=True));candidate_job('s01-dev60',accepted,read(P/'split_manifest.json')['dev_stages']['60']);print(json.dumps(dict(candidates=len(accepted),rejected=len(audit)-len(accepted))),flush=True)

if __name__=='__main__':main()
