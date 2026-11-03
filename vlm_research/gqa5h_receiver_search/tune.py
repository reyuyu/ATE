"""Close DEV search and freeze the complete TUNE comparison before outcomes."""
from .common import *
from .experiments import register,candidate_job

def main():
    assert not (P/'tune_shortlist.json').exists()
    assert not (P/'baselines/TUNE').exists()
    candidates=['q80-late-R-group','natural-S-late-R-group','adaptive-60-80-budget30','adaptive-80-100-budget35']
    for cid,label in [('baseline-q100','Semantic-first-Q100'),('baseline-sem-q80','Semantic-first-Q80'),('baseline-sem-k','Semantic-k')]:
        register(cid,dict(family='historical',label=label),'Audited baseline eligible to win; all original input-only token rules, no new generation on TUNE.')
    atomic_json(P/'tune_shortlist.json',dict(job_id='r18-frozen-shortlist-tune240',new_candidates=candidates,created_utc=utc(),created_monotonic=time.monotonic(),selection_rule='Maximize correct answers under mean R strictly below 50%; ties lower R, simpler family, lexical ID. Historical Q100/Q80/S also eligible.',reason='One mature simple <50% candidate, one mature <42% candidate, and DEV-trained routers covering the original <=30% and <=35% frontiers. Four methods fixed before any TUNE outcome. TUNE is selection data, not confirmation.',dropped=dict(q85='Same 168/180 as Q80+late anchor, higher budget; no benefit on the first screen.',q50='320/360 below Q80 and the <=30% DEV-trained router.',q60='324/360 below natural S+late 325; additional lower-budget performance remains in DEV registry.',risk='Five-fold LOO and swap models do not generalize on DEV; preserve negative evidence.',fixed_roles='Expanded DEV360: 317-320 vs Q80 322.'),candidate_definitions={cid:sha(P/'candidates'/cid/'definition.json') for cid in candidates},stats='Single FINAL Winner only; four exact paired comparisons with Holm correction, 20000 image-cluster bootstrap pointwise 95% intervals; no noninferiority claim.',no_final_outcomes_opened=True))
    event('DEV_SEARCH_CLOSED',shortlist=candidates)
    candidate_job('r18-frozen-shortlist-tune240',candidates,read(P/'split_manifest.json')['ids']['TUNE'])

if __name__=='__main__':main()
