"""Explicit stage decisions, frozen before the next generation job."""
import argparse
from .common import *
from .experiments import candidate_job

def main(stage):
    if stage=='180':
        cids=['S-assistant-article','S-assistant-noun-article','S-assistant-fill49','S-assistant-fill49-early','S-assistant-fill49-role','S-assistant-fill46','S-article','S-noun-article','S-qtail3','S-question-extra4-role','S-allQ-period','S-instruction4','S-assistant','S'];job='s02-dev180';ids=read(P/'split_manifest.json')['dev_stages']['180'];reason='60-question scores are coarse. Retain leading combinations, budget/frontier variants, two exact-4 controls and original S/Assistant. Expand before selecting on small differences.'
    else:raise ValueError(stage)
    atomic_json(P/(job+'_decision.json'),dict(candidates=cids,reason=reason,prior_summary_sha256=sha(P/'jobs/s01-dev60/summary.csv'),UTC=utc(),before_next_outputs=True));candidate_job(job,cids,ids)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');main(p.parse_args().stage)
