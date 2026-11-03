"""Select once on TUNE, bind a single deployable selector, then unseal FINAL."""
import argparse,importlib.util
from .common import *

def main(tune_job):
    guard(reserve=1500);assert not (P/'winner_freeze.json').exists();shortlist=read(P/'tune_shortlist.json');assert shortlist['job_id']==tune_job
    rows=list(csv.DictReader((P/'jobs'/tune_job/'summary.csv').open()));baselines=list(csv.DictReader((P/'baselines/TUNE/summary.csv').open()));eligible=[]
    for r in rows:
        assert r['candidate_id'] in shortlist['new_candidates'] and int(r['n'])==240
        eligible.append(dict(candidate_id=r['candidate_id'],correct=int(r['correct']),mean_R_ratio=float(r['mean_R_ratio']),origin='new_candidate'))
    mapping={'Question-Only':'baseline-q100','Semantic-first-Q80':'baseline-sem-q80','Semantic-k':'baseline-sem-k'}
    for r in baselines:
        if r['condition'] in mapping:eligible.append(dict(candidate_id=mapping[r['condition']],correct=int(r['correct']),mean_R_ratio=float(r['mean_R_ratio']),origin='historical_baseline'))
    eligible=[r for r in eligible if r['mean_R_ratio']<.5];assert eligible
    complexity={'historical':0,'role':1,'structural_prune':1,'late_relay':1,'risk':2,'router':3}
    for r in eligible:r['complexity']=complexity[read(P/'candidates'/r['candidate_id']/'definition.json')['config']['family']]
    eligible.sort(key=lambda r:(-r['correct'],r['mean_R_ratio'],r['complexity'],r['candidate_id']));winner=eligible[0];cid=winner['candidate_id'];definition=read(P/'candidates'/cid/'definition.json');source=P/'candidates'/cid/'selection_source.py';assert sha(source)==definition['source_sha256']
    loader=importlib.util.spec_from_file_location('final_frozen_selector',source);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    split=read(P/'split_manifest.json');manifest=[];minimal_keys=['question','Q','R','groups','k80','historical_writers']
    # Only public input metadata is opened here. No FINAL sample(), Gold,
    # history record, model answer, NLL or attention value is accessible.
    for q in split['ids']['FINAL']:
        s=spec(q);minimal={k:s[k] for k in minimal_keys};w=module.select(minimal,definition['config']);assert w==module.select(s,definition['config']) and w==sorted(set(w)) and set(w)<=set(s['R'])
        manifest.append(dict(question_id=q,image_sha256=split['metadata'][q]['image_sha256'],writers=w,k=len(w),R=len(s['R']),R_ratio=len(w)/len(s['R']),input_hashes=s['input_hashes'],input_spec_sha256=sha(P/'input_specs'/f'{q}.json')))
    budget=float(np.mean([r['R_ratio'] for r in manifest]));assert budget<.5,'Frozen Winner exceeds authorized FINAL input budget; do not change Winner after this input-only check.'
    atomic_json(P/'final_writer_manifest.json',dict(candidate_id=cid,source_sha256=sha(source),config_sha256=digest(definition['config']),cases=manifest,input_only=True))
    atomic_json(P/'selection_rules.json',dict(candidate_id=cid,config=definition['config'],source_path=str(source.relative_to(ROOT)),source_sha256=sha(source),input_keys=minimal_keys,full_word_closure='Original LLaVA tokenizer offsets and original semantic groups; never split an English word.',budget_rule='Exact historical per-question k80 for fixed-Q80 candidates; otherwise the frozen configuration, with actual mean k/R reported.',fallback='Deterministic exact whole-group DP; assert on infeasible inputs, never choose using Gold or model outputs.',runtime_uses_Gold=False,runtime_uses_image_contents=False,runtime_uses_question_ID=False,statistics=dict(bootstrap=20000,seed=20261104,cluster='Image SHA256',CIs='Exploratory pointwise 95%',paired_comparison_family=['Winner-All','Winner-Q100','Winner-Q80','Winner-S'],multiplicity='Holm correction on four exact paired sign/McNemar tests; pointwise bootstrap intervals remain explicitly unadjusted.',noninferiority_claim=False)))
    writecsv(P/'tune_winner_ranking.csv',eligible)
    atomic_json(P/'winner_freeze.json',dict(candidate_id=cid,tune_winner=winner,ranking=eligible,rule='Maximize TUNE exact complete-answer correct count under strict mean R<50%; ties: lower R, simpler method, lexical Candidate ID. Updated budget authorized by user before all candidate testing.',definition_sha256=sha(P/'candidates'/cid/'definition.json'),selector_source_sha256=sha(source),selection_rules_sha256=sha(P/'selection_rules.json'),final_writer_manifest_sha256=sha(P/'final_writer_manifest.json'),split_sha256=sha(P/'split_manifest.json'),tune_job=tune_job,tune_results_sha256=sha(P/'jobs'/tune_job/'results.csv'),baseline_TUNE_results_sha256=sha(P/'baselines/TUNE/per_question.csv'),Gold_and_outputs_on_FINAL_not_yet_opened=True,final_input_budget=budget,freeze_monotonic=time.monotonic(),freeze_utc=utc(),remaining_seconds=remaining(),source_hashes=source_hashes()))
    event('WINNER_FROZEN',candidate_id=cid,tune_correct=winner['correct'],tune_R=winner['mean_R_ratio'],final_input_R=budget);print(json.dumps(read(P/'winner_freeze.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tune-job',required=True);main(p.parse_args().tune_job)
