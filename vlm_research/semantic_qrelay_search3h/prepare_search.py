import io,shutil,unittest
from .common import *
from . import search_selection
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus

def prepare():
    PRIVATE.mkdir(exist_ok=True);prior=read(PHASE1/'protocol.json');assert remaining()>3600;state=hardware();gpu=prior['gpus'][0];assert gpu in idle_gpus(state);assert shutil.disk_usage(P).free>5*1024**3
    for p,h in prior['shared_sources'].items():assert sha(p)==h
    shutil.copyfile(PHASE1/'split_manifest.json',P/'split_manifest.json')
    candidates={'S-assistant':dict(old=dict(anchors=['assistant'])),'S':dict(old={}), 'Q':dict(base='Q')}
    anchors=['period','postQ','space','phrase','or','word','single']
    for a in anchors:candidates['Q-'+a]=dict(anchors=[a])
    for a,b in [('period','phrase'),('period','word'),('postQ','period'),('postQ','phrase'),('space','period'),('space','phrase'),('word','phrase'),('or','phrase'),('single','word')]:candidates[f'Q-{a}-{b}']=dict(anchors=[a,b])
    for rule in ['whatwhich','long20','short10','notyesno']:
        for a in [None,'period','phrase','postQ']:candidates['Hybrid-'+rule+('-'+a if a else '')]=dict(base='conditional',rule=rule,anchors=[] if a is None else [a])
    budget=[];accepted=[];ids=sorted(read(P/'split_manifest.json')['metadata'])
    for cid,c in candidates.items():
        ratios=[]
        for q in ids:
            sp=spec(q);w=search_selection.select(sp,c);ratios.append(len(w)/len(sp['R']))
        mean=float(np.mean(ratios));okay=mean<.5;budget.append(dict(candidate_id=cid,mean_R_ratio=mean,eligible=okay,config=json.dumps(c,sort_keys=True)))
        D=P/'candidates'/cid;D.mkdir(parents=True);shutil.copyfile(search_selection.__file__,D/'selection_source.py');atomic_json(D/'definition.json',dict(candidate_id=cid,config=c,source_sha256=sha(D/'selection_source.py'),created_UTC=utc(),input_only=True,eligible=okay,mean_all3000_R_ratio=mean))
        if okay:accepted.append(cid)
    writecsv(P/'CPU_candidate_budget.csv',budget)
    protocol=dict(prior,source_hashes=source_hashes(),search_selector_sha256=sha(search_selection.__file__),split_sha256=sha(P/'split_manifest.json'),clock_sha256=sha(P/'clock.json'),accepted_candidates=accepted,phase='second small-sample search of Q plus native one/two-token relay groups',same_original_three_hour_clock=True,phase1_frozen_commit='03832c68f7bb09adea0af55b82014344806131b4',all_results_exploratory=True)
    atomic_json(P/'protocol.json',protocol);atomic_json(P/'resource_preflight.json',dict(**state,UTC=utc()))
    from .tests import Tests
    s=io.StringIO();r=unittest.TextTestRunner(stream=s,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));(P/'cpu_tests.log').write_text(s.getvalue());assert r.wasSuccessful();atomic_json(P/'CPU_GATE.json',dict(passed=True,tests=r.testsRun,all_candidates_all3000_whole_groups_S_preserved=True,UTC=utc()));print('FROZEN_PHASE2',len(accepted),flush=True)
