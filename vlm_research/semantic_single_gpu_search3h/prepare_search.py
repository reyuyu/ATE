import io,shutil,unittest
from .common import *
from . import search_selection
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus

def prepare():
    PRIVATE.mkdir(exist_ok=True);prior=read(RECENT/'protocol.json');clock=read(P/'clock.json');assert remaining()>3600;state=hardware();gpu=prior['gpus'][0];assert gpu in idle_gpus(state);assert gpu['uuid']=='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
    assert shutil.disk_usage(P).free>5*1024**3
    for p,h in prior['shared_sources'].items():assert sha(p)==h
    for p,e in prior['model_files'].items():assert sha(p)==e['sha256']
    shutil.copyfile(RECENT/'split_manifest.json',P/'split_manifest.json')
    candidates={'S':dict(old={}), 'S-assistant':dict(old=dict(anchors=['assistant'])),'S-assistant-noun-article':dict(old=dict(anchors=['assistant'],categories=['noun','article']))}
    for cap in [.46,.49]:
        for order in ['late','early','middle','relation','logic','noun','function','attribute','syntax']:candidates[f'SA-cap{int(cap*100)}-{order}']=dict(cap=cap,order=order)
    for anchor in ['tail','space']:
        for order in ['late','relation','function']:candidates[f'SA-{anchor}-cap49-{order}']=dict(cap=.49,order=order,anchor=anchor)
    for rule in ['relation','whatwhich','yesno','verb']:candidates['SA-complete-'+rule]=dict(mode='conditional',rule=rule)
    for n in [10,15,20]:
        for rule in ['short','long']:candidates[f'SA-complete-{rule}{n}']=dict(mode='conditional',rule=rule,length=n)
    budget=[];accepted=[];ids=sorted(read(P/'split_manifest.json')['metadata'])
    for cid,c in candidates.items():
        ratios=[]
        for q in ids:
            sp=spec(q);w=search_selection.select(sp,c);ratios.append(len(w)/len(sp['R']))
        mean=float(np.mean(ratios));okay=mean<.5;budget.append(dict(candidate_id=cid,mean_R_ratio=mean,eligible=okay,config=json.dumps(c,sort_keys=True)))
        D=P/'candidates'/cid;D.mkdir(parents=True);shutil.copyfile(search_selection.__file__,D/'selection_source.py');atomic_json(D/'definition.json',dict(candidate_id=cid,config=c,source_sha256=sha(D/'selection_source.py'),created_UTC=utc(),input_only=True,eligible=okay,mean_all3000_R_ratio=mean))
        if okay:accepted.append(cid)
    writecsv(P/'CPU_candidate_budget.csv',budget)
    protocol=dict(prior,gpus=[gpu],source_hashes=source_hashes(),search_selector_sha256=sha(search_selection.__file__),split_sha256=sha(P/'split_manifest.json'),clock_sha256=sha(P/'clock.json'),plan=['all eligible DEV60','top8 DEV180','top4 DEV360','top2 CHECK600','one frozen winner ALL3000'],selection_rule='Max complete-answer correct count, tie lower mean R then candidate ID; original S preserved; mean R<50%.',all_data_previously_seen=True,no_confirmatory_holdout=True,accepted_candidates=accepted,one_GPU_only=True)
    atomic_json(P/'protocol.json',protocol);atomic_json(P/'resource_preflight.json',dict(**state,UTC=utc()))
    from .tests import Tests
    s=io.StringIO();r=unittest.TextTestRunner(stream=s,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));(P/'cpu_tests.log').write_text(s.getvalue());assert r.wasSuccessful();atomic_json(P/'CPU_GATE.json',dict(passed=True,tests=r.testsRun,all_candidates_all3000_whole_groups_S_preserved=True,UTC=utc()));print('FROZEN',len(accepted),'eligible candidates',flush=True)
    return accepted
