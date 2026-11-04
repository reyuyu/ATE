import ast,collections,io,shutil,unittest
from .common import *
from .selection import selected
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus

def run_tests(start):
    from .tests import Tests
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    log=P/'cpu_tests.log'
    if log.exists():shutil.copyfile(log,P/'cpu_tests_initial_attempt.log')
    log.write_text(stream.getvalue());assert r.wasSuccessful(),stream.getvalue();atomic_json(P/'CPU_GATE.json',dict(passed=True,tests=r.testsRun,seconds=time.monotonic()-start,UTC=utc()));event('PREPARED',seconds=time.monotonic()-start);print(json.dumps(dict(passed=True,seconds=time.monotonic()-start,remaining_minutes=remaining()/60)))

def main():
    start=time.monotonic()
    if (P/'protocol.json').exists():verify_sources();run_tests(start);return
    oldclock=read(P/'clock.json');atomic_json(P/'initial_5h_clock_before_user_3h.json',oldclock)
    # The user's explicit 3-hour override keeps the conservative original start.
    newclock=dict(oldclock,deadline_monotonic=oldclock['start_monotonic']+10800,gpu_stop_monotonic=oldclock['start_monotonic']+9000,design_stop_monotonic=oldclock['start_monotonic']+7200,deadline=(datetime.datetime.fromisoformat(oldclock['start_time'])+datetime.timedelta(hours=3)).isoformat(),authorization='User: no new data, 3 hours, improve from S toward 87%, preserve mean R<50%.')
    (P/'clock.json').write_text(json.dumps(newclock,indent=2)+'\n');guard(reserve=1800);PRIVATE.mkdir(parents=True,exist_ok=True)
    oldsplit=read(OLDP/'split_manifest.json');meta=read(OLDP/'expanded3000/manifest.json')['metadata'];dev=oldsplit['ids']['DEV'];used={meta[q]['image_sha256'] for q in dev};check=[]
    for t in TASKS:
        picked=[]
        for q in sorted((q for q in meta if meta[q]['task']==t),key=lambda q:digest(['round2-check600',q,meta[q]['image_sha256']])):
            im=meta[q]['image_sha256']
            if im not in used:picked.append(q);used.add(im)
            if len(picked)==100:break
        assert len(picked)==100;check+=picked
    splits=dict(ids=dict(DEV=dev,CHECK=check,REST=sorted(set(meta)-set(dev)-set(check))),dev_stages=oldsplit['dev_stages'],metadata=meta,all_original_3000_previously_seen=True,no_confirmatory_holdout=True,selection='Fixed SHA by task and image, no correctness filtering.')
    atomic_json(P/'split_manifest.json',splits)
    prior=read(OLDP/'protocol.json');state=hardware();free=idle_gpus(state);gpus=prior['gpus'];assert all(g in free for g in gpus);assert shutil.disk_usage(P).free>3*1024**3
    for path,value in prior['shared_sources'].items():assert sha(path)==value
    oldworker=OLDP.parents[1]/'gqa5h_receiver_search/worker.py';newworker=Path(__file__).with_name('worker.py')
    def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(oldworker),functions(newworker);assert a['smoke']==b['smoke'] and a['collect']==b['collect']
    for path,entry in prior['model_files'].items():assert sha(path)==entry['sha256']
    atomic_json(P/'protocol.json',dict(parent_commit='a1c9d81db89f59142a28295f2f1eeaa2b93e8fe6',scope='Same frozen GQA3000, exploratory only, preserve original S and add whole existing R groups; no Prompt change.',model_revision=REVISION,precision='float16',attention='eager',transformers='4.49.0',max_new_tokens=32,logits_to_keep=1,gpus=gpus,shared_sources=prior['shared_sources'],model_files=prior['model_files'],split_sha256=sha(P/'split_manifest.json'),clock_sha256=sha(P/'clock.json'),numeric_smoke_and_collect_AST_equal_to_parent=True,budget_limit=.5,stats=dict(bootstrap=20000,seed=20261104,cluster='Image SHA256',interpretation='Exploratory adaptive search. Winner expansion is descriptive; no fresh holdout.')))
    atomic_json(P/'resource_preflight.json',dict(**state,selected=gpus,free_bytes=shutil.disk_usage(P).free,UTC=utc()))
    run_tests(start)

if __name__=='__main__':main()
