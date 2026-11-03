"""One frozen Winner only: optional historical-pool expansion after FINAL600.

The original split and its authorization remain immutable. A narrowly scoped
wrapper admits only the predeclared remaining IDs; no inference function changes.
"""
import argparse,collections,fcntl,subprocess,traceback
from .common import *
from . import common,worker as backend
from .deploy import load_selector
from transformers import AutoProcessor
from semantic_visual_writers.groups import token_groups,intact
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus

E=P/'expanded3000'
def activate():
    m=read(E/'manifest.json');freeze=read(P/'winner_freeze.json');assert sha(P/'winner_freeze.json')==m['winner_freeze_sha256'] and m['candidate_id']==freeze['candidate_id']
    allowed=set(m['new_ids']);old=common.authorized
    def authorize(q):return 'EXPANSION' if q in allowed else old(q)
    common.authorized=authorize;backend.authorized=authorize
    return m

def prepare():
    start=time.monotonic();guard(reserve=3600);assert read(P/'jobs/final-winner-600/analysis_completion.json')['passed'];assert read(P/'gpu_release.json')['all_own_children_exited'];assert not E.exists();E.mkdir()
    source=P/'candidates'/read(P/'winner_freeze.json')['candidate_id']/'selection_source.py';f=read(P/'winner_freeze.json');assert sha(source)==f['selector_source_sha256']
    pool=read(HIST/'sample_manifest.json')['samples'];oldids=set(q for ss in read(P/'split_manifest.json')['ids'].values() for q in ss);remaining_ids=sorted((s['question_id'] for s in pool if s['question_id'] not in oldids),key=lambda q:digest(['expand-frozen-v1',q]));assert len(remaining_ids)==1800
    metadata={s['question_id']:{k:s[k] for k in ['question_id','task','image_sha256']} for s in pool};assign={q:int(digest(['worker-v1',metadata[q]['image_sha256']]),16)%2 for q in remaining_ids}
    # Freeze membership and single immutable algorithm before exposing any
    # additional outcomes. No subsequent selection or hyperparameter search.
    atomic_json(E/'manifest.json',dict(candidate_id=f['candidate_id'],winner_freeze_sha256=sha(P/'winner_freeze.json'),selector_source_sha256=sha(source),new_ids=remaining_ids,metadata=metadata,workers=assign,reason='User permits expansion of the single frozen Winner. FINAL600 is complete; expand without retuning to obtain descriptive historical GQA3000 performance.',interpretation='Mixed historical pool: DEV and TUNE participated in selection. Expanded 1800 is not automatically image-disjoint; clean-image subset reported separately.',original_split_unchanged=True,protocol_sha256=sha(P/'protocol.json'),execution_source_sha256=sha(__file__),backend_source_sha256=sha(backend.__file__),shared_source_hashes=source_hashes(),utc=utc(),monotonic=time.monotonic()))
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);pos=read(BASE/'results/noun_article_visual_writers_pilot/pos_parse_all3000.json');hashes=read(HIST/'input_hashes.json');activate();history=History();index=[];selector=load_selector(P);lookup={s['question_id']:s for s in pool}
    for q in remaining_ids:
        s=lookup[q];original_hash=digest(s);clean={k:v for k,v in s.items() if k!='answer'};clean['input_mapping']={k:v for k,v in s['input_mapping'].items() if k!='gold_continuation_ids'}
        atomic_json(PRIVATE/'samples'/f'{q}.json',clean);atomic_json(PRIVATE/'gold'/f'{q}.json',dict(answer=s['answer']))
        file=HIST/'token_manifests'/(s['task']+'__'+q+'.json');history.committed(file);old=read(file);chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
        assert digest(chat)==s['input_mapping']['chat_prompt_sha256'];groups,Q,words=token_groups(processor.tokenizer,s['question'],chat,s['input_mapping']['prompt_ids'],s['input_mapping']['visual_positions'],old['relay_positions']);assert groups==old['groups'] and Q==old['question_positions'] and words==old['words']
        startchar=chat.replace('<image>','<image>'*576).index(s['question'])
        for g in groups:
            matches=[t for t in pos[q]['tokens'] if any(w['in_question'] and set(w['positions'])&set(g['positions']) and t['start']<w['end']-startchar and t['end']>w['start']-startchar for w in words)];g['pos']=sorted({t['pos'] for t in matches});g['dep']=sorted({t['dep'] for t in matches})
        sp=dict(question=s['question'],task=s['task'],Q=Q,R=old['relay_positions'],groups=groups,words=words,k80=old['budgets']['80']['actual_k'],historical_writers=old['writers'],historical_aliases=old['aliases'],input_hashes=hashes[s['task']+'__'+q],original_sample_sha256=original_hash,prompt_ids_sha256=old['prompt_ids_sha256'],historical_token_manifest_sha256=sha(file),input_sample_sha256=digest(clean));atomic_json(P/'input_specs'/f'{q}.json',sp);w=selector(sp);assert intact(w,groups) and set(w)<=set(sp['R']);index.append(dict(question_id=q,writers=w,k=len(w),R=len(sp['R']),input_hashes=sp['input_hashes'],input_spec_sha256=sha(P/'input_specs'/f'{q}.json'),worker=assign[q]))
    history.close();atomic_json(E/'token_manifest.json',dict(cases=index,selector_sha256=sha(source)));atomic_json(E/'CPU_preparation.json',dict(passed=True,new_questions=1800,seconds=time.monotonic()-start,token_manifest_sha256=sha(E/'token_manifest.json'),historical_manifest_audit=history.files,no_tuning=True,utc=utc()));event('FROZEN_WINNER_EXPANSION_PREPARED',seconds=time.monotonic()-start,new_questions=1800);print(json.dumps(dict(prepared=True,n=1800,seconds=time.monotonic()-start)),flush=True)

def run_worker(rank):
    started=time.monotonic();m=activate();guard(gpu=True,reserve=1800);torch.set_num_threads(4);p=verify_sources();W=E/f'worker{rank}';W.mkdir(exist_ok=True);lock=(W/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu=p['gpus'][rank];state=hardware();assert gpu in idle_gpus(state) and os.environ['CUDA_VISIBLE_DEVICES']==gpu['uuid'];atomic_json(W/'preflight.json',dict(**state,selected=gpu,utc=utc(),passed=True))
    assert sha(__file__)==m['execution_source_sha256'] and sha(backend.__file__)==m['backend_source_sha256'];model,proc,env=backend.load_model(MODEL);assert torch.cuda.device_count()==1;atomic_json(W/'environment.json',{**env,'pid':os.getpid(),'gpu':gpu,'utc':utc()})
    initialP=backend.P;backend.P=E
    try:backend.smoke(model,proc,rank)
    finally:backend.P=initialP
    atomic_json(W/'READY.json',dict(passed=True,utc=utc()))
    while not (E/'SMOKE_GATE.json').exists():guard(gpu=True);time.sleep(.25)
    tasks=sorted((r for r in read(E/'token_manifest.json')['cases'] if r['worker']==rank),key=lambda r:(m['metadata'][r['question_id']]['image_sha256'],r['question_id']));total=0;seconds=0.;ms=0.
    for i,r in enumerate(tasks):
        q=r['question_id'];inputs=backend.prepared(model,proc,sample(q));mode,dt,dm=backend.collect(model,proc,inputs,q,r['writers'],rank,'expanded-winner-1800');total+=mode=='new';seconds+=dt;ms+=dm
        if (i+1)%50==0:print('EXPANSION_PROGRESS',rank,i+1,len(tasks),flush=True)
    atomic_json(W/'completion.json',dict(passed=True,new_generations=total,n=len(tasks),inference_seconds=seconds,cuda_event_ms=ms,resident_seconds=time.monotonic()-started,pid=os.getpid(),utc=utc()));print('EXPANSION_DONE',rank,total,flush=True)

def driver():
    m=activate();guard(gpu=True,reserve=2400);assert read(E/'CPU_preparation.json')['passed'];assert sha(__file__)==m['execution_source_sha256'];state=hardware();gpus=read(P/'protocol.json')['gpus'];assert all(g in idle_gpus(state) for g in gpus);atomic_json(E/'resource_preflight.json',dict(**state,utc=utc()));children=[];logs=[];start=time.monotonic()
    try:
        for rank,gpu in enumerate(gpus):
            W=E/f'worker{rank}';W.mkdir(exist_ok=True);log=(W/'run.log').open('a');logs.append(log);env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'PYTHONUNBUFFERED':'1'};children.append(subprocess.Popen([os.sys.executable,'-m','gqa5h_receiver_search.expansion','worker','--rank',str(rank)],env=env,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT))
        atomic_json(E/'driver_started.json',dict(pid=os.getpid(),children=[c.pid for c in children],utc=utc(),monotonic=start))
        while any(c.poll() is None for c in children):
            guard(gpu=True,reserve=10)
            if any(c.poll() not in [None,0] for c in children):raise RuntimeError('Expansion worker failed; stop only own children')
            if not (E/'SMOKE_GATE.json').exists() and all((E/f'worker{i}/READY.json').exists() for i in range(2)):atomic_json(E/'SMOKE_GATE.json',dict(passed=True,workers=2,utc=utc()));print('EXPANSION_BOTH_SMOKES_PASSED',flush=True)
            time.sleep(1)
        assert all(c.returncode==0 for c in children)
    except BaseException:
        atomic_json(E/'STOP.json',dict(traceback=traceback.format_exc(),utc=utc()))
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=20)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        raise
    finally:
        for log in logs:log.close()
        state=hardware();selected={g['uuid'] for g in gpus};busy=[r for r in state['compute_processes'].splitlines() if r.split(',')[0].strip() in selected];atomic_json(E/'gpu_release.json',dict(**state,utc=utc(),selected_uuids=sorted(selected),all_own_children_exited=all(c.poll() is not None for c in children),selected_devices_no_compute_processes=not busy,resident_driver_seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','driver','worker']);ap.add_argument('--rank',type=int);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='driver':driver()
    else:run_worker(a.rank)
