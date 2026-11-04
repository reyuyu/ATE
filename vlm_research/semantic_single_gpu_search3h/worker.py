"""Persistent one-model workers, explicit jobs, and a hard-deadline owner watchdog."""
import argparse,fcntl,subprocess,traceback
from .common import *
from gqa3000_token_visual_writing.engine import generate,ReceiverMask
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.engine import cache_fingerprint
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from vlm_common import load_model
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus
import query_role_smoke_fix.run as repaired

@torch.inference_mode()
def smoke(model,proc,rank):
    checks=[];h=History()
    for q in SMOKE_IDS:
        s=sample(q);sp=spec(q);inputs=prepared(model,proc,s);before=tensor_hashes(inputs);assert before==sp['input_hashes'];fp=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        ids,rec=repaired.native(model,inputs);reference=[v[-1].float() for v in rec.logits]
        g,_,trace,_,parent=generate(model,proc.tokenizer,inputs,sp['Q'],noop=True,capture=True,retain_cache=True)
        assert ids==g['token_ids'] and len(reference)==len(trace) and all(torch.equal(torch.from_numpy(a).float(),b) for a,b in zip(trace,reference));parent_hash=cache_fingerprint(parent);results=[]
        for label in ['Question-Only','Semantic-k','Semantic-first-Q80']:
            historical=h.get(q,label);calls=[0]*32;handles=[]
            try:
                for k,l in enumerate(model_layers(model)):
                    def mlp(m,args,out,k=k):assert torch.isfinite(out).all();calls[k]+=1
                    handles.append(l.mlp.register_forward_hook(mlp))
                gg,first,_,a,cache=generate(model,proc.tokenizer,inputs,historical['writers'],capture=False,retain_cache=True)
            finally:
                for hh in handles:hh.remove()
            assert calls==[gg['length']]*32 and gg['token_ids']==historical['generation']['token_ids'] and gg['text']==historical['generation']['text']
            path=read(historical['source']['path'])['raw']['path']
            with np.load(path) as z:assert torch.equal(torch.from_numpy(first),torch.from_numpy(z['first_logits']))
            assert cache_fingerprint(parent)==parent_hash and all(a.data_ptr()!=b.data_ptr() for la,lb in zip(parent.to_legacy_cache(),cache.to_legacy_cache()) for a,b in zip(la,lb))
            assert (a[:,0]==0).all() and (a[:,4:]==1).all();results.append(dict(condition=label,token_ids_exact=True,first_logits_torch_equal=True,all32_MLP=True,cache_independent=True,forbidden_actual_probability_zero=True));del cache
        del parent
        try:
            with ReceiverMask(model,inputs,sp['Q']):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert fp==hook_footprint(model) and versions=={n:p._version for n,p in model.named_parameters()} and before==tensor_hashes(inputs)
        checks.append(dict(question_id=q,Native_noop_all_valid_logits_torch_equal=True,replays=results,hook_exception_cleanup=True,weights_inputs_unchanged=True));print('SMOKE_PASS',rank,q,flush=True)
    h.close();atomic_json(P/f'worker{rank}/smoke.json',dict(passed=True,cases=checks,utc=utc(),full_logits_archived=False))

def collect(model,proc,inputs,q,w,rank,job):
    guard(gpu=True,reserve=30);path=result_path(q,w)
    if path.exists():
        r=read(path);assert r['writers']==w and sha(r['raw']['path'])==r['raw']['sha256'];return 'resumed',0.,0.
    before=tensor_hashes(inputs);assert before==spec(q)['input_hashes'];fp=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()};calls=[0]*32;handles=[];features=[]
    start=time.monotonic();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
    try:
        for k,l in enumerate(model_layers(model)):
            def mlp(m,args,out,k=k):assert torch.isfinite(out).all();calls[k]+=1
            handles.append(l.mlp.register_forward_hook(mlp))
        def projected(m,args,out):features.append(hashlib.sha256(out.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        handles.append(model.multi_modal_projector.register_forward_hook(projected));ev0.record()
        g,first,_,audit,_=generate(model,proc.tokenizer,inputs,w,capture=False);ev1.record();torch.cuda.synchronize();cuda_ms=float(ev0.elapsed_time(ev1))
    finally:
        for hh in handles:hh.remove()
    seconds=time.monotonic()-start
    assert len(features)==1 and calls==[g['length']]*32 and before==tensor_hashes(inputs) and fp==hook_footprint(model) and versions=={n:p._version for n,p in model.named_parameters()}
    assert (audit[:,0]==0).all() and (audit[:,4:]==1).all() and g['own_cache'] and g['own_prefix']
    metadata=dict(question_id=q,writers=w,worker=rank,first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),input_hashes=before,generation=g,projected_visual_sha256=features[0],MLP_calls=calls,weights_unchanged=True,hook_cleanup=True,job_id=job,seconds=seconds,cuda_event_ms=cuda_ms,utc=utc(),worker_source_sha256=sha(__file__))
    raw=atomic_npz(PRIVATE/'compact_audits'/q/(canonical(w)+'.npz'),audit=audit,token_ids=np.array(g['token_ids']),record_metadata=np.array(json.dumps(metadata,sort_keys=True)))
    atomic_json(path,dict(**metadata,raw=raw));return 'new',seconds,cuda_ms

def worker(rank):
    started=time.monotonic();torch.set_num_threads(4);p=verify_sources();W=P/f'worker{rank}';W.mkdir(exist_ok=True);lock=(W/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu=p['gpus'][rank]
    state=hardware();assert gpu in idle_gpus(state) and os.environ['CUDA_VISIBLE_DEVICES']==gpu['uuid'];atomic_json(W/'preflight.json',dict(**state,selected=gpu,utc=utc(),passed=True))
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1;atomic_json(W/'environment.json',{**env,'pid':os.getpid(),'gpu':gpu,'utc':utc()});smoke(model,proc,rank);atomic_json(W/'READY.json',dict(passed=True,utc=utc()))
    while not (P/'SMOKE_GATE.json').exists():guard(gpu=True);time.sleep(.25)
    total=0;seconds=0.;cuda_ms=0.;lastq=None;inputs=None;feature_cache={}
    while not (P/'END_WORKERS.json').exists():
        guard(gpu=True,reserve=10);jobs=sorted((P/'jobs').glob('*/manifest.json'))
        todo=[j for j in jobs if not (j.parent/f'worker{rank}_done.json').exists()]
        if not todo:time.sleep(.3);continue
        for jf in todo:
            job=read(jf);own=[r for r in job['tasks'] if r['worker']==rank];new=0;elapsed=0.;ms=0.
            for i,r in enumerate(own):
                q=r['question_id'];authorized(q)
                if q!=lastq:inputs=prepared(model,proc,sample(q));lastq=q
                mode,dt,dm=collect(model,proc,inputs,q,r['writers'],rank,job['job_id']);new+=mode=='new';elapsed+=dt;ms+=dm
                record=read(result_path(q,r['writers']));feature=record['projected_visual_sha256']
                if q in feature_cache:assert feature_cache[q]==feature
                feature_cache[q]=feature
                if (i+1)%20==0:print('JOB_PROGRESS',job['job_id'],rank,i+1,len(own),'new',new,flush=True)
            total+=new;seconds+=elapsed;cuda_ms+=ms;atomic_json(jf.parent/f'worker{rank}_done.json',dict(passed=True,job_id=job['job_id'],new_generations=new,requested=len(own),inference_seconds=elapsed,cuda_event_ms=ms,utc=utc()));print('JOB_DONE',job['job_id'],rank,new,flush=True)
    atomic_json(W/'completion.json',dict(passed=True,new_generations=total,inference_seconds=seconds,cuda_event_ms=cuda_ms,resident_seconds=time.monotonic()-started,pid=os.getpid(),utc=utc()))

def driver():
    p=verify_sources();children=[];logs=[];start=time.monotonic()
    try:
        for rank,g in enumerate(p['gpus']):
            W=P/f'worker{rank}';W.mkdir(exist_ok=True);log=(W/'run.log').open('a');logs.append(log);env={**os.environ,'CUDA_VISIBLE_DEVICES':g['uuid'],'PYTHONUNBUFFERED':'1'}
            child=subprocess.Popen([os.sys.executable,'-m','semantic_single_gpu_search3h.worker','worker','--rank',str(rank)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(child)
        assert children,'No idle GPU available'
        atomic_json(P/'driver_started.json',dict(pid=os.getpid(),children=[c.pid for c in children],utc=utc(),monotonic=start))
        while any(c.poll() is None for c in children):
            if remaining(gpu=True)<10:raise TimeoutError('270-minute GPU stop boundary reached')
            guard()
            if any(c.poll() not in [None,0] for c in children):raise RuntimeError('Worker failed; stop only owned children')
            if not (P/'SMOKE_GATE.json').exists() and all((P/f'worker{i}/READY.json').exists() for i in range(len(children))):
                atomic_json(P/'SMOKE_GATE.json',dict(passed=True,workers=len(children),utc=utc()));print('ALL_SMOKES_PASSED',flush=True)
            time.sleep(1)
        assert all(c.returncode==0 for c in children)
    except BaseException:
        if not (P/'STOP.json').exists():atomic_json(P/'STOP.json',dict(utc=utc(),traceback=traceback.format_exc()))
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=20)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        raise
    finally:
        for log in logs:log.close()
        state=hardware();selected={g['uuid'] for g in p['gpus']};busy=[r for r in state['compute_processes'].splitlines() if r.split(',')[0].strip() in selected]
        atomic_json(P/'gpu_release.json',dict(**state,utc=utc(),selected_uuids=sorted(selected),all_own_children_exited=all(c.poll() is not None for c in children),selected_devices_no_compute_processes=not busy,resident_driver_seconds=time.monotonic()-start))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['worker','driver']);a.add_argument('--rank',type=int);args=a.parse_args()
    try:worker(args.rank) if args.stage=='worker' else driver()
    except BaseException:
        if args.stage=='worker' and not (P/'STOP.json').exists():atomic_json(P/'STOP.json',dict(rank=args.rank,pid=os.getpid(),utc=utc(),traceback=traceback.format_exc()))
        raise

