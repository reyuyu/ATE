import argparse,unittest,time,signal
from relay_source_causality.common import hardware
from gqa3000_token_visual_writing.common import verify as verify_history
from .common import *
def freeze():
    from .tests import Metrics
    r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Metrics));assert r.wasSuccessful();assert read(OUT/'cpu_input_validation.json')['passed']
    old=verify_history();state=hardware();busy={l.split(',')[0].strip() for l in state['compute_processes'].splitlines()};idle=[]
    for row in state['table'].splitlines():
        i,u,n,used,free,util=[x.strip() for x in row.split(',')]
        if u not in busy and int(used)<200 and int(free)>22000 and int(util)==0:idle.append(dict(index=int(i),uuid=u,name=n))
    state['selected']=idle[:2] if len(idle)>=2 else [];atomic_json(OUT/'gpu_selection.json',state)
    atomic_json(OUT/'protocol.json',dict(status='FROZEN' if len(idle)>=2 else 'STOP_NO_TWO_IDLE_GPUS',base_commit=BASE,semantic_commit=SEM,sources=sources(),case_manifest_sha256=binding(),rules_sha256=sha256(OUT/'ranking_rules.json'),gpus=state['selected'],model_revision=REVISION,model_files=old['model_files'],transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],environment='FP16/Eager/Transformers4.49.0; torch.inference_mode; logits_to_keep=1; independent native KV per Prefill',cpu_tests=r.testsRun,worker_main_cases=[8,8],extra_validation_prefills=6,total_prefill_calls=22,decode_steps=0,answer_reuse_only=True))
    print('FROZEN',state['selected'],flush=True)
def run():
    p=read(OUT/'protocol.json');assert len(p['gpus'])==2 and p['sources']==sources() and p['case_manifest_sha256']==binding()
    children=[];logs=[];start=time.perf_counter()
    try:
        for rank in [0,1]:
            log=(OUT/f'worker{rank}/run.log');log.parent.mkdir(exist_ok=True);f=log.open('w');logs.append(f);env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpus'][rank]['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            children.append(subprocess.Popen([os.sys.executable,'-u','-m','semantic_omitted_token_attention_mass.worker','--rank',str(rank)],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parent))
        deadline=time.monotonic()+100
        while not all((OUT/f'worker{r}/ready.json').exists() for r in [0,1]):
            assert not list(OUT.glob('worker*/STOP.json')) and all(c.poll() is None for c in children),'Worker failed during smoke'
            assert time.monotonic()<deadline,'Smoke timeout';time.sleep(.2)
        assert all(read(OUT/f'worker{r}/smoke_validation.json')['passed'] for r in [0,1]);atomic_json(OUT/'GO.json',dict(both_smokes_passed=True));print('BOTH GPU SMOKES PASSED; RELEASE16-CASE MATRIX',flush=True)
        while any(c.poll() is None for c in children):
            assert not list(OUT.glob('worker*/STOP.json')) and all(c.poll() in [None,0] for c in children),'Worker failed; cancel our remaining worker only';time.sleep(.2)
        assert all(c.returncode==0 for c in children) and len(list((OUT/'checkpoints').glob('*.json')))==16
        atomic_json(OUT/'gpu_completion.json',dict(passed=True,worker_pids=[c.pid for c in children],wall_seconds=time.perf_counter()-start,cases=16,native_main_prefills=16,total_prefill_calls=22,decode_steps=0,new_answers=0,both_smokes_passed=True));print('OBSERVATION COMPLETE',round(time.perf_counter()-start,2),flush=True)
    except BaseException:
        import traceback
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=15)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        atomic_json(OUT/'STOP.json',dict(status='STOPPED',traceback=traceback.format_exc(),only_owned_worker_pids=[c.pid for c in children],hardware=hardware()));raise
    finally:
        for f in logs:f.close()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);args=a.parse_args();globals()[args.stage]()
