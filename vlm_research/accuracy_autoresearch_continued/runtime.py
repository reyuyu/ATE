"""Reuse the immutable engine; isolate new outputs and verify old record bindings."""
import argparse
import os
import time
import traceback

from ..accuracy_autoresearch import common as original
from ..accuracy_autoresearch import worker as w
from ..accuracy_autoresearch.common import *

PREVIOUS = Path('/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h')
PRIOR = PREVIOUS/'vlm_research/results/accuracy_autoresearch_6h'
OUT = ROOT/'vlm_research/results/accuracy_autoresearch_6h_continued'


def sources():
    value=original.source_hashes()
    value[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    return value


def read_record(c,k,s):
    current=OUT/'records'/config_id(c)/(k+'.json')
    old=PRIOR/'records'/config_id(c)/(k+'.json')
    path=current if current.exists() else old
    if not path.exists(): return None
    blob=read(path); r=blob['payload']
    assert digest(r)==blob['sha256'] and r['sample_sha256']==digest(s) and r['config']==c
    assert r['sources']==(sources() if path==current else original.source_hashes())
    assert r['generation']['score']==score(s,r['generation']['text'])
    if c['prompt_variant']=='original':
        assert r['input_binding']['prompt_ids']==s['input_mapping']['prompt_ids']
    return r


def configure():
    assert read(ROOT/'task_deadline.json')==read(PREVIOUS/'task_deadline.json')
    assert original.source_hashes()==read(PRIOR/'execution_freeze.json')['sources']
    w.OUT=OUT
    w.source_hashes=sources
    w.read_record=read_record
    w.path_for=lambda c,k: OUT/'records'/config_id(c)/(k+'.json')


def worker(rank):
    configure(); frozen=read(OUT/'execution_freeze.json')
    assert sources()==frozen['sources']
    device=frozen['devices'][rank]
    assert os.environ['CUDA_VISIBLE_DEVICES']==device['uuid']
    state=hardware(); assert device['uuid'] not in state['processes']
    pool=read(PRIOR/'sample_manifest.json'); parts=read(PRIOR/'split_manifest.json')
    adapter=None
    try:
        adapter=w.Adapter('llava')
        old_env=read(PRIOR/f'worker{rank}/environment.json')
        for name in ('model_id','model_path','model_revision','python','torch','transformers',
                     'cuda_runtime','parameter_dtypes','cuda_visible_devices'):
            assert adapter.environment[name]==old_env[name],name
        atomic(OUT/f'worker{rank}/environment.json',adapter.environment)
        w.smoke(adapter,rank,pool,parts)
        k=next(k for k in parts['coarse'] if parts['image_shard'][pool[k]['image_sha256']]==rank)
        s=pool[k]; audits=[]
        native=config('native'); layout,writers,_=w.prepare(adapter,s,native)
        value=w.generate(adapter,layout,native,writers,2,True,True)
        prior=read_record(native,k,s)['generation']
        assert w.tensor_hash(value[1][0])==prior['first_logits_sha256']
        assert value[0]['token_ids']==prior['token_ids'][:len(value[0]['token_ids'])]
        for mode in ('none','decode'):
            c=config('all',mode,32); layout,writers,_=w.prepare(adapter,s,c)
            actual=w.generate(adapter,layout,c,writers,2,True,True)
            reference=w.generate(adapter,layout,c,writers,2,True,True,True)
            eq=w.equivalence('llava',reference[1],actual[1],reference[3],actual[3])
            assert eq['passed'] and actual[0]['token_ids']==reference[0]['token_ids']
            assert all(e['blocked_probability_max']==0 for e in actual[0]['audit'])
            audits.append(dict(config=c,numerical=eq,events=actual[0]['audit']))
        atomic(OUT/f'worker{rank}/additional_smoke.json',dict(status='PASS',audits=audits,
            exact_historical_native_logits_and_prefix=True,sources=sources()))
        while not all((OUT/f'worker{i}/additional_smoke.json').exists() for i in range(len(frozen['devices']))):
            ensure_time(True)
            assert not any((OUT/f'worker{i}/STOP.json').exists() for i in range(len(frozen['devices'])))
            time.sleep(1)
        while time.time()<deadline('stop_new_gpu_work_by') and not (OUT/'SHUTDOWN.json').exists():
            for path in sorted((OUT/'jobs').glob('*.json')):
                j=read(path); marker=OUT/f'worker{rank}/jobs'/(j['id']+'.json')
                assert j['sources']==sources() and j['samples_sha256']==sha(PRIOR/'sample_manifest.json')
                if marker.exists():
                    assert read(marker)['job_sha256']==digest(j)
                    if j.get('finish_after'): return
                    continue
                if j['split']=='RECHECK':
                    assert j['selection_sha256']==digest(read(OUT/'selection.json'))
                keys=[k for k in j['samples'] if parts['image_shard'][pool[k]['image_sha256']]==rank]
                started=time.time()
                for ix,k in enumerate(keys):
                    ensure_time(True)
                    assert not any((OUT/f'worker{i}/STOP.json').exists() for i in range(len(frozen['devices'])))
                    for c in j['configs']:
                        if read_record(c,k,pool[k]) is None: w.generate_one(adapter,k,pool[k],c,j)
                    if j.get('performance') and not (OUT/'performance'/j['id']/(k+'.json')).exists():
                        w.performance(adapter,k,pool[k],j['configs'],j)
                    if (ix+1)%10==0 or ix+1==len(keys):
                        print('PROGRESS',rank,j['id'],ix+1,len(keys),round(time.time()-started,1),flush=True)
                atomic(marker,dict(status='COMPLETE',at=now(),samples=len(keys),
                    job_sha256=digest(j),elapsed_seconds=time.time()-started))
                if j.get('finish_after'): return
            time.sleep(1)
    except BaseException:
        atomic(OUT/f'worker{rank}/STOP.json',dict(at=now(),traceback=traceback.format_exc()))
        raise
    finally:
        del adapter
        w.torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/released.json',dict(at=now(),pid=os.getpid(),sources=sources()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True)
    worker(ap.parse_args().rank)
