"""Three fixed quality conditions; image-cluster-isolated GPU workers."""
import argparse
import collections
import gc
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .common import *
from .engine import generate, prepare
from ..accuracy_autoresearch.common import config
from ..accuracy_autoresearch.engine import generate as inherited
from ..fastp2_bypass_writer.common import hardware
from ..fastp2_bypass_writer.run import operators
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def check_stop():
    assert not (OUT/'STOP_REQUESTED').exists(), 'Task stop requested'
    assert not list(OUT.glob('worker*/failure.json')), 'A task worker failed'
    v = os.statvfs('/data')
    assert v.f_bavail*v.f_frsize > 20*2**30, 'Disk safety floor reached'


def smoke(a, rank, manifest, partition):
    checks = []
    for key in partition['smoke']:
        check_stop()
        s = manifest[key]; layout, binding = prepare(a,s)
        native = generate(a,layout,'Native',3,capture=True,fixed=True)
        noop = generate(a,layout,'Native',3,capture=True,fixed=True,noop=True)
        assert all(torch.equal(x,y) for x,y in zip(native[2],noop[2]))
        with torch.inference_mode():
            hf = a.model.generate(**a.inputs,do_sample=False,num_beams=1,
                max_new_tokens=32,use_cache=True,logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
        natural = generate(a,layout,'Native',capture=True)
        assert hf.sequences[0,layout.length:].tolist() == natural[0]['token_ids']
        assert len(natural[1])==len(hf.scores)
        assert all(torch.equal(x,y.detach().float().cpu()) for x,y in zip(natural[1],hf.scores))
        noop_natural=generate(a,layout,'Native',capture=True,noop=True)
        assert noop_natural[0]['token_ids']==natural[0]['token_ids']
        assert len(noop_natural[2])==len(natural[2])
        assert all(torch.equal(x,y) for x,y in zip(natural[2],noop_natural[2]))
        del noop_natural
        with a.visual_replay():replay=generate(a,layout,'Native',capture=True)
        assert all(torch.equal(x,y) for x,y in zip(natural[2],replay[2]))
        del replay
        first_native = tensor_hash(native[1][0]); del native,noop,natural,hf
        details = []
        for label in LABELS:
            if label=='Native':continue
            with operators(a) as (calls,mlp,shapes):
                actual = generate(a,layout,label,3,capture=True,audit=True,fixed=True)
            assert all(mlp[i]==3 for i in range(32)), 'MLP skipped'
            assert all(calls[i,op]==1 for i in (2,3,4,5)
                for op in ('q_proj','k_proj','v_proj','o_proj')), 'Bypass not physical'
            if label == LABELS[0]:
                reference = inherited(a,layout,config('all','both',head=1),list(layout.relay),3,True,True)
                assert all(torch.equal(x,y) for x,y in zip(actual[2],reference[2]))
                gate = dict(passed=True,strict_historical_C3_logits=True)
            else:
                reference = generate(a,layout,label,3,capture=True,audit=True,dense=True,fixed=True)
                gate = equivalence('llava',reference[1],actual[1],reference[3],actual[3])
                assert gate['passed'], gate
            ref_tokens = reference[0]['token_ids']
            assert actual[0]['token_ids'][:len(ref_tokens)] == ref_tokens
            assert all(r['blocked_probability_max']==0 for r in actual[0]['attention_audit'])
            for event in actual[0]['visual_audit']:
                assert (event['answer_image_mass']>0) == (event['layer'] in KEEP[label])
                if event['prefill'] and event['layer']>=20:
                    assert event['relay_image_mass'] == 0
            assert actual[0]['kv_audit']['retained_image_layers'] == list(KEEP[label])
            assert actual[0]['kv_audit']['historical_cache_copies'] == 0
            assert len(actual[3]) == 32 and len(actual[4]) == 3
            for x,y in zip(actual[4],reference[4]):
                assert x.keys()==y.keys() and all(torch.equal(x[f],y[f]) for f in x)
            plain = generate(a,layout,label)
            with a.visual_replay():
                replay0 = generate(a,layout,label,capture=True)
                replay1 = generate(a,layout,label,capture=True)
            assert plain[0]['token_ids']==replay0[0]['token_ids']==replay1[0]['token_ids']
            assert all(torch.equal(x,y) for x,y in zip(replay0[2],replay1[2]))
            details.append(dict(condition=label,equivalence=gate,tokens=plain[0]['token_ids'],
                blocked_probability_max=0,retained_image_layers=list(KEEP[label]),
                visual_hidden_cutoff=CUTOFF[label],no_extra_R_writing=True,
                qkv_bypass_and_full_mlp=True,rotary_exact=True,vision_replay_exact=True,
                first_logits_sha256=plain[0]['first_logits_sha256']))
            del actual,reference,plain,replay0,replay1
        a.verify()
        checks.append(dict(question_id=key,input_binding=binding,native_noop_exact=True,
            native_HF_generation_exact=True,native_first_sha256=first_native,conditions=details))
    atomic(OUT/f'worker{rank}/smoke.json',dict(status='PASS',at=now(),sources=sources(),checks=checks))


def wait_smokes():
    start = time.monotonic()
    while not all((OUT/f'worker{r}/smoke.json').exists() for r in range(2)):
        check_stop()
        assert time.monotonic()-start < 900, 'Peer smoke timeout'
        time.sleep(1)


def worker(rank):
    f = freeze_check(); binding = digest(f)
    for path,value in f['weights'].items():assert sha(path)==value,'Model or Processor file SHA changed'
    device = DEVICES[rank]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == device['uuid']
    h = hardware()
    assert device['uuid'] not in h['processes'], 'Selected GPU is no longer idle'
    atomic(OUT/f'worker{rank}/preflight.json',dict(at=now(),pid=os.getpid(),**h))
    a, db = None, None
    start = time.monotonic()
    try:
        entries = samples(); manifest = {s['question_id']:s for s in entries}
        partition = read(OUT/'partition.json')
        a = Adapter('llava')
        atomic(OUT/f'worker{rank}/environment.json',a.environment)
        smoke(a,rank,manifest,partition)
        wait_smokes()
        selected = [s for s in entries if partition['image_shard'][s['image_sha256']]==rank]
        selected.sort(key=lambda s:(digest([SEED,'order',s['image_sha256']]),s['question_id']))
        assert len(selected) == partition['question_counts'][rank]
        del entries,manifest
        db = database(rank,binding)
        existing = {}
        for row in db.execute('SELECT payload,sha256 FROM results'):
            r = unpack(row)
            assert r['binding']==binding and r['rank']==rank
            existing[r['question_id'],r['condition']] = r
        expected = {s['question_id']:s for s in selected}
        assert all(q in expected and c in LABELS for q,c in existing)
        done = correct = 0
        previous, cached, visual_hashes, pixel_hash = None, None, None, None
        quality_start = time.monotonic()
        fresh = 0
        for s in selected:
            check_stop()
            qid = s['question_id']; sd = digest(s)
            old = [existing.get((qid,label)) for label in LABELS]
            if all(old):
                for r in old:
                    assert r['sample_sha256']==sd and r['score']==score(s,r['generation']['text'])
                done += 1
                continue
            layout, ib = prepare(a,s)
            assert sha(s['image_path'])==s['image_sha256'],'Input image SHA changed'
            if previous == s['image_sha256']:
                assert a.input_hashes['pixel_values']==pixel_hash
                a.visual,a.visual_hashes = cached,visual_hashes
            else:
                previous = s['image_sha256']
                pixel_hash = a.input_hashes['pixel_values']
            with a.visual_replay():
                for label, r in zip(LABELS,old):
                    if r:
                        assert r['sample_sha256']==sd and r['input_binding']==ib
                        continue
                    g,*_ = generate(a,layout,label)
                    record = dict(question_id=qid,condition=label,image_id=s['image_id'],
                        image_sha256=s['image_sha256'],rank=rank,at=now(),binding=binding,
                        sample_sha256=sd,input_binding=ib,generation=g,score=score(s,g['text']))
                    put(db,record)
                    fresh += 1
            cached,visual_hashes = a.visual,a.visual_hashes
            a.verify()
            done += 1
            if done % 50 == 0 or done==len(selected):
                elapsed = time.monotonic()-quality_start
                state = dict(at=now(),stage='quality',questions=done,total=len(selected),
                    fresh_generations=fresh,elapsed_seconds=elapsed,
                    generations_per_second=fresh/max(elapsed,1e-6),
                    remaining_seconds=(len(selected)-done)*3/(fresh/max(elapsed,1e-6)) if fresh else None)
                atomic(OUT/f'worker{rank}/progress.json',state)
                print('PROGRESS',json.dumps(state),flush=True)
            if done % 500 == 0:
                assert sources()==f['sources'], 'Source changed during execution'
        assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==len(selected)*3
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic(OUT/f'worker{rank}/completion.json',dict(status='COMPLETE',at=now(),questions=done,
            generations=len(selected)*3,elapsed_seconds=time.monotonic()-start,
            database_sha256=sha(OUT/f'worker{rank}/results.sqlite')))
    except BaseException:
        atomic(OUT/f'worker{rank}/failure.json',dict(status='PARTIAL',at=now(),traceback=traceback.format_exc()))
        raise
    finally:
        if db:
            db.close()
        cached = visual_hashes = a = None
        gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/f'worker{rank}/release.json',dict(at=now(),pid=os.getpid(),
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            process_exit_releases_context=True))


def driver():
    freeze_check(); h = hardware()
    assert all(d['uuid'] not in h['processes'] for d in DEVICES), 'GPU0/1 must both be idle'
    atomic(OUT/'driver_preflight.json',dict(at=now(),**h))
    processes, streams = [], []
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        for rank,d in enumerate(DEVICES):
            path = OUT/f'worker{rank}/execution.log'; path.parent.mkdir(parents=True,exist_ok=True)
            stream = path.open('a'); streams.append(stream)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=d['uuid'],PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            processes.append(subprocess.Popen([sys.executable,'-u','-m',
                'vlm_research.gqa_testdev_c3.run','--rank',str(rank)],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT))
        atomic(OUT/'driver_pids.json',dict(at=now(),driver=os.getpid(),workers=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            check_stop()
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Worker exited with failure')
            state = {str(i):read(OUT/f'worker{i}/progress.json')
                for i in range(2) if (OUT/f'worker{i}/progress.json').exists()}
            print('STATUS',json.dumps(state),flush=True)
            time.sleep(30)
        codes = [p.returncode for p in processes]
        atomic(OUT/'driver_exit.json',dict(at=now(),returncodes=codes))
        assert codes == [0,0]
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        for stream in streams:
            stream.close()
        atomic(OUT/'resource_release.json',dict(at=now(),task_worker_pids=[p.pid for p in processes],
            all_workers_exited=all(p.poll() is not None for p in processes),**hardware()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver',action='store_true')
    parser.add_argument('--rank',type=int,choices=[0,1])
    args = parser.parse_args()
    if args.driver:
        driver()
    else:
        assert args.rank is not None
        worker(args.rank)
