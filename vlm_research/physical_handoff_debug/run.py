"""Bounded diagnosis, unchanged gates, ten fresh regression questions, and real microbenchmarks."""
import argparse
import gc
import os
import subprocess
import time
import traceback

import torch

from ..physical_visual_handoff.engine import Adapter, run
from .common import OUT, HISTORY, atomic, read, key, prepare, verify, digest, write_csv
from .diagnose import diagnose, compare_runs
from .stable import numerical_kernels


def worker(kind):
    start = time.time()
    frozen = verify()
    folder = OUT / kind
    manifest = read(folder/'manifest.json')
    specs = read(folder/'input_bindings.json')
    baseline = read(HISTORY/kind/'baselines.json')
    adapter = Adapter(kind)
    atomic(folder/'environment.json', dict(adapter.environment, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
           cuda_name=torch.cuda.get_device_name(), model=adapter.model.config._name_or_path,
           gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                         '--format=csv,noheader'], text=True), start_time=start))

    def execute(s, condition, candidate='native', **kw):
        adapter.prepare(s, specs[key(s)])
        with numerical_kernels(adapter, candidate):
            result = run(adapter, s, specs[key(s)], condition, **kw)
        adapter.verify()
        return result

    # Gate the unmodified same-shape execution before any shape-changing diagnostics.
    s = manifest['smoke'][0]
    a, al, _ = execute(s, 'N0', native=True, capture=True)
    b, bl, _ = execute(s, 'N0', capture=True)
    assert a['token_ids'] == baseline[key(s)]['token_ids'] == b['token_ids']
    assert a['all_valid_logits_sha256'] == b['all_valid_logits_sha256']
    assert all(torch.equal(x, y) for x, y in zip(al, bl))
    atomic(folder/'native_noop.json', dict(status='PASS',question=key(s),all_valid_logits_exact=True,
                                         historical_native_tokens_exact=True))
    print('NATIVE_NOOP', kind, 'PASS', flush=True)
    diagnose(adapter, manifest, specs)
    print('OPERATOR_DIAGNOSIS', kind, read(folder/'diagnosis_complete.json'), flush=True)
    del a, b, al, bl

    if kind == 'llava':
        rows = []
        for s in manifest['historical43']:
            a, al, ah = execute(s, 'D1', max_tokens=1, audit=True, capture=True)
            b, bl, bh = execute(s, 'P1', max_tokens=1, audit=True, capture=True)
            proof = compare_runs(kind,a,b,al,bl,ah,bh)
            rows.append(dict(question=key(s), image_sha256=s['image_sha256'], **proof,
                             decision_risk=proof['dense_top1_top2_margin']<=2*proof['max_abs_logits']))
            write_csv(folder/'historical43_prefill_audit.csv',rows)
        print('HISTORICAL43', kind, 'COMPLETE', flush=True)

    registry = []
    selected = None
    for candidate in frozen['kernel_candidate_order']:
        passed = True
        for s in manifest['smoke']:
            a, al, ah = execute(s, 'D1', candidate, audit=True, capture=True)
            b, bl, bh = execute(s, 'P1', candidate, audit=True, capture=True)
            proof = compare_runs(kind,a,b,al,bl,ah,bh)
            row = dict(model=kind, candidate=candidate, phase='frozen_smoke',question=key(s), **proof)
            registry.append(row)
            atomic(folder/'candidate_registry.json',registry)
            print('CANDIDATE', kind, candidate, key(s), proof['overall_pass'],
                  proof['max_abs_logits'],proof['hidden_relative_rmse'], flush=True)
            if not proof['overall_pass']:
                passed = False
                break
        if passed:
            selected = candidate
            break
    atomic(folder/'selected_candidate.json',dict(candidate=selected,smoke_passed=selected is not None))

    regression = []
    kv_rows = []
    valid = selected is not None
    if selected is not None:
        for phase,samples in [('smoke_P2',manifest['smoke']),('new10',manifest['new10'])]:
            for s in samples:
                paired = {}
                for c in ('D1','P1','P2'):
                    a, al, ah = execute(s,c,selected,audit=True,capture=True)
                    paired[c] = (a,al,ah)
                    cache = a['cache']
                    assert cache['old_tensor_references_alive'] == 0
                    if c == 'P2':
                        image = set(specs[key(s)]['visual_positions'] if kind=='llava'
                                    else specs[key(s)]['layout']['image_positions'])
                        assert all(not image.intersection(r['original_key_positions']) for r in cache['layers'])
                    kv_rows.append(dict(model=kind,phase=phase,question=key(s),condition=c,
                                        live_kv_bytes=cache['live_kv_bytes'],
                                        old_tensor_references_alive=cache['old_tensor_references_alive'],
                                        cache=cache,execution=a['execution'],mask_checks=len(a['mask_events'])))
                    atomic(folder/'checkpoints'/phase/key(s)/(c+'.json'), dict(
                        protocol_sha256=digest(frozen), sample_sha256=digest(s),generation=a))
                for c in ('P1','P2'):
                    a,al,ah=paired['D1'];b,bl,bh=paired[c]
                    proof=compare_runs(kind,a,b,al,bl,ah,bh)
                    regression.append(dict(model=kind,candidate=selected,phase=phase,question=key(s),physical=c,**proof))
                    valid &= proof['overall_pass']
                atomic(folder/'regression.json',regression)
                atomic(folder/'kv_audit.json',kv_rows)
                print('REGRESSION',kind,phase,key(s),valid,flush=True)
                if not valid:
                    break
            if not valid:
                break
    atomic(folder/'quality_gate.json', dict(status='PASS' if valid else 'FAILED',candidate=selected,
           regression_records=len(regression),fresh_questions=len({r['question'] for r in regression if r['phase']=='new10'}),
           historical_partial_unchanged=True))
    del al, bl, ah, bh
    gc.collect();torch.cuda.empty_cache()

    # Bounded diagnostics remain useful even if equivalence fails: never label them validated speedups.
    s=manifest['smoke'][0]
    timings=[]
    candidates=list(dict.fromkeys(['native']+([selected] if selected and selected!='native' else [])))
    for candidate in candidates:
        for length in frozen['benchmark']['fixed_generated_lengths']:
            for c in ('D1','P1','P2'):
                gc.collect();torch.cuda.empty_cache()
                for rep in range(frozen['benchmark']['warmups']+frozen['benchmark']['repeats']):
                    a,_,_=execute(s,c,candidate,max_tokens=length,fixed=True)
                    if rep>=frozen['benchmark']['warmups']:
                        timings.append(dict(model=kind,kernel=candidate,condition=c,fixed_tokens=length,
                            repetition=rep-frozen['benchmark']['warmups'],
                            classification='SMALL_SAMPLE_VALIDATED_PERFORMANCE' if valid and candidate==selected else 'UNVALIDATED_PERFORMANCE',
                            vision_ms=a['vision_ms'],language_prefill_ms=a['language_prefill_ms'],ttft_ms=a['ttft_ms'],
                            decode_ms_per_token=sum(a['decode_ms'])/len(a['decode_ms']),end_to_end_ms=a['latency_ms'],
                            live_kv_bytes=a['cache']['live_kv_bytes'],peak_allocated=a['peak_allocated'],
                            peak_reserved=a['peak_reserved'],initial_allocated=a['initial_allocated'],
                            token_ids=a['token_ids'],question=key(s)))
                        write_csv(folder/'microbenchmark.csv',timings)
                print('BENCHMARK',kind,candidate,length,c,flush=True)
    adapter.verify();verify()
    atomic(folder/'completion.json',dict(status='COMPLETE',correctness='PASS' if valid else 'FAILED',
           total_seconds=time.time()-start,selected_candidate=selected,
           scope='Bounded numerical diagnosis, not 60/240 quality validation',
           no_leftover_hooks=True,parameters_and_inputs_unchanged=True))
    del adapter
    gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',kind,round(time.time()-start,2),'seconds',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker']);p.add_argument('--kind',choices=['llava','qwen'])
    args=p.parse_args()
    try:
        if args.stage=='prepare':prepare()
        else:worker(args.kind)
    except BaseException:
        atomic(OUT/(args.kind or 'preflight')/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':main()
