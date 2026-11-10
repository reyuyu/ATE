"""Optional independent physical port: validate first, never alter selections."""
import gc
import json
import os
import time
import traceback
from collections import defaultdict

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import physical
from .engine import generate as functional
from .freeze import OUT,FIRST,check
from ..qwen3vl_overnight.common import MODEL,GPUS,TASKS,digest,sha,prepare,scorers,write,table,disk
from ..qwen3vl_overnight.store import Store
from ..qwen3vl_overnight.engine import run as baseline
from ..qwen3vl_svh_numerical_repair.engine import generate as raw_native
from ..qwen3vl_svh_sqa7908_repair.engine import generate as raw_svh
from ..qwen3vl_svh_preflight48.run import resource,strict_equal
from ..qwen3vl_svh_numerical_repair.run import compare


def worker(shard,scope='Held-out'):
    check()
    folder=OUT/('Physical' if scope=='Held-out' else 'Physical-Full')
    selected=json.loads((OUT/'final_selection.json').read_text())
    names=[n for n in dict.fromkeys([selected['quality'],selected['tradeoff'],selected['minimum_budget']]) if n!='K0']
    if scope=='Full':
        names=[selected['quality']]
        assert names[0]!='K0'
    grid=json.loads((OUT/'candidate_grid.json').read_text())
    mapping=json.loads((FIRST/'shards.json').read_text())
    probes=json.loads((OUT/'Validation_manifest.json').read_text())
    held=json.loads((OUT/(scope+'_manifest.json')).read_text())
    rows=[r for r in held if mapping[r['key']]['shard']==shard]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPUS[shard]
    assert GPUS[shard] not in resource()['processes']
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    model=store=None
    try:
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        probe=min(probes,key=lambda r:(r['input_length'],r['key']))
        pi,pl=prepare(processor,probe)
        a,av,_=baseline(model,processor,pi,pl,'Native',cap=2,capture=True)
        b,bv,_=baseline(model,processor,pi,pl,'Native',cap=2,capture=True,noop=True)
        assert strict_equal((a,av),(b,bv)), 'PHYSICAL_PORT_NATIVE_NOOP_FAILED'
        del a,av,b,bv,pi
        statuses={}
        for name in names:
            evidence=[]
            try:
                for task in TASKS:
                    row=min((r for r in probes if r['benchmark']==task),key=lambda r:(r['input_length'],r['key']))
                    inputs,layout=prepare(processor,row)
                    assert digest(layout)==row['expected_input_binding_sha256']
                    reference=functional(model,processor,inputs,layout,grid[name]['layers'],cap=8,capture=True,audit=True)
                    candidate=physical.generate(model,processor,inputs,layout,grid[name]['layers'],cap=8,capture=True,audit=True)
                    numerical=compare(*reference,*candidate)
                    assert all(e['mlp_and_norm_executed'] for e in candidate[0]['audit']['attention_events'] if 'mlp_and_norm_executed' in e)
                    assert len([e for e in candidate[0]['audit']['attention_events'] if 'mlp_and_norm_executed' in e and e.get('role')!='VISUAL_KV_BUILDER'])==36*len(candidate[0]['token_ids'])
                    evidence.append(dict(key=row['key'],structural='PASS',strict_numerical='PASS' if all(s['pass'] for s in numerical) else 'FAIL_RETAINED',
                        comparisons=numerical,kv=candidate[0]['audit']['kv'],layers=candidate[0]['audit']['layers'],
                        construction=candidate[0]['late_visual_KV_construction'],reference_tokens=reference[0]['token_ids'],
                        physical_tokens=candidate[0]['token_ids'],mode='RELAXED_NUMERICAL_QUALITY_ONLY'))
                    del reference,candidate,inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                statuses[name]='STRUCTURAL_PASS_RELAXED_QUALITY_ONLY'
            except (RuntimeError,AssertionError,ValueError) as exc:
                # This optional port cannot block or rewrite the frozen functional search.
                statuses[name]='BLOCKED_PHYSICAL_PORT'
                evidence.append(dict(error=repr(exc),traceback=traceback.format_exc(),status='FAIL_NOT_OVERRIDDEN'))
                gc.collect()
                torch.cuda.empty_cache()
            write(folder/f'smoke{shard}_{name}.json',dict(condition=name,status=statuses[name],evidence=evidence,
                native_noop_checked=True,original_svh_untouched=True))
        write(folder/f'eligible{shard}.json',statuses)
        # Both devices must independently validate a method before held-out generation.
        while not all((folder/f'eligible{s}.json').exists() for s in [0,1]):
            assert not (FIRST/'STOP_USER').exists()
            time.sleep(2)
        both=[json.loads((folder/f'eligible{s}.json').read_text()) for s in [0,1]]
        eligible=[name for name in names if all(s[name]=='STRUCTURAL_PASS_RELAXED_QUALITY_ONLY' for s in both)]
        bind=digest(dict(protocol=sha(OUT/'protocol.json'),selection=sha(OUT/'final_selection.json'),stage='Physical-'+scope,eligible=eligible))
        store=Store(folder/f'shard{shard}.sqlite',bind)
        store.plan([r['key'] for r in rows],eligible)
        scoring=scorers()
        done=0
        for row in rows:
            for name in eligible:
                if store.state(row['key'],name)!='PENDING':
                    continue
                assert not (FIRST/'STOP_USER').exists() and not (folder/'STOP_STRUCTURE.json').exists()
                disk()
                inputs,layout=prepare(processor,row)
                assert digest(layout)==row['expected_input_binding_sha256']
                store.begin(row['key'],name)
                try:
                    result,_=physical.generate(model,processor,inputs,layout,grid[name]['layers'],cap=row['max_new_tokens'])
                    record={k:v for k,v in result.items() if k!='audit'}
                    record.update(status='COMPLETE',scoring=scoring.score(row,result['text']),
                        kv=result['audit']['kv'],structural='PASS',method='PHYSICAL_SVHR_WITH_CHARGED_KV_CONSTRUCTION')
                    del result
                except torch.OutOfMemoryError as exc:
                    record=dict(status='SKIPPED_OOM',error=str(exc),traceback=traceback.format_exc(),
                        peak_allocated=torch.cuda.max_memory_allocated())
                record.update(key=row['key'],condition=name,benchmark=row['benchmark'],gold=row['gold'],
                    question_id=row['question_id'],image_sha256=row['image_sha256'],input_binding_sha256=digest(layout),
                    gpu_uuid=GPUS[shard],protocol_sha256=sha(OUT/'protocol.json'),image_tokens=row['image_tokens'],
                    strategy=row.get('strategy'),mme_task=row.get('task'),pair_id=row.get('pair_id'),circular_group=row.get('circular_group'))
                store.finish(row['key'],name,record)
                done+=1
                del inputs
                gc.collect()
                if record['status']=='SKIPPED_OOM':
                    torch.cuda.empty_cache()
                    pi,pl=prepare(processor,probe)
                    baseline(model,processor,pi,pl,'Native',cap=2)
                    del pi
                    gc.collect()
                    torch.cuda.empty_cache()
                if done%16==0:
                    state=resource()
                    assert not [line for line in state['processes'].splitlines() if GPUS[shard] in line and f', {os.getpid()},' not in line]
                    write(folder/f'progress{shard}.json',dict(counts=store.counts(),last=row['key'],time=time.time()))
        # Timing is a separate bound cap32 engineering experiment, never task-quality scoring.
        timing=[r for r in probes if mapping[r['key']]['shard']==shard] if eligible and scope=='Held-out' else []
        timing=sorted(timing,key=lambda r:digest(['SVHR_TIMING',r['key']]))[:10]
        timing_records=[]
        for row in timing:
            disk()
            inputs,layout=prepare(processor,row)
            ordered=['Native','SVH']+eligible
            for repeat in range(-3,5):
                for name in ordered[repeat%len(ordered):]+ordered[:repeat%len(ordered)]:
                    try:
                        torch.cuda.synchronize()
                        wall=time.perf_counter()
                        # The quality Guard synchronizes each layer. Exclude it
                        # from every method's timed path to avoid a false speedup.
                        if name=='Native':
                            result,_=raw_native(model,processor,inputs,layout,'N',max_tokens=32)
                        elif name=='SVH':
                            result,_=raw_svh(model,processor,inputs,layout,'C',max_tokens=32)
                        else:
                            result,_=physical.generate(model,processor,inputs,layout,grid[name]['layers'],cap=32)
                        torch.cuda.synchronize()
                        result['e2e_ms']=1000*(time.perf_counter()-wall)
                        if repeat>=0:
                            timing_records.append(dict(key=row['key'],condition=name,repeat=repeat,token_ids=result['token_ids'],
                                e2e_ms=result['e2e_ms'],ttft_ms=result['ttft_ms'],decode_ms_per_token=result['decode_ms_per_token'],
                                peak_allocated=result['peak_allocated'],generated_tokens=result['generated_tokens'],
                                live_KV_bytes=result['audit']['kv']['live_bytes'],capacity_KV_bytes=result['audit']['kv']['capacity_bytes'],
                                extra_construction_ms=result.get('late_visual_KV_construction',{}).get('extra_prefill_ms',0),
                                gpu_uuid=GPUS[shard],cap=32,vision_recomputed=True,cold_request_KV=True,
                                quality_Guard_excluded_from_all_methods=True,wall_boundary='before full generation call through synchronized return'))
                        del result
                    except torch.OutOfMemoryError as exc:
                        write(folder/f'timing_OOM_{shard}_{row["key"].replace(":","_")}_{name}.json',dict(error=str(exc)))
                        gc.collect()
                        torch.cuda.empty_cache()
                gc.collect()
            del inputs
        table(folder/f'timing_raw{shard}.csv',timing_records)
        write(folder/f'completion{shard}.json',dict(status='TERMINAL',eligible=eligible,counts=store.counts(),timed_records=len(timing_records)))
    except BaseException as exc:
        write(folder/'STOP_STRUCTURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),shard=shard))
        raise
    finally:
        if store:
            store.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(folder/f'gpu_after_inprocess{shard}.json',resource())
