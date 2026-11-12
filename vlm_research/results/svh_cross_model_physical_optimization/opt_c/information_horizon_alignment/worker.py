"""One resident model per GPU; independent atomic, hash-checked SQLite output."""
import argparse
import gc
import importlib.metadata
import shutil
import time
import traceback

from .common import *


def smoke(a,stage,rank,rows):
    import torch
    from .engine import prepare,author_generate,hf_generate
    from ..mme_seed_quality.engine import identity_attention
    selected = read(OUT/'smoke_manifest.json')
    lookup = {s['key']:s for s in rows}
    details = []
    for entry in selected:
        s = dict(lookup[entry['key']],max_new_tokens=3)
        layout,binding = prepare(a,s)
        before = digest({k:tensor_sha(v) for k,v in a.inputs.items() if isinstance(v,torch.Tensor)})
        if stage == 'A':
            native,nv = author_generate(a,s,capture=True,trace=True)
            noop,iv = author_generate(a,s,capture=True,noop=True)
            alternative,av = author_generate(a,s,capture=True,inactive_prune=True)
            assert native['token_ids'] == noop['token_ids'] == alternative['token_ids']
            assert len(nv)==len(iv)==len(av)
            assert all(torch.equal(x,y) and torch.equal(x,z) for x,y,z in zip(nv,iv,av))
            checks = dict(noop_logits_torch_equal=True,author_inactive_prune_logits_torch_equal=True,
                every_layer_full_prefill_and_decode=True,no_pruning=True,native=native)
        else:
            native,nv = hf_generate(a,layout,'HF-Native',3,capture=True)
            noop,iv = hf_generate(a,layout,'HF-Native',3,capture=True,noop=True)
            assert native['token_ids']==noop['token_ids']
            assert all(torch.equal(x,y) for x,y in zip(nv,iv))
            with torch.inference_mode():
                hf = a.model.generate(**a.inputs,do_sample=False,num_beams=1,
                    max_new_tokens=3,use_cache=True,logits_to_keep=1,
                    return_dict_in_generate=True,output_scores=True)
            assert hf.sequences[0,layout.length:].tolist()==native['token_ids']
            assert all(torch.equal(x,y.cpu()) for x,y in zip(nv,hf.scores))
            checks = dict(noop_logits_torch_equal=True,official_hf_generate_logits_torch_equal=True,
                physical=[])
            del hf
            for condition in CONDITIONS['B'][1:]:
                from ..fastp2_bypass_writer.run import operators
                from ..physical_visual_handoff.engine import equivalence
                with operators(a) as (calls,mlp,shapes):
                    actual,pv = hf_generate(a,layout,condition,3,capture=True,audit=True)
                dense,dv = hf_generate(a,layout,condition,3,capture=True,audit=True,dense=True)
                assert actual['token_ids']==dense['token_ids'], (s['key'],condition,'Dense token divergence')
                assert len(pv)==len(dv)
                equivalent=equivalence('llava',dv,pv,dense.pop('_hidden'),actual.pop('_hidden'))
                assert equivalent['passed'],(s['key'],condition,equivalent)
                for left,right in zip(actual.pop('_rotary'),dense.pop('_rotary')):
                    assert all(torch.equal(left[k],right[k]) for k in left)
                assert all(mlp[i]==len(actual['token_ids']) for i in range(32))
                assert all(calls[i,op]==1 for i in (2,3,4,5) for op in ('q_proj','k_proj','v_proj','o_proj'))
                diffs=[]
                for x,y in zip(pv,dv):
                    assert torch.allclose(x,y,atol=0.10,rtol=0.003), (condition,float((x-y).abs().max()))
                    diffs.append(float((x-y).abs().max()))
                assert actual['attention_audit'] and all(x['blocked_probability_max']==0 for x in actual['attention_audit'])
                keep = [] if condition=='Physical-C3' else list(range(11,25))
                assert actual['kv_audit']['retained_image_layers']==keep
                if keep:
                    for event in actual['visual_audit']:
                        assert event['answer_image_mass']>0 if event['layer'] in keep else event['answer_image_mass']==0
                        if event['prefill'] and event['layer']>=20:
                            assert event['relay_image_mass']==0
                # The unfused historical C3 route is a separate exact reference.
                if condition=='Physical-C3':
                    from ..mme_seed_quality.engine import greedy
                    old,ov = greedy(a,layout,'Physical-C3',capture=True)
                    assert old['token_ids'][:len(actual['token_ids'])] == actual['token_ids']
                    assert all(torch.equal(x,y.reshape(1,-1).float()) for x,y in zip(pv,ov))
                checks['physical'].append(dict(condition=condition,dense_tokens_equal=True,
                    dense_max_abs=diffs,mask_probability_zero=True,retained_image_layers=keep,
                    kv_allocations=actual['kv_audit']['allocations'],
                    historical_cache_copies=actual['kv_audit']['historical_cache_copies'],
                    numerical_equivalence=equivalent,all_MLP_executed=True,projection_shapes=shapes))
                del actual,pv,dense,dv
        assert digest({k:tensor_sha(v) for k,v in a.inputs.items() if isinstance(v,torch.Tensor)})==before
        rec=dict(key=s['key'],benchmark=s['benchmark'],binding=binding,input_unchanged=True,
                 hook_cleanup=True,checks=checks,status='PASSED',at=now())
        atomic(OUT/f'smoke/{stage}/rank{rank}/{digest(s["key"])[:16]}.json',rec)
        details.append(dict(key=s['key'],status='PASSED'))
        print('SMOKE',stage,rank,s['key'],'PASSED',flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    atomic(OUT/f'smoke/{stage}/rank{rank}/completion.json',dict(status='PASSED',cases=details,at=now()))


def run(a,stage,rank,rows,p):
    import torch
    from .engine import prepare,author_generate,hf_generate
    from .score import score
    assert all(read(OUT/f'smoke/{stage}/rank{r}/completion.json')['status']=='PASSED' for r in range(4))
    db = database(stage,rank,p)
    start = time.monotonic()
    count = 0
    for s in (s for s in rows if s['shard']==rank):
        assert not (OUT/f'STOP_{stage}.json').exists(), 'Task-owned peer stopped'
        assert shutil.disk_usage('/data').free>1024**3, '1 GiB output safety floor'
        conditions = [c for c in CONDITIONS[stage] if not prior(db,s,c)]
        if not conditions:
            continue
        layout,binding = prepare(a,s)
        input_hash = digest({k:tensor_sha(v) for k,v in a.inputs.items() if isinstance(v,torch.Tensor)})
        for condition in conditions:
            begin = time.monotonic()
            generated,_ = author_generate(a,s) if stage=='A' else hf_generate(a,layout,condition,s['max_new_tokens'])
            assert digest({k:tensor_sha(v) for k,v in a.inputs.items() if isinstance(v,torch.Tensor)})==input_hash
            generated.update(score(s,generated['text']))
            record=dict(key=s['key'],benchmark=s['benchmark'],condition=condition,stage=stage,
                question_id=s['question_id'],author_question_id=s['author_question_id'],
                image_sha256=s['image_sha256'],sample_sha256=digest(s),binding=binding,
                result=generated,rank=rank,gpu_uuid=UUIDS[rank],at=now(),
                generation_seconds=time.monotonic()-begin,protocol_sha256=digest(p))
            put(db,s,condition,record)
            count += 1
        if count%50 < len(conditions):
            atomic(OUT/f'workers/{stage}/rank{rank}/progress.json',dict(at=now(),pid=os.getpid(),
                stage=stage,rank=rank,benchmark=s['benchmark'],new_records=count,
                total_records=db.execute('SELECT COUNT(*) FROM results').fetchone()[0],
                elapsed_seconds=time.monotonic()-start,last_key=s['key']))
            print('PROGRESS',stage,rank,count,s['benchmark'],s['key'],flush=True)
    expected = sum(s['shard']==rank for s in rows)*len(CONDITIONS[stage])
    actual = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    assert actual==expected,(actual,expected)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    atomic(OUT/f'workers/{stage}/rank{rank}/completion.json',dict(status='COMPLETE',
        stage=stage,rank=rank,records=actual,at=now(),elapsed_seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('A','B'),required=True)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    stage,rank=args.stage,args.rank
    p=verify(stage)
    rows=manifest()
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
    import torch
    from .engine import load
    assert torch.cuda.device_count()==1
    tag=dict(at=now(),pid=os.getpid(),stage=stage,rank=rank,gpu_uuid=UUIDS[rank],
        torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
        protocol_sha256=digest(p),source_sha256=digest(source_hashes()))
    atomic(OUT/f'workers/{stage}/rank{rank}/startup.json',tag)
    a=None
    try:
        a=load(stage)
        smoke(a,stage,rank,rows)
        if not args.smoke_only:
            begin=time.monotonic()
            while not all((OUT/f'smoke/{stage}/rank{r}/completion.json').exists() for r in range(4)):
                assert not (OUT/f'STOP_{stage}.json').exists(),'Peer smoke failed'
                assert time.monotonic()-begin<1800,'Smoke barrier timeout'
                time.sleep(1)
            run(a,stage,rank,rows,p)
    except BaseException:
        failure=dict(**tag,status='FAILED',traceback=traceback.format_exc(),failure_at=now())
        atomic(OUT/f'workers/{stage}/rank{rank}/failure.json',failure)
        atomic(OUT/f'STOP_{stage}.json',failure)
        raise
    finally:
        del a
        gc.collect()
        torch.cuda.empty_cache()
        atomic(OUT/f'workers/{stage}/rank{rank}/release.json',dict(at=now(),pid=os.getpid(),
            gpu_uuid=UUIDS[rank],remaining_torch_allocated=torch.cuda.memory_allocated(),
            remaining_torch_reserved=torch.cuda.memory_reserved(),process_exiting=True))


if __name__=='__main__':
    main()
