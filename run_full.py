"""Two independent resident workers; same-image methods paired on one GPU."""
import argparse
import gc
import subprocess
import time
import traceback
from full_common import *


def main(shard):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(UUIDS[shard] in line for line in processes.splitlines()),processes
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[shard]
    assert json.loads((OUT/'cpu_ready.json').read_text())['status']=='PASS'
    p=check()
    disk()
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from vlm_research.qwen3vl_overnight.engine import run
    from vlm_research.qwen3vl_overnight.common import prepare
    from vlm_research.qwen3vl_early_windows.experiment import hook_counts,tensor_digest
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rows=[r for r in json.loads((OUT/'sample_manifest.json').read_text()) if r['shard']==shard]
    store=Store(OUT/f'worker{shard}.sqlite')
    bindings=Store(OUT/f'bindings{shard}.sqlite')
    model=None
    start=time.time()
    count=0
    write(OUT/f'worker{shard}_start.json',dict(pid=os.getpid(),time=start,gpu_uuid=UUIDS[shard],free_bytes=disk(),processes=processes))
    try:
        processor=AutoProcessor.from_pretrained(p['model'],local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(p['model'],local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        assert model.generation_config.eos_token_id==p['generation']['eos']
        hooks=hook_counts(model)

        def prepared(row):
            new,text,ids=render(processor,row,p['suffix_exact_from_prior'])
            inputs,layout=prepare(processor,new)
            assert layout['prompt']==text and layout['input_ids']==ids
            bindings.put(row['key'],layout)
            return new,inputs,layout

        smoke=[]
        for task in TASKS:
            pool=sorted([r for r in rows if r['benchmark']==task],key=lambda r:(r['input_length'],r['key']))
            row=pool[len(pool)//2]
            new,inputs,layout=prepared(row)
            n,a,_=run(model,processor,inputs,layout,'Native',capture=True,cap=2)
            noop,b,_=run(model,processor,inputs,layout,'Native',capture=True,noop=True,cap=2)
            assert len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))
            assert n['token_ids']==noop['token_ids']
            s,c,st=run(model,processor,inputs,layout,'SVH',capture=True,cap=2)
            again,d,_=run(model,processor,inputs,layout,'SVH',capture=True,cap=2)
            assert len(c)==len(d) and all(torch.equal(x,y) for x,y in zip(c,d))
            assert s['token_ids']==again['token_ids'] and hooks==hook_counts(model)
            smoke.append(dict(key=row['key'],task=task,native_noop='TORCH_EQUAL',svh_repeat='TORCH_EQUAL',structural=st,
                              strict_dense_physical_gate='FAIL_RETAINED'))
            del inputs,a,b,c,d,n,noop,s,again
        write(OUT/f'smoke{shard}.json',dict(status='PASS_WRAPPER_SMOKE_ONLY',cases=smoke,gpu_uuid=UUIDS[shard]))
        # Both authorized cards must pass the same limited smoke before full quality generation.
        wait_start=time.monotonic()
        while not all((OUT/f'smoke{i}.json').exists() for i in [0,1]):
            assert not any((OUT/f'failure{i}.json').exists() for i in [0,1])
            assert time.monotonic()-wait_start<1800
            time.sleep(2)
        for row in sorted(rows,key=lambda r:(r['benchmark'],r['image_sha256'],r['key'])):
            if row['reuse_validation']:
                continue
            todo=[cond for cond in ['Native','SVH'] if store.get(row['key']+'|'+cond) is None]
            if not todo:
                for cond in ['Native','SVH']:
                    assert store.get(row['key']+'|'+cond)['protocol_sha256']==sha(OUT/'p2_full_protocol.json')
                continue
            disk()
            if count%100==0:
                check()
            write(OUT/f'current{shard}.json',dict(key=row['key'],conditions=todo,time=time.time()))
            try:
                new,inputs,layout=prepared(row)
                for cond in todo:
                    t=time.perf_counter()
                    result,_,structure=run(model,processor,inputs,layout,cond,cap=row['max_new_tokens'])
                    assert structure['status']=='PASS' and hook_counts(model)==hooks
                    record=dict(key=row['key'],benchmark=row['benchmark'],question_id=row['question_id'],
                        circular_group=row.get('circular_group'),image_sha256=row['image_sha256'],gold=row['gold'],
                        prompt='P2',condition=cond,gpu_uuid=UUIDS[shard],input_binding_sha256=digest(layout),
                        protocol_sha256=sha(OUT/'p2_full_protocol.json'),prompt_sha256=new['prompt_sha256'],
                        input_length=layout['input_length'],image_tokens=layout['image_tokens'],
                        source='NEW_GENERATION',state='COMPLETE',elapsed_s=time.perf_counter()-t,structural=structure)
                    fields=['text','token_ids','generated_tokens','eos','truncated','empty','vision_ms','prefill_ms','ttft_ms',
                        'decode_ms','decode_ms_per_token','e2e_ms','peak_allocated','peak_reserved']
                    record.update({k:result[k] for k in fields})
                    record.update(scored(row,result['text'],result['truncated']))
                    store.put(row['key']+'|'+cond,record)
                    count+=1
                    if count%40==0:
                        write(OUT/f'progress{shard}.json',dict(new_records=count,key=row['key'],time=time.time(),elapsed_s=time.time()-start))
                        print('PROGRESS',shard,count,row['key'],round(time.time()-start,1),flush=True)
                    del result
                del inputs
            except torch.cuda.OutOfMemoryError as exc:
                # Preserve planned denominator and never treat incomplete Circular groups as complete.
                for cond in todo:
                    if store.get(row['key']+'|'+cond) is None:
                        store.put(row['key']+'|'+cond,dict(key=row['key'],benchmark=row['benchmark'],condition=cond,
                            state='SKIPPED_OOM',error=str(exc),image_sha256=row['image_sha256'],
                            protocol_sha256=sha(OUT/'p2_full_protocol.json')))
                if 'inputs' in locals():
                    del inputs
                gc.collect()
                torch.cuda.empty_cache()
                print('OOM_RECORDED',row['key'],flush=True)
        write(OUT/f'complete{shard}.json',dict(status='COMPLETE_TERMINAL_RECORDS',new_records=count,time=time.time(),elapsed_s=time.time()-start,sources_unchanged=bool(check())))
    except BaseException as exc:
        write(OUT/f'failure{shard}.json',dict(error=repr(exc),traceback=traceback.format_exc(),new_records=count))
        raise
    finally:
        store.conn.close()
        bindings.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'release{shard}.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[shard],allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),time=time.time(),process_exit_releases_cuda_context=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,choices=[0,1],required=True)
    main(parser.parse_args().shard)
