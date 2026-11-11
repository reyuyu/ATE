"""Two single-model resident workers, owned and stopped by the coordinator."""
import argparse
import gc
import subprocess
import time
import traceback
from common import *


def prediction_record(result, **metadata):
    value=dict(result)
    value.update(metadata)
    return value


def main(shard,smoke_only=False):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from vlm_research.qwen3vl_overnight.engine import run as frozen_run
    from engine import Request,ACTIVE
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[shard]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(UUIDS[shard] in x for x in processes.splitlines())
    p=check()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model=store=bindings=None
    write(OUT/f'worker{shard}_start.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[shard],time=time.time(),processes=processes))
    try:
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        rows=json.loads((OUT/'sample_manifest.json').read_text())
        store=Store(OUT/f'worker{shard}.sqlite')
        bindings=Store(OUT/f'bindings{shard}.sqlite')
        hist=Store(OUT/'historical.sqlite')
        def bound(row):
            inputs,layout=prepare(processor,row)
            assert hashlib.sha256(layout['prompt'].encode()).hexdigest()==row['prompt_sha256']
            assert sha(row['image_path'])==row['image_sha256']
            for cond in ['Native','SVH']:
                old=hist.get(row['key']+'|'+cond)
                assert old['input_binding_sha256']==digest(layout),(row['key'],cond,'INPUT_BINDING_MISMATCH')
            bindings.put(row['key'],layout)
            return inputs,layout
        smoke=[]
        for task in ['TextVQA','ScienceQA-IMG','MMBench']:
            pool=sorted([r for r in rows if r['split']=='Discovery' and r['shard']==shard and r['analysis_task']==task],key=lambda r:r['input_length'])
            row=pool[len(pool)//2]
            inputs,layout=bound(row)
            native,a,_=frozen_run(model,processor,inputs,layout,'Native',capture=True,cap=2)
            noop,b,_=frozen_run(model,processor,inputs,layout,'Native',capture=True,noop=True,cap=2)
            assert native['token_ids']==noop['token_ids'] and all(torch.equal(x,y) for x,y in zip(a,b))
            frozen,a,_=frozen_run(model,processor,inputs,layout,'SVH',capture=True,cap=3)
            with Request(model,processor,inputs,layout,collect=ACTIVE,audit=True,cap=3) as request:
                zero,b=request.generate([],capture=True)
                w14,c=request.generate(list(range(11,25)),capture=True)
                w11,d=request.generate(list(range(11,22)),capture=True)
                assert torch.equal(a[0],b[0]) and torch.equal(b[0],c[0]) and torch.equal(c[0],d[0])
                assert zero['token_ids']==frozen['token_ids']
                # Same full QK extent: real masked visual columns replace the frozen temporary zeros.
                errors=[float((x-y).abs().max()) for x,y in zip(a,b)]
                assert len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b)),('K0_NOT_EXACT_FROZEN',errors)
                phy,e=request.generate(list(range(11,22)),physical=True,capture=True)
                assert w11['token_ids']==phy['token_ids'] and len(d)==len(e)
                strict=max(float((x-y).abs().max()) for x,y in zip(d,e))
                assert strict<=0.02,('NEW_DECODE_PHYSICAL_GATE',strict)
                smoke.append(dict(key=row['key'],native_noop='TORCH_EQUAL',main_prefix='TORCH_EQUAL_FROZEN_SVH',
                    frozen_K0_all_steps='TORCH_EQUAL',physical_decode_max_logit_error=strict,
                    mask=phy['permissions'],late_builder=w14['temporary_sidecar'],historical_strict_gate='FAIL_RETAINED'))
                non_target=zero['prefill_non_target_hashes']
            with Request(model,processor,inputs,layout,collect=ACTIVE,prefill_layers=list(range(11,25)),audit=True,cap=3) as request:
                unified,uv=request.generate(list(range(11,25)),capture=True)
                unified_physical,pv=request.generate(list(range(11,25)),physical=True,capture=True)
                assert unified['prefill_non_target_hashes']==non_target,'NON_TARGET_PREFILL_CHANGED'
                assert unified['token_ids']==unified_physical['token_ids']
                uerror=max(float((x-y).abs().max()) for x,y in zip(uv,pv))
                assert uerror<=0.02,('UNIFIED_DECODE_PORT_ERROR',uerror)
                smoke[-1].update(unified_non_target_prefill='TORCH_EQUAL',unified_physical_decode_error=uerror,
                    unified_prefill_layers=unified['prefill_layers'])
            del inputs,native,noop,frozen,a,b,c,d,e
        write(OUT/f'smoke{shard}.json',dict(status='PASS_NEW_PATH_ONLY',cases=smoke,gpu_uuid=UUIDS[shard],time=time.time()))
        print('SMOKE_PASS',shard,flush=True)
        if smoke_only:
            return
        while not (OUT/'STOP').exists():
            jobs=sorted((OUT/'jobs').glob('*.json')) if (OUT/'jobs').exists() else []
            pending=[j for j in jobs if not (OUT/'jobs'/f'{j.stem}.done{shard}').exists()]
            if not pending:
                time.sleep(.5)
                continue
            job_path=pending[0]
            job=json.loads(job_path.read_text())
            check()
            chosen={r['key'] for r in job.get('rows',[])}
            pool=[r for r in rows if r['key'] in chosen and r['shard']==shard]
            if job.get('kind')=='STOP':
                break
            if job.get('kind')=='PERFORMANCE':
                assert sha(ROOT/'performance.py')==job['performance_source_sha256']
                from performance import execute
                execute(model,processor,pool,job,shard,bound)
                write(OUT/'jobs'/f'{job_path.stem}.done{shard}',dict(job=job_path.name,time=time.time()))
                continue
            done=0
            for row in sorted(pool,key=lambda r:(r['image_sha256'],r['key'])):
                if (OUT/'STOP').exists():
                    break
                disk()
                todo=[]
                for name,layers in job['conditions'].items():
                    key=job['stage']+'|'+job['view']+'|'+row['key']+'|'+name
                    if store.get(key) is None:
                        todo.append((key,name,layers))
                if not todo:
                    continue
                if row.get('historical_blocked_status'):
                    for key,name,layers in todo:
                        store.put(key,dict(key=row['key'],condition=name,stage=job['stage'],view=job['view'],
                            benchmark=row['benchmark'],analysis_task=row['analysis_task'],image_sha256=row['image_sha256'],
                            circular_group=row.get('circular_group'),status='BLOCKED_HISTORICAL_OOM',
                            original_status=row['historical_blocked_status'],layers=layers,
                            original_visual_budget_preserved=True))
                    continue
                write(OUT/f'current{shard}.json',dict(job=job_path.name,key=row['key'],conditions=[x[1] for x in todo],time=time.time()))
                inputs,layout=bound(row)
                for key,name,layers in todo:
                    prefill=layers if job['view']=='Unified' else layers if job['view']=='P1Only' else job.get('prefill_layers',[])
                    decode=[] if job['view']=='P1Only' else layers
                    with Request(model,processor,inputs,layout,collect=ACTIVE,prefill_layers=prefill,cap=row['max_new_tokens']) as request:
                        result,_=request.generate(decode)
                        value=prediction_record(result,key=row['key'],condition=name,stage=job['stage'],view=job['view'],
                            benchmark=row['benchmark'],analysis_task=row['analysis_task'],image_sha256=row['image_sha256'],
                            circular_group=row.get('circular_group'),question=row['question'],gold=row['gold'],
                            mme_task=row.get('task'),pair_id=row.get('pair_id'),strategy=row.get('strategy'),
                            gpu_uuid=UUIDS[shard],protocol_sha256=sha(OUT/'protocol.json'),input_binding_sha256=digest(layout),
                            prompt_sha256=row['prompt_sha256'],layers=layers,prefill_layers=prefill)
                        value.update(scored(row,result['text'],result['truncated']))
                        old=hist.get(row['key']+'|SVH')
                        value['first_token_flip_vs_svh']=result['token_ids'][0]!=old['token_ids'][0]
                        if job['view']=='DecodeOnly':
                            assert not value['first_token_flip_vs_svh'],'MAIN_VIEW_CHANGED_FIRST_TOKEN'
                        store.put(key,value)
                        done+=1
                del inputs
                if done%20==0:
                    print('PROGRESS',shard,job['stage'],done,row['key'],flush=True)
            write(OUT/'jobs'/f'{job_path.stem}.done{shard}',dict(job=job_path.name,new_records=done,time=time.time()))
            print('JOB_DONE',shard,job_path.name,done,flush=True)
    except BaseException as exc:
        write(OUT/f'failure{shard}.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        for db in [store,bindings]:
            if db is not None:
                db.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'release{shard}.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[shard],time=time.time(),
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),process_exit_releases_context=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    main(args.shard,args.smoke_only)
