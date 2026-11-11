"""One resident frozen Qwen instance per authorized GPU; private checkpoints."""
import argparse
import gc
import subprocess
import time
import traceback

from prompt_common import *


def wait_gate(name, seconds=7200):
    start = time.monotonic()
    while not (OUT / name).exists():
        if (OUT / 'STOP.json').exists():
            raise RuntimeError('Controller stopped this experiment')
        assert time.monotonic() - start < seconds, ('Gate timeout', name)
        time.sleep(2)
    return json.loads((OUT / name).read_text())


def safety(shard):
    current = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(UUIDS[shard] in line for line in current.splitlines()), current
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUIDS[shard]
    assert not subprocess.run(['pgrep', '-f', 'python.*-m vlm_research.qwen3vl_(overnight|svhr_search)'], capture_output=True).stdout
    disk()


def main(shard):
    safety(shard)
    p = check_sources()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from vlm_research.qwen3vl_overnight.engine import run
    from vlm_research.qwen3vl_overnight.common import prepare
    from vlm_research.qwen3vl_early_windows.experiment import tensor_digest, hook_counts
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rows = [r for r in json.loads((OUT / 'sample_manifest.json').read_text()) if r['shard'] == shard]
    old = Store(OUT / 'p0_reused.sqlite')
    store = Store(OUT / f'worker{shard}.sqlite')
    bindings = Store(OUT / f'bindings{shard}.sqlite')
    model = None
    records = 0
    write(OUT / f'worker{shard}_resource_start.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], free_bytes=disk(), processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)))
    try:
        processor = AutoProcessor.from_pretrained(p['model'], local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(p['model'], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        assert model.generation_config.eos_token_id == p['generation']['eos']
        hooks = hook_counts(model)

        def prepared(row, prompt):
            new, rendered, ids = render(processor, row, prompt)
            inputs, layout = prepare(processor, new)
            assert layout['prompt'] == rendered and layout['input_ids'] == ids
            baseline = old.get(row['key']+'|Native')
            assert baseline['input_binding_sha256'] == row['historical_input_binding_sha256']
            assert baseline['input_binding']['input_tensor_hashes']['pixel_values'] == layout['input_tensor_hashes']['pixel_values']
            assert baseline['input_binding']['image_grid_thw'] == layout['image_grid_thw']
            assert baseline['input_binding']['image_positions'] == layout['image_positions']
            if prompt == 'P0':
                assert layout == baseline['input_binding']
            bindings.put(row['key']+'|'+prompt, layout)
            return new, inputs, layout

        smoke = []
        for task in TASKS:
            pool = sorted([r for r in rows if r['stage'] == 'Discovery' and r['benchmark'] == task], key=lambda r: (r['input_length'],r['key']))
            row = pool[len(pool)//2]
            for prompt in ['P0', 'P1'] + (['P2'] if task != 'GQA' else []):
                _, inputs, layout = prepared(row, prompt)
                native, a, st = run(model,processor,inputs,layout,'Native',capture=True,cap=2)
                noop, b, _ = run(model,processor,inputs,layout,'Native',capture=True,noop=True,cap=2)
                assert len(a) == len(b) and all(torch.equal(x,y) for x,y in zip(a,b))
                assert native['token_ids'] == noop['token_ids']
                svh, c, structure = run(model,processor,inputs,layout,'SVH',capture=True,cap=2)
                again, d, _ = run(model,processor,inputs,layout,'SVH',capture=True,cap=2)
                assert len(c)==len(d) and all(torch.equal(x,y) for x,y in zip(c,d))
                assert svh['token_ids'] == again['token_ids'] and structure['status']=='PASS'
                assert hooks == hook_counts(model)
                smoke.append(dict(key=row['key'],task=task,prompt=prompt,native_noop='TORCH_EQUAL_PASS',
                    svh_repeat='TORCH_EQUAL_PASS', structural=structure,
                    preserved_original_numerical_status='FAIL_RETAINED', no_dense_physical_revalidation_claim=True))
                del native,noop,svh,again,a,b,c,d,inputs
        write(OUT / f'smoke{shard}.json', dict(status='PASS_INPUT_AND_WRAPPER_CHECKS', gpu_uuid=UUIDS[shard], records=smoke,
            original_algorithm_numeric_gate='FAIL_RETAINED', frozen_sources_unchanged=True))
        wait_gate('discovery_gate.json')
        for stage in ['Discovery', 'Validation']:
            choice = None if stage=='Discovery' else wait_gate('selection_frozen.json')
            todo = sorted([r for r in rows if r['stage'] == stage], key=lambda r: (r['image_sha256'],r['key']))
            expected = 0
            for row in todo:
                prompts = conditions(row['benchmark'])[1:] if stage=='Discovery' else [choice['selected'][row['benchmark']]]
                prompts = [v for v in prompts if v!='P0']
                for prompt in prompts:
                    for condition in ['Native','SVH']:
                        expected += 1
                        key = '|'.join([stage,row['key'],prompt,condition])
                        previous = store.get(key)
                        if previous:
                            assert previous['protocol_sha256'] == sha(OUT/'protocol.json')
                            continue
                        disk()
                        new, inputs, layout = prepared(row, prompt)
                        start = time.perf_counter()
                        result, _, structure = run(model, processor, inputs, layout, condition, cap=row['max_new_tokens'])
                        assert structure['status']=='PASS' and hooks == hook_counts(model)
                        assert all(tensor_digest(v)==layout['input_tensor_hashes'][k] for k,v in inputs.items())
                        record = dict(key=row['key'],stage=stage,benchmark=row['benchmark'],question_id=row['question_id'],
                            image_sha256=row['image_sha256'],circular_group=row.get('circular_group'),prompt=prompt,
                            condition=condition,gold=row['gold'],gpu_uuid=UUIDS[shard],input_binding_sha256=digest(layout),
                            protocol_sha256=sha(OUT/'protocol.json'),prompt_sha256=new['prompt_sha256'],
                            input_length=layout['input_length'],image_tokens=layout['image_tokens'],
                            elapsed_s=time.perf_counter()-start,source='NEW_GENERATION',structural=structure)
                        fields=['text','token_ids','generated_tokens','eos','truncated','empty','vision_ms','prefill_ms','ttft_ms',
                                'decode_ms','decode_ms_per_token','e2e_ms','peak_allocated','peak_reserved']
                        record.update({k: result[k] for k in fields})
                        record.update(scored(row,result['text'],result['truncated']))
                        store.put(key,record)
                        records += 1
                        if records%20==0:
                            write(OUT/f'progress{shard}.json',dict(stage=stage,new_records=records,key=key,time=time.time()))
                            print('PROGRESS',shard,stage,records,key,flush=True)
                        del result,inputs
            write(OUT/f'{stage.lower()}_complete{shard}.json',dict(status='COMPLETE',expected_new_records=expected,source_hashes_unchanged=bool(check_sources())))
        wait_gate('performance_gate.json')
        probes = json.loads((OUT/'performance_samples.json').read_text())
        performance=[]
        for task in TASKS:
            row = next(r for r in rows if r['key']==probes[f'{shard}|{task}'])
            for mode in ['natural_EOS','fixed32']:
                for repeat in range(-1,3):
                    pairs=[(prompt,condition) for prompt in conditions(task) for condition in ['Native','SVH']]
                    if repeat%2:
                        pairs.reverse()
                    for prompt,condition in pairs:
                        new,inputs,layout=prepared(row,prompt)
                        eos=model.generation_config.eos_token_id
                        try:
                            if mode=='fixed32':
                                model.generation_config.eos_token_id=[]
                            result,_,structure=run(model,processor,inputs,layout,condition,cap=32 if mode=='fixed32' else row['max_new_tokens'])
                        finally:
                            model.generation_config.eos_token_id=eos
                        if mode=='fixed32':
                            assert result['generated_tokens']==32
                        if repeat>=0:
                            fields=['vision_ms','prefill_ms','ttft_ms','decode_ms','decode_ms_per_token','e2e_ms','generated_tokens','eos','peak_allocated']
                            performance.append(dict(key=row['key'],benchmark=task,prompt=prompt,condition=condition,mode=mode,repeat=repeat,
                                gpu_uuid=UUIDS[shard],input_binding_sha256=digest(layout),token_ids=result['token_ids'],**{k:result[k] for k in fields}))
                            table(OUT/f'performance_raw{shard}.csv',performance)
                        del result,inputs
        write(OUT/f'worker_complete{shard}.json',dict(status='COMPLETE',new_generations=records,timing_records=len(performance),sources_unchanged=bool(check_sources())))
    except BaseException as exc:
        write(OUT/f'failure{shard}.json',dict(error=repr(exc),traceback=traceback.format_exc(),records=records))
        raise
    finally:
        store.conn.close()
        bindings.conn.close()
        old.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'release{shard}.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[shard],allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            context_released_on_process_exit=True,processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True,choices=[0,1])
    main(parser.parse_args().shard)
