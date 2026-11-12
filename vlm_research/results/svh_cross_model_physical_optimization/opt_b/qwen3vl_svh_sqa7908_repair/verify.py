"""Frozen candidate and independent full-generation regressions."""
import argparse
import gc
import json
import os
import time
import traceback
from contextlib import ExitStack

import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from .run import OUT,OLD,UUID,KEY,MODEL,GATE,sources,samples
from .engine import generate as physical
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_numerical_repair.run import compare
from ..qwen3vl_svh_l28_repair.run import OperatorTrace,compare_records
from ..qwen3vl_svh_preflight48.freeze import prepare,sha,write,table
from ..qwen3vl_svh_preflight48.run import resource,CompactTrace,compare_traces,validate_semantics,strict_equal
from ..qwen3vl_early_windows.experiment import hook_counts,tensor_digest


def freeze():
    assert not torch.cuda.is_initialized() and not (OUT/'repair_protocol.json').exists()
    assert json.loads((OUT/'isolation_completion.json').read_text())['status']=='PASS'
    write(OUT/'repair_protocol.json',dict(source_files=sources(),gate=GATE,gpu_uuid=UUID,
        isolation_sha256=sha(OUT/'isolation_completion.json'),scope='Prefill L1-L6 physically compact Q/K/V/O descriptor only',
        no_FP32_inference=True,no_extra_P1_projection=True,inputs_unchanged=True,
        samples_sha256=sha(OUT/'sample_manifest.json'),old_vision_runtime='eager'))


def execute(mode,model,processor):
    folder=OUT/('fixed_probe' if mode=='probe' else 'independent_verification')
    assert not folder.exists()
    assert os.getpid()!=json.loads((OUT/'diagnosis_completion.json').read_text())['pid']
    hooks=hook_counts(model)
    eager,matmul=qm.eager_attention_forward,torch.matmul
    parameters=[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    results=[]
    for row in samples()[:1] if mode=='probe' else samples():
        inputs,layout=prepare(processor,row)
        assert layout==json.loads((OLD/'input_bindings.json').read_text())[row['key']]
        case=folder/row['key'].replace(':','_').replace('/','_')
        records,traces={},{}
        for name in ['Dense','Physical'] if mode=='probe' else ['Native','Noop','Dense','Physical']:
            trace=(OperatorTrace if mode=='probe' else CompactTrace)(model,layout) if name in ['Dense','Physical'] else None
            with ExitStack() as stack:
                if trace:
                    stack.enter_context(trace.context())
                fn=physical if name=='Physical' else generate
                result,logits=fn(model,processor,inputs,layout,'C' if name=='Physical' else 'F' if name=='Dense' else 'N',
                    audit=True,capture=True,max_tokens=1 if mode=='probe' else row['max_new_tokens'],
                    **({'noop':name=='Noop'} if name!='Physical' else {}))
            assert hook_counts(model)==hooks and qm.eager_attention_forward is eager and torch.matmul is matmul
            assert all(bool(torch.isfinite(x).all()) for x in logits)
            assert {k:tensor_digest(v) for k,v in inputs.items()}==layout['input_tensor_hashes']
            records[name]=result,logits
            if trace:
                traces[name]=trace
            write(case/(name+'.json'),result)
            np.savez_compressed(case/(name+'_logits.npz'),logits=torch.stack(logits).numpy(),token_ids=result['token_ids'])
            print(mode,row['key'],name,len(result['token_ids']),flush=True)
        comparison=compare(*records['Dense'],*records['Physical'])
        table(case/'logits_comparison.csv',comparison)
        if mode=='probe':
            operators=compare_records(traces['Dense'],traces['Physical'])
            table(case/'operators.csv',operators)
            extra=dict(first=next((r for r in operators if not r['equal'] and r['operator']!='mask'),None))
            write(case/'first_difference.json',extra)
            semantic=True
        else:
            assert strict_equal(records['Native'],records['Noop'])
            layers,semantics=compare_traces(traces['Dense'],traces['Physical'])
            table(case/'layer_comparison.csv',layers)
            write(case/'semantics.json',semantics)
            structural=validate_semantics(records['Physical'][0],layout)
            assert len(records['Physical'][0]['audit']['early_aligned_linears'])==24
            semantic=all(semantics[k] for k in ['all_layer_steps_captured','all_mrope_positions_equal','deepstack_equal','cache_positions_equal'])
            extra=dict(structural=structural,real_decode_steps=len(comparison)-1)
        passed=bool(comparison) and all(r['pass'] for r in comparison) and semantic
        entry=dict(key=row['key'],status='PASS' if passed else 'FAIL',
            mean_abs=max(r['mean_abs'] for r in comparison),max_abs=max(r['max_abs'] for r in comparison),
            tokens_equal=records['Dense'][0]['token_ids']==records['Physical'][0]['token_ids'],**extra)
        write(case/'validation.json',entry)
        results.append(entry)
        assert parameters==[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
        print('FIXED',json.dumps(entry),flush=True)
        if not passed:
            write(OUT/(mode+'_completion.json'),dict(status='FAIL',pid=os.getpid(),rows=results))
            raise RuntimeError('Candidate numerical failure; no evaluation unlock')
        del records,traces,inputs,result,logits
        gc.collect()
    write(OUT/(mode+'_completion.json'),dict(status='PASS',pid=os.getpid(),rows=results,
        gate=GATE,completed=len(results),runtime='VISION_EAGER_TEXT_EAGER',formal_quality_started=False))


def gpu(mode):
    p=json.loads((OUT/'repair_protocol.json').read_text())
    assert p['source_files']==sources() and p['gate']==GATE
    if mode=='verify':
        assert json.loads((OUT/'probe_completion.json').read_text())['status']=='PASS'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    before=resource()
    assert UUID not in before['processes']
    write(OUT/(mode+'_gpu_before.json'),before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    try:
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        execute(mode,model,processor)
    except BaseException as exc:
        write(OUT/(mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','probe','verify'])
    mode=parser.parse_args().mode
    freeze() if mode=='freeze' else gpu(mode)
