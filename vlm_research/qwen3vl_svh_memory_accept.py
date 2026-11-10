"""Fresh-process full-model memory smoke under a separately labelled runtime."""
import argparse
import gc
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

from .qwen3vl_svh_vision_memory.run import OUT,OLD,UUID,MODEL,samples,sources
from .qwen3vl_svh_vision_memory.runtime import RUNTIME,vision_runtime
from .qwen3vl_svh_numerical_repair.engine import generate
from .qwen3vl_svh_preflight48.freeze import prepare,sha,write
from .qwen3vl_svh_preflight48.run import resource,strict_equal
from .qwen3vl_early_windows.experiment import hook_counts,tensor_digest

KEYS=['MME:landmark/0adb088e24d88123.jpg:1','MME:landmark/01121b9b3c79afc4.jpg:1','TextVQA:36759']


def freeze():
    assert not torch.cuda.is_initialized() and not (OUT/'memory_accept_protocol.json').exists()
    assert json.loads((OUT/'vision_parity.json').read_text())['status']=='PASS'
    write(OUT/'memory_accept_protocol.json',dict(runtime=RUNTIME,source_files=sources(),
        script_sha256=sha(Path(__file__)),keys=KEYS,vision_parity_sha256=sha(OUT/'vision_parity.json'),
        baseline_old_eager_stays_BLOCKED_MEMORY=True,formal_scores=False))


def gpu():
    p=json.loads((OUT/'memory_accept_protocol.json').read_text())
    assert p['source_files']==sources() and p['script_sha256']==sha(Path(__file__))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    before=resource()
    assert UUID not in before['processes']
    write(OUT/'memory_accept_gpu_before.json',before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
    hooks=hook_counts(model)
    weights=[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    rows=[]
    bindings=json.loads((OLD/'input_bindings.json').read_text())
    try:
        for key in KEYS:
            row=next(r for r in samples() if r['key']==key)
            inputs,layout=prepare(processor,row)
            assert layout==bindings[key]
            pair={}
            folder=OUT/'memory_accept'/key.replace(':','_').replace('/','_')
            assert not folder.exists()
            for name in ['Native','Noop']:
                audit=[]
                with vision_runtime(audit):
                    result,logits=generate(model,processor,inputs,layout,'N',noop=name=='Noop',audit=True,
                        capture=True,max_tokens=row['max_new_tokens'])
                assert len(audit)==27
                assert hook_counts(model)==hooks
                assert all(bool(torch.isfinite(x).all()) for x in logits)
                assert {k:tensor_digest(v) for k,v in inputs.items()}==layout['input_tensor_hashes']
                pair[name]=result,logits
                write(folder/(name+'.json'),dict(result=result,runtime=RUNTIME,vision_audit=audit))
                np.savez_compressed(folder/(name+'_logits.npz'),logits=torch.stack(logits).numpy())
                print('MEMORY ACCEPT',key,name,result['token_ids'],'peakMiB',result['peak_allocated']/2**20,flush=True)
            assert strict_equal(pair['Native'],pair['Noop'])
            entry=dict(key=key,status='PASS',runtime=RUNTIME,native_noop_equal=True,
                input_length=layout['input_length'],image_tokens=layout['image_tokens'],
                peak_allocated=max(v[0]['peak_allocated'] for v in pair.values()),
                peak_reserved=max(v[0]['peak_reserved'] for v in pair.values()),
                tokens=pair['Native'][0]['token_ids'],eos=pair['Native'][0]['eos'],
                input_tensor_hashes=layout['input_tensor_hashes'])
            rows.append(entry)
            write(folder/'validation.json',entry)
            assert weights==[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
            del pair,inputs,logits,result
            gc.collect()
        write(OUT/'memory_accept_completion.json',dict(status='PASS',runtime=RUNTIME,rows=rows,
            old_eager_status='BLOCKED_MEMORY',formal_quality_started=False,pid=os.getpid(),gpu_uuid=UUID))
    except BaseException as exc:
        write(OUT/'memory_accept_failure.json',dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),
            key=key,rows=rows,peak_allocated=torch.cuda.max_memory_allocated()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','gpu'])
    mode=parser.parse_args().mode
    freeze() if mode=='freeze' else gpu()
