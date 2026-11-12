"""GPU3: reproduce original vision OOM, then isolate same-value low-memory softmax."""
import argparse
import gc
import json
import os
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from .runtime import RUNTIME, row_attention, vision_runtime
from ..qwen3vl_svh_numerical_repair.run import MODEL, GATE, metrics, source_files
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_preflight48.freeze import prepare, write, sha
from ..qwen3vl_svh_preflight48.run import resource
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data/lk/ATE-worktrees/qwen3vl-svh-qk-numerical-repair')
OLD=BASE/'vlm_research/results/qwen3vl_svh_fasttrack_qk'
OUT=ROOT/'vlm_research/results/qwen3vl_svh_vision_memory'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_svh_vision_memory')
UUID='GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'
KEY='MME:landmark/0adb088e24d88123.jpg:1'


def samples():
    return [c['sample'] for c in json.loads((OLD/'numerical_cases.json').read_text())]


def sources():
    return {**source_files(),**{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}}


def freeze():
    assert not torch.cuda.is_initialized() and not (OUT/'protocol.json').exists()
    failure=json.loads((OLD/'worker0_failure.json').read_text())
    assert failure['current']['key']==KEY and 'OutOfMemoryError' in failure['error']
    row=next(r for r in samples() if r['key']==KEY)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    inputs,layout=prepare(processor,row)
    assert layout==json.loads((OLD/'input_bindings.json').read_text())[KEY]
    config=json.loads((MODEL/'config.json').read_text())['vision_config']
    patches=int(inputs['image_grid_thw'].prod(-1).sum())
    shape=[1,config['num_heads'],patches,patches]
    write(OUT/'cpu_diagnosis.json',dict(old_log_path=str(OLD/'worker0_failure.json'),old_log_sha256=sha(OLD/'worker0_failure.json'),
        row=row,layout=layout,vision_patches=patches,language_image_tokens=layout['image_tokens'],
        expected_softmax_shape=shape,fp32_matrix_bytes=4*shape[1]*patches**2,
        visual_depth=config['depth'],batch=1,old_error=failure['error'],
        old_peak_layer='UNRESOLVED_UNTIL_REPRODUCTION',runtime=RUNTIME))
    write(OUT/'protocol.json',dict(base_commit='d5ca88d7de604728eb1f0bcea3185f01d266e125',
        source_files=sources(),input_binding_sha256=sha(OUT/'cpu_diagnosis.json'),
        gpu_uuid=UUID,gate=GATE,old_runtime='VISION_EAGER_TEXT_EAGER',candidate_runtime=RUNTIME,
        no_image_or_weight_or_precision_changes=True,official_author_protocol='BLOCKED',plan='ATE_UNIFIED_HF_B'))
    print('CPU MEMORY DIAGNOSIS',shape,'FP32 bytes',4*shape[1]*patches**2,flush=True)


def diagnose():
    assert not (OUT/'oom_reproduction.json').exists()
    row=next(r for r in samples() if r['key']==KEY)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    inputs,layout=prepare(processor,row)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
    hooks=hook_counts(model)
    current={}
    captured={}
    original=qm.eager_attention_forward
    try:
        with ExitStack() as stack:
            for i,block in enumerate(model.model.visual.blocks):
                def begin(module,args,kwargs,i=i):
                    current.update(layer=i,hidden_shape=list(args[0].shape if args else kwargs['hidden_states'].shape))
                handle=block.attn.register_forward_pre_hook(begin,with_kwargs=True)
                stack.callback(handle.remove)
            def observed(module,q,k,v,attention_mask=None,**kwargs):
                if isinstance(module,qm.Qwen3VLVisionAttention):
                    current.update(query_shape=list(q.shape),key_shape=list(k.shape),value_shape=list(v.shape),
                        scaling=kwargs['scaling'],allocated_before=torch.cuda.memory_allocated(),
                        reserved_before=torch.cuda.memory_reserved(),dtype=str(q.dtype))
                    captured.update(q=q.detach().cpu().clone(),k=k.detach().cpu().clone(),v=v.detach().cpu().clone(),
                        scaling=kwargs['scaling'],layer=current['layer'])
                return original(module,q,k,v,attention_mask,**kwargs)
            stack.enter_context(patch.object(qm,'eager_attention_forward',observed))
            generate(model,processor,inputs,layout,'N',audit=False,capture=True,max_tokens=1)
        write(OUT/'oom_reproduction.json',dict(status='UNEXPECTED_FITS',current=current,peak=torch.cuda.max_memory_allocated()))
    except torch.OutOfMemoryError as exc:
        write(OUT/'oom_reproduction.json',dict(status='BLOCKED_MEMORY_ORIGINAL_EAGER',
            error=repr(exc),traceback=traceback.format_exc(),current=current,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            input_hashes=layout['input_tensor_hashes'],gpu_uuid=UUID,pid=os.getpid()))
        PRIVATE.mkdir(parents=True,exist_ok=True)
        path=PRIVATE/'actual_failing_vision_qkv.pt'
        assert not path.exists()
        torch.save(captured,path)
        write(OUT/'capture.json',dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        print('ORIGINAL OOM',json.dumps(current),'peak',torch.cuda.max_memory_allocated(),flush=True)
    finally:
        assert hook_counts(model)==hooks
        del model,inputs,captured
        gc.collect()
        torch.cuda.empty_cache()


def isolate():
    assert not (OUT/'isolated_softmax.json').exists()
    item=json.loads((OUT/'capture.json').read_text())
    assert sha(item['path'])==item['sha256']
    data=torch.load(item['path'],map_location='cpu',weights_only=True)
    q,k,v=[data[name].cuda() for name in ['q','k','v']]
    dummy=SimpleNamespace(num_key_value_groups=1,training=False)
    with torch.inference_mode():
        dense,weights=qm.eager_attention_forward(dummy,q,k,v,None,scaling=data['scaling'])
        dense_peak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        candidate,probs=row_attention(dummy,q,k,v,None,scaling=data['scaling'])
        output=metrics(dense,candidate)
        equal=torch.equal(weights,probs)
    write(OUT/'isolated_softmax.json',dict(status='PASS' if equal and output['equal'] else 'FAIL',
        runtime=RUNTIME,probabilities_torch_equal=equal,attention_output=output,
        actual_vision_layer=data['layer'],query_shape=list(q.shape),dtype=str(q.dtype),
        native_reference_no_language_model_loaded=True,reference_peak_allocated=dense_peak,
        candidate_peak_with_reference_retained=torch.cuda.max_memory_allocated(),
        note='No speed claim; candidate peak here includes retained full reference probabilities'))
    assert equal and output['equal'],'Row softmax not bitwise equal at actual failing shape'
    print('ISOLATED VISION SOFTMAX PASS',list(q.shape),flush=True)


def vision_parity():
    assert json.loads((OUT/'isolated_softmax.json').read_text())['status']=='PASS'
    assert not (OUT/'vision_parity.json').exists()
    # A vision-only GPU placement permits the otherwise OOM reference without changing its arithmetic.
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cpu'}).eval()
    visual=model.model.visual.to('cuda:0')
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    selected=[r for r in samples() if r['key']==KEY or r['key']=='MME:landmark/01121b9b3c79afc4.jpg:1']
    rows=[]
    for row in selected:
        inputs,layout=prepare(processor,row)
        px=inputs['pixel_values'].cuda().to(torch.bfloat16)
        grid=inputs['image_grid_thw'].cuda()
        with torch.inference_mode():
            reference=visual(px,grid_thw=grid)
            logs=[]
            with vision_runtime(logs):
                candidate=visual(px,grid_thw=grid)
        a=[reference[0]]+reference[1]
        b=[candidate[0]]+candidate[1]
        checks=[metrics(x,y) for x,y in zip(a,b)]
        entry=dict(key=row['key'],vision_tokens=layout['image_tokens'],outputs=checks,
            bitwise_equal=all(r['equal'] for r in checks),vision_calls=len(logs),
            input_tensor_hashes=layout['input_tensor_hashes'],language_model_on_cpu=True)
        rows.append(entry)
        print('ALL27 VISION PARITY',row['key'],entry['bitwise_equal'],flush=True)
        assert entry['bitwise_equal'],'Vision feature parity failed'
        del reference,candidate,px,grid,inputs
        gc.collect()
    write(OUT/'vision_parity.json',dict(status='PASS',runtime=RUNTIME,rows=rows))


def gpu(mode):
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_files']==sources() and p['gate']==GATE
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    state=resource()
    assert UUID not in state['processes']
    assert int(next(l for l in state['gpus'].splitlines() if UUID in l).split(',')[-2].split()[0])>23500
    write(OUT/(mode+'_gpu_before.json'),state)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    try:
        {'diagnose':diagnose,'isolate':isolate,'vision_parity':vision_parity}[mode]()
    except BaseException as exc:
        write(OUT/(mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','diagnose','isolate','vision_parity'])
    mode=parser.parse_args().mode
    freeze() if mode=='freeze' else gpu(mode)
