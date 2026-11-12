"""GPU1: capture actual L0/L1 states, isolate the early projection shape."""
import argparse
import gc
import json
import os
import traceback
from contextlib import ExitStack,contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

from ..qwen3vl_svh_qk_repair.engine import generate as physical_generate
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_numerical_repair.run import GATE,MODEL,metrics,compare,source_files
from ..qwen3vl_svh_l28_repair.run import OperatorTrace,compare_records
from ..qwen3vl_svh_preflight48.freeze import prepare,sha,write,table
from ..qwen3vl_svh_preflight48.run import resource,strict_equal
from ..qwen3vl_early_windows.experiment import hook_counts,tensor_digest
from ..qwen3vl_c3v2_l22_kernel_diagnosis.candidates import LtPlan

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data/lk/ATE-worktrees/qwen3vl-svh-qk-numerical-repair')
OLD=BASE/'vlm_research/results/qwen3vl_svh_fasttrack_qk'
OUT=ROOT/'vlm_research/results/qwen3vl_svh_sqa7908_repair'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_svh_sqa7908_repair')
UUID='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
KEY='ScienceQA-IMG:7908'


def sources():
    return {**source_files(),**{str(p.relative_to(ROOT)):sha(p) for d in
        ['qwen3vl_svh_qk_repair','qwen3vl_svh_l28_repair','qwen3vl_svh_sqa7908_repair']
        for p in (ROOT/'vlm_research'/d).glob('*.py')}}


def samples():
    cases={c['sample']['key']:c['sample'] for c in json.loads((OLD/'numerical_cases.json').read_text())}
    validation=json.loads((OLD/'numerical_completion.json').read_text())
    regressions=[cases[r['key']] for r in validation['rows'] if r['status']=='PASS']
    assert len(regressions)==13 and KEY not in [r['key'] for r in regressions]
    return [cases[KEY]]+regressions


def freeze():
    assert not torch.cuda.is_initialized() and not (OUT/'diagnostic_protocol.json').exists()
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    bindings=json.loads((OLD/'input_bindings.json').read_text())
    for row in samples():
        inputs,layout=prepare(processor,row)
        assert layout==bindings[row['key']]
        del inputs
    write(OUT/'sample_manifest.json',samples())
    write(OUT/'diagnostic_protocol.json',dict(base_commit='d5ca88d7de604728eb1f0bcea3185f01d266e125',
        source_files=sources(),gpu_uuid=UUID,gate=GATE,model_revision=MODEL.name,
        input_bindings_path=str(OLD/'input_bindings.json'),input_bindings_sha256=sha(OLD/'input_bindings.json'),
        samples_sha256=sha(OUT/'sample_manifest.json'),dtype='bfloat16',attention='eager',
        no_quality_evaluation=True,source_failure_sha256=sha(OLD/'numerical/ScienceQA-IMG_7908/validation.json')))
    print('CPU FROZEN',len(samples()),flush=True)


class Capture(OperatorTrace):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.actual={}

    @contextmanager
    def context(self):
        with super().context(),ExitStack() as stack:
            def put(name,value):
                if self.step==0:
                    self.actual[name]=value.detach().cpu().clone()
            h=self.d.layers[0].register_forward_hook(lambda m,a,o:put('L0_output',o))
            stack.callback(h.remove)
            h=self.d.layers[1].register_forward_pre_hook(lambda m,a,k:put('L1_input',a[0]),with_kwargs=True)
            stack.callback(h.remove)
            norm=self.d.layers[1].input_layernorm
            h=norm.register_forward_pre_hook(lambda m,a:put('L1_norm_input',a[0]))
            stack.callback(h.remove)
            h=norm.register_forward_hook(lambda m,a,o:put('L1_norm_output',o))
            stack.callback(h.remove)
            for i in range(1,7):
                for name in ['q_proj','k_proj','v_proj','o_proj']:
                    mod=getattr(self.d.layers[i].self_attn,name)
                    h=mod.register_forward_pre_hook(lambda m,a,i=i,name=name:put(f'L{i}_{name}_input',a[0]))
                    stack.callback(h.remove)
                    h=mod.register_forward_hook(lambda m,a,o,i=i,name=name:put(f'L{i}_{name}_output',o))
                    stack.callback(h.remove)
            yield self


def diagnose():
    assert not (OUT/'diagnosis_completion.json').exists()
    row=samples()[0]
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    inputs,layout=prepare(processor,row)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
    hooks=hook_counts(model)
    records,traces={},{}
    for name in ['Native','Noop','Dense','Physical']:
        trace=Capture(model,layout) if name in ['Dense','Physical'] else None
        with ExitStack() as stack:
            if trace:
                stack.enter_context(trace.context())
            fn=physical_generate if name=='Physical' else generate
            result,vectors=fn(model,processor,inputs,layout,'C' if name=='Physical' else 'F' if name=='Dense' else 'N',
                audit=True,capture=True,max_tokens=1,**({'noop':name=='Noop'} if name!='Physical' else {}))
        assert hook_counts(model)==hooks
        assert {k:tensor_digest(v) for k,v in inputs.items()}==layout['input_tensor_hashes']
        records[name]=result,vectors
        if trace:
            traces[name]=trace
        write(OUT/'diagnosis'/(name+'.json'),result)
        np.savez_compressed(OUT/'diagnosis'/(name+'_logits.npz'),logits=torch.stack(vectors).numpy())
        with np.load(OLD/'numerical/ScienceQA-IMG_7908'/(name+'_logits.npz')) as old:
            assert np.array_equal(old['logits'][0],vectors[0].numpy()),'Failure did not exactly reproduce'
        print('DIAGNOSE',name,result['token_ids'],flush=True)
    assert strict_equal(records['Native'],records['Noop'])
    operators=compare_records(traces['Dense'],traces['Physical'])
    table(OUT/'diagnosis/operator_comparison.csv',operators)
    first=next(r for r in operators if not r['equal'] and r['operator']!='mask')
    assert first['layer']==1 and first['operator']=='q_proj'
    actual=[]
    for name in ['L0_output','L1_input','L1_norm_input','L1_norm_output','L1_q_proj_input','L1_q_proj_output']:
        a=traces['Dense'].actual[name]
        b=traces['Physical'].actual[name]
        if a.shape!=b.shape:
            assert a.shape[1]==b.shape[1]+1
            a=a[:,:-1]
        actual.append(dict(operator=name,dense_shape=list(traces['Dense'].actual[name].shape),
            physical_shape=list(b.shape),**metrics(a,b)))
    assert all(r['equal'] for r in actual[:-1]) and not actual[-1]['equal']
    table(OUT/'diagnosis/actual_upstream.csv',actual)
    table(OUT/'diagnosis/logits_comparison.csv',compare(*records['Dense'],*records['Physical']))
    PRIVATE.mkdir(parents=True,exist_ok=True)
    path=PRIVATE/'actual_early_projection_inputs.pt'
    assert not path.exists()
    torch.save({name:t.actual for name,t in traces.items()},path)
    write(OUT/'capture.json',dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    write(OUT/'diagnosis_completion.json',dict(status='REPRODUCED',first=first,upstream=actual,
        pid=os.getpid(),gpu_uuid=UUID,input_binding=layout,full_original_prefill_N=layout['input_length']))


def isolate():
    assert not (OUT/'isolation_completion.json').exists()
    c=json.loads((OUT/'capture.json').read_text())
    assert sha(c['path'])==c['sha256']
    actual=torch.load(c['path'],map_location='cpu',weights_only=True)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    workspace=torch.empty(8*2**20,dtype=torch.uint8,device='cuda')
    rows=[]
    with torch.inference_mode():
        for i in range(1,7):
            for name in ['q_proj','k_proj','v_proj','o_proj']:
                label=f'L{i}_{name}'
                a,b=[actual[k][label+'_input'] for k in ['Dense','Physical']]
                assert a.shape[1]==b.shape[1]+1
                if not torch.equal(a[:,:-1],b):
                    continue
                key=f'model.language_model.layers.{i}.self_attn.{name}.weight'
                with safe_open(MODEL/index[key],framework='pt',device='cpu') as file:
                    w=file.get_tensor(key).cuda()
                x=a.cuda().reshape(-1,a.shape[-1]).contiguous()
                compact=x[:-1].contiguous()
                dense=torch.nn.functional.linear(x,w)
                native=torch.nn.functional.linear(compact,w)
                dp=LtPlan(x.shape[0],w.shape[0],w.shape[1],workspace)
                cp=LtPlan(compact.shape[0],w.shape[0],w.shape[1],workspace)
                try:
                    assert cp.select(dp,0)==0
                    aligned=cp.run(compact,w)
                    entry=dict(layer=i,operator=name,input_prefix_equal=True,dense_M=x.shape[0],physical_M=compact.shape[0],
                        K=x.shape[1],N=w.shape[0],dtype=str(x.dtype),dense_descriptor_config=dp.config(),compact_config=cp.config(),
                        native_replays_dense=torch.equal(dense.cpu(),actual['Dense'][label+'_output'].reshape(dense.shape)),
                        native_replays_physical=torch.equal(native.cpu(),actual['Physical'][label+'_output'].reshape(native.shape)),
                        before=metrics(dense[:-1],native),after=metrics(dense[:-1],aligned))
                    assert entry['native_replays_dense'] and entry['native_replays_physical'] and entry['after']['equal']
                    rows.append(entry)
                    print('ISOLATE',json.dumps(entry),flush=True)
                finally:
                    cp.close()
                    dp.close()
                del w,x,compact,dense,native,aligned
    assert any(r['layer']==1 and r['operator']=='q_proj' and not r['before']['equal'] for r in rows)
    write(OUT/'isolation_completion.json',dict(status='PASS',pid=os.getpid(),rows=rows,
        no_model_loaded=True,conclusion='Identical real input rows; original-M cuBLASLt descriptor aligns compact BF16 projection'))


def gpu(mode):
    p=json.loads((OUT/'diagnostic_protocol.json').read_text())
    assert p['source_files']==sources() and p['gate']==GATE
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    state=resource()
    assert UUID not in state['processes']
    assert int(next(l for l in state['gpus'].splitlines() if UUID in l).split(',')[-2].split()[0])>23500
    write(OUT/(mode+'_gpu_before.json'),state)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    try:
        {'diagnose':diagnose,'isolate':isolate}[mode]()
    except BaseException as exc:
        write(OUT/(mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','diagnose','isolate'])
    mode=parser.parse_args().mode
    freeze() if mode=='freeze' else gpu(mode)
