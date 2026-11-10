"""Only question 20705853 plus the preceding five regression inputs."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_svh_numerical_repair.engine import generate
from ..qwen3vl_svh_numerical_repair.run import GATE, MODEL, Trace, audit_semantics, compare, metrics, sha, source_files
from ..qwen3vl_svh_preflight48.freeze import prepare, table, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_l28_20705853_repair'
OLD = ROOT / 'vlm_research/results/qwen3vl_svh_expanded_preflight48'
PRIVATE = Path('/data/lk/ATE-resources/experiments/qwen3vl_svh_l28_20705853_repair')
UUID = 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'
ORDER = ['layer_input','input_norm','q_input','q_proj','k_input','k_proj','v_input','v_proj',
         'q_norm','k_norm','rope_q','rope_k','mask','QK_text','softmax_text','probabilities',
         'AV','o_input','o_proj','attention_output','attention_residual','post_norm',
         'gate_input','gate_proj','activation','up_input','up_proj','down_input','down_proj','mlp_output','layer_output']


def samples():
    cases = json.loads((OLD/'numerical_cases.json').read_text())
    by_key = {r['sample']['key']:r['sample'] for r in cases}
    previous = json.loads((OLD/'numerical_progress.json').read_text())
    keys = [r['key'] for r in previous]
    assert len(keys) == 6 and keys[-1] == 'GQA:20705853'
    return [by_key[k] for k in keys]


def source_snapshot():
    return {**source_files(), **{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}}


def compare_records(a, b):
    rows = []
    for key in sorted(a.records.keys() & b.records.keys(), key=lambda k:(k[0],k[1],ORDER.index(k[2]))):
        x,y = a.records[key],b.records[key]
        positions = sorted(set(x['positions']) & set(y['positions']))
        xx = x['value'].index_select(x['axis'],torch.tensor([x['positions'].index(p) for p in positions]))
        yy = y['value'].index_select(y['axis'],torch.tensor([y['positions'].index(p) for p in positions]))
        if x['keys'] is not None:
            keys = sorted(set(x['keys']) & set(y['keys']))
            xx=xx[..., [x['keys'].index(p) for p in keys]]
            yy=yy[..., [y['keys'].index(p) for p in keys]]
        if key[2] == 'mask':
            # finfo.min and -inf both forbid a key; compare the permission graph.
            xx = xx > -1e20
            yy = yy > -1e20
        rows.append(dict(step=key[0],layer=key[1],operator=key[2],
                         dense_shape=list(x['value'].shape),physical_shape=list(y['value'].shape),
                         compared_shape=list(xx.shape),**metrics(xx,yy)))
    return rows


class OperatorTrace(Trace):
    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(super().context())
            for layer in self.d.layers:
                for branch,names in [(layer.self_attn,['q','k','v','o']), (layer.mlp,['gate','up','down'])]:
                    for name in names:
                        mod=getattr(branch,name+'_proj')
                        h=mod.register_forward_pre_hook(lambda m,args,name=name:self.put(name+'_input',args[0]))
                        stack.callback(h.remove)
                h=layer.mlp.act_fn.register_forward_hook(lambda m,a,o:self.put('activation',o))
                stack.callback(h.remove)
            eager=qm.eager_attention_forward
            def observed(module,q,k,v,attention_mask=None,**kw):
                if not hasattr(module,'layer_idx'):
                    return eager(module,q,k,v,attention_mask,**kw)
                keys=self.logical(k.shape[-2])
                text_keys=[j for j,p in enumerate(keys) if p not in self.images]
                allowed=None if attention_mask is None else attention_mask[...,text_keys]>-1e20
                real_matmul=torch.matmul
                softmax=torch.nn.functional.softmax
                call=0
                def matmul(x,y,*args,**kwargs):
                    nonlocal call
                    out=real_matmul(x,y,*args,**kwargs)
                    call+=1
                    if call==1:
                        value=out[...,text_keys]
                        if allowed is not None:value=torch.where(allowed,value,0.)
                        self.put('QK_text',value,2,[keys[j] for j in text_keys])
                    return out
                def soft(x,*args,**kwargs):
                    # Capture the actual masked/scaled logits, not a recomputation.
                    value=x[...,text_keys]
                    if allowed is not None:value=torch.where(allowed,value,0.)
                    self.put('softmax_text',value,2,[keys[j] for j in text_keys])
                    return softmax(x,*args,**kwargs)
                with patch.object(torch,'matmul',matmul),patch.object(torch.nn.functional,'softmax',soft):
                    return eager(module,q,k,v,attention_mask,**kw)
            stack.enter_context(patch.object(qm,'eager_attention_forward',observed))
            yield self


def cpu_prepare():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Preserve earlier freeze'
    selected=samples()
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    bindings=json.loads((OLD/'input_bindings.json').read_text())
    for row in selected:
        inputs,layout=prepare(processor,row)
        assert layout==bindings[row['key']]
    write(OUT/'protocol.json',dict(base_commit='dfd169ad206fabbd38de24a631108e716c661383',
          revision=MODEL.name,gate=GATE,algorithm_unchanged=True,dtype='BF16',eager=True,
          question_ids=[s['question_id'] for s in selected],diagnosis_question='20705853',
          inputs_and_bindings_sha256={p.name:sha(p) for p in [OLD/'numerical_cases.json',OLD/'input_bindings.json']},
          prohibited=['59-boundary run','48-task pretest','full evaluation','automatic push'],
          source_files=source_snapshot(),gpu_uuid=UUID,
          decode_diagnostic='Natural generation remains EOS-stopped. Additional fixed-prefix cache calls, if needed for two decode steps, are diagnostic only.'))
    write(OUT/'cpu_preparation.json',dict(status='PASS',samples_bound=6,cuda_initialized=False))


def preflight():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
    state=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    assert UUID not in raw, 'GPU acquired by another task'
    line=next(l for l in state.splitlines() if UUID in l)
    assert '3090' in line and int(line.split(',')[-2].strip().split()[0])>23500
    return dict(time=time.time(),gpus=state,processes=raw,selected_uuid=UUID,pid=os.getpid())


def diagnose(model,processor):
    folder=OUT/'diagnosis_original'
    assert not folder.exists()
    row=samples()[-1]
    inputs,layout=prepare(processor,row)
    traces,results={},{}
    hooks=hook_counts(model)
    for name,condition in [('Native','N'),('Noop','N'),('Dense','F'),('Physical','C')]:
        trace=OperatorTrace(model,layout) if name in ('Dense','Physical') else None
        with ExitStack() as stack:
            if trace:stack.enter_context(trace.context())
            result,vectors=generate(model,processor,inputs,layout,condition,repaired=name=='Physical',
                                   audit=True,capture=True,noop=name=='Noop',max_tokens=32)
        assert hook_counts(model)==hooks
        write(folder/(name+'.json'),result)
        np.savez_compressed(folder/(name+'_logits.npz'),logits=torch.stack(vectors).numpy(),token_ids=result['token_ids'])
        results[name]=result,vectors
        if trace:traces[name]=trace
        print('DIAGNOSE',name,result['text'],result['token_ids'],flush=True)
    assert all(torch.equal(a,b) for a,b in zip(results['Native'][1],results['Noop'][1]))
    comparisons=compare(*results['Dense'],*results['Physical'])
    table(folder/'logits_comparison.csv',comparisons)
    rows=compare_records(traces['Dense'],traces['Physical'])
    table(folder/'operator_comparison.csv',rows)
    write(folder/'positions_cache_semantics.json',audit_semantics(traces['Dense'],traces['Physical']))
    mismatches=[r for r in rows if not r['equal'] and r['operator']!='mask']
    first=mismatches[0] if mismatches else None
    write(folder/'first_divergence.json',dict(first=first,first_per_layer=[next(r for r in mismatches if r['step']==0 and r['layer']==i)
          for i in range(36) if any(r['step']==0 and r['layer']==i for r in mismatches)]))
    PRIVATE.mkdir(parents=True,exist_ok=True)
    raw_path=PRIVATE/'original_operator_trace.pt'
    torch.save({name:dict(records=t.records,positions=t.positions,caches=t.caches) for name,t in traces.items()},raw_path)
    write(folder/'raw_trace.json',dict(path=str(raw_path),sha256=sha(raw_path),bytes=raw_path.stat().st_size))
    print('FIRST',json.dumps(first),flush=True)
    print('L28',json.dumps([r for r in mismatches if r['step']==0 and r['layer']==28]),flush=True)


def gpu_run(mode):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['source_files']==source_snapshot(), 'Changed diagnostic snapshot'
    write(OUT/(mode+'_gpu_before.json'),preflight())
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=None
    try:
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
               attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        if mode=='diagnose':diagnose(model,processor)
        else:raise ValueError(mode)
    except BaseException as exc:
        write(OUT/(mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','diagnose'])
    args=parser.parse_args()
    cpu_prepare() if args.mode=='prepare' else gpu_run(args.mode)
