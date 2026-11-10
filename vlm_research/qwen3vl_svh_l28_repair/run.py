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
from .engine import AVAlignedExecution, generate as fixed_generate, restore_query_rows
from ..qwen3vl_c3v2_physical import engine as physical_engine

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


def cpu_prepare_fixed():
    assert not torch.cuda.is_initialized()
    old=json.loads((OUT/'protocol.json').read_text())
    assert old['gate']==GATE
    assert not (OUT/'fixed_protocol.json').exists()
    for name,h in old['source_files'].items():
        if 'qwen3vl_svh_l28_repair/' not in name:
            path=ROOT/name if not name.startswith('/') else Path(name)
            assert sha(path)==h, ('Frozen historical implementation changed',name)
    write(OUT/'fixed_protocol.json',dict(base_commit='dfd169ad206fabbd38de24a631108e716c661383',
        diagnosis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gate=GATE,source_files=source_snapshot(),gpu_uuid=UUID,questions=[r['question_id'] for r in samples()],
        verify_order=['20705853']+[r['question_id'] for r in samples()[:-1]],
        repair='Only post-exit Prefill AV gets temporary zero query rows at original positions; no new hidden/KV rows or projections',
        native_and_dense_reference_unchanged=True,dtype='bfloat16',
        two_decode_steps='Diagnostic fixed-prefix calls consume Dense natural [wood,EOS]; step2 is explicitly after natural EOS, not a generation result',
        no_evaluations=True,independent_new_process_required=True))


def isolate():
    folder=OUT/'isolated_AV'
    assert not folder.exists()
    trace=torch.load(PRIVATE/'original_operator_trace.pt',map_location='cpu',weights_only=False)
    rows=[]
    for layer in [26,27,28]:
        dense,physical=trace['Dense'],trace['Physical']
        p=dense['records'][(0,layer,'probabilities')]['value']
        pp=physical['records'][(0,layer,'probabilities')]['value']
        v=dense['caches'][(0,layer)]['v']
        vp=physical['caches'][(0,layer)]['v']
        assert torch.equal(p,pp) and torch.equal(v,vp)
        text=dense['positions'][layer]['cp']
        n=p.shape[-1]
        image=[i for i in range(n) if i not in text]
        assert float(p[...,image].abs().max())==0.
        pc=p.cuda()
        vc=v.cuda()
        full_v=vc.new_zeros((*vc.shape[:-2],n,vc.shape[-1]))
        ix=torch.tensor(text,device=vc.device)
        full_v.index_copy_(-2,ix,vc)
        full_v=qm.repeat_kv(full_v,4)
        expanded,ix=restore_query_rows(pc,text,n)
        compact=torch.matmul(pc,full_v).transpose(1,2).contiguous().cpu()
        restored=torch.matmul(expanded,full_v).index_select(-2,ix).transpose(1,2).contiguous().cpu()
        expected_d=dense['records'][(0,layer,'AV')]['value']
        expected_p=physical['records'][(0,layer,'AV')]['value']
        r=dict(layer=layer,probabilities_equal=True,text_V_equal=True,
               compact_P_shape=list(pc.shape),dense_P_shape=list(expanded.shape),V_shape=list(full_v.shape),
               compact_P_stride=list(pc.stride()),dense_P_stride=list(expanded.stride()),V_stride=list(full_v.stride()),
               compact_reproduces_saved_physical=torch.equal(compact,expected_p),
               restored_reproduces_saved_dense=torch.equal(restored,expected_d),
               before=metrics(expected_d,compact),after=metrics(expected_d,restored),
               dtype=str(pc.dtype),allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        rows.append(r)
        write(folder/('L'+str(layer)+'.json'),r)
        np.savez_compressed(folder/('L'+str(layer)+'_witness.npz'),probabilities=p.float().numpy(),
             text_V=v.float().numpy(),dense_AV=expected_d.float().numpy(),physical_AV=expected_p.float().numpy(),
             compact_replay=compact.float().numpy(),restored_replay=restored.float().numpy(),positions=np.asarray(text))
        print('ISOLATE',json.dumps(r),flush=True)
        assert r['compact_reproduces_saved_physical'] and r['restored_reproduces_saved_dense'], 'AV geometry hypothesis not independently verified'
    write(folder/'completion.json',dict(status='PASS',layers=[26,27,28],model_loaded=False,
          hypothesis='Original query-row geometry changes BF16 AV reduction; zero-row restoration reproduces Dense real outputs exactly',rows=rows))


def fixed_prefix_diagnostic(model,processor,inputs,layout,prefix,fixed):
    """Two cache-consuming Decode calls, explicitly distinct from EOS-stopped generation."""
    device=next(model.parameters()).device
    x={k:v.to(device) for k,v in inputs.items()}
    execution=(AVAlignedExecution if fixed else physical_engine.Execution)(model,layout,'C' if fixed else 'F',max_tokens=3,audit=True)
    cache=None
    scores=[]
    ids=[]
    with torch.inference_mode(),execution.context():
        for t in range(3):
            kw=dict(x) if t==0 else dict(input_ids=torch.tensor([[prefix[t-1]]],device=device),
                 attention_mask=torch.ones((1,layout['input_length']+t),device=device,dtype=torch.long))
            kw.update(use_cache=True,past_key_values=cache,logits_to_keep=1,
                 cache_position=torch.arange(layout['input_length'],device=device) if t==0 else torch.tensor([layout['input_length']+t-1],device=device))
            out=model(**kw)
            assert bool(torch.isfinite(out.logits).all())
            scores.append(out.logits[0,-1].detach().float().cpu())
            ids.append(int(out.logits[0,-1].argmax()))
            cache=out.past_key_values
    result=dict(token_ids=ids,prefix=prefix,diagnostic_only=True,not_free_generation=True,
                steps=[dict(t=0,prefill=True),dict(t=1,consumed=prefix[0],after_natural_eos=False),
                       dict(t=2,consumed=prefix[1],after_natural_eos=True)],audit=execution.audit())
    return result,scores


def verify(model,processor):
    assert json.loads((OUT/'isolated_AV/completion.json').read_text())['status']=='PASS'
    folder=OUT/'independent_verification'
    assert not folder.exists()
    old_pid=json.loads((OUT/'diagnose_gpu_before.json').read_text())['pid']
    assert old_pid!=os.getpid()
    hooks=hook_counts(model)
    saved_eager,saved_matmul=qm.eager_attention_forward,torch.matmul
    weights=[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
    selected=samples()
    rows=[]
    for row in [selected[-1]]+selected[:-1]:
        inputs,layout=prepare(processor,row)
        results,traces={},{}
        target=row['question_id']=='20705853'
        item=folder/row['question_id']
        for name,kind in [('Native','N'),('Noop','N'),('Dense','F'),('Fixed','C')]:
            trace=OperatorTrace(model,layout) if target and name in ('Dense','Fixed') else None
            with ExitStack() as stack:
                if trace:stack.enter_context(trace.context())
                if name=='Fixed':
                    result,vectors=fixed_generate(model,processor,inputs,layout,'C',audit=True,capture=True,max_tokens=32)
                else:
                    result,vectors=generate(model,processor,inputs,layout,kind,audit=True,capture=True,
                                           noop=name=='Noop',max_tokens=32)
            assert hooks==hook_counts(model) and qm.eager_attention_forward is saved_eager and torch.matmul is saved_matmul
            assert {k:tensor_digest(v) for k,v in inputs.items()}==layout['input_tensor_hashes']
            write(item/(name+'.json'),result)
            np.savez_compressed(item/(name+'_logits.npz'),logits=torch.stack(vectors).numpy(),token_ids=result['token_ids'])
            results[name]=result,vectors
            if trace:traces[name]=trace
        strict=(results['Native'][0]['token_ids']==results['Noop'][0]['token_ids']
                and len(results['Native'][1])==len(results['Noop'][1])
                and all(torch.equal(x,y) for x,y in zip(results['Native'][1],results['Noop'][1])))
        assert strict
        comparisons=compare(*results['Dense'],*results['Fixed'])
        table(item/'logits_comparison.csv',comparisons)
        assert comparisons and all(r['pass'] for r in comparisons), 'Frozen numerical gate failed after repair'
        result=results['Fixed'][0]
        assert len(result['audit']['AV_query_row_alignment'])==14
        assert all(r['image_hidden_rows']==r['persistent_dummy_KV']==0 for r in result['audit']['AV_query_row_alignment'])
        assert all(r['forbidden_probability']==0 for r in result['audit']['probability'])
        assert result['audit']['deepstack']==[0,1,2]
        entry=dict(question_id=row['question_id'],native_noop_strict=strict,pass_gate=True,
            worst_mean_abs=max(r['mean_abs'] for r in comparisons),worst_max_abs=max(r['max_abs'] for r in comparisons),
            complete_tokens_equal=True,eos=result['eos'],generated_tokens=result['generated_tokens'],
            image_tokens=layout['image_tokens'],input_length=layout['input_length'],peak_allocated=result['peak_allocated'],
            extra_AV_alignment_flops=result['audit']['extra_AV_query_alignment_flops'])
        if target:
            operator_rows=compare_records(traces['Dense'],traces['Fixed'])
            table(item/'operator_comparison.csv',operator_rows)
            write(item/'semantics.json',audit_semantics(traces['Dense'],traces['Fixed']))
            entry['first_remaining_operator_difference']=next((r for r in operator_rows if not r['equal'] and r['operator']!='mask'),None)
            assert entry['first_remaining_operator_difference'] is None, 'Target operator mismatch remains'
            with torch.inference_mode():
                hf=model.generate(**{k:v.cuda() for k,v in inputs.items()},do_sample=False,num_beams=1,
                    max_new_tokens=32,use_cache=True,logits_to_keep=1,return_dict_in_generate=True,output_logits=True,
                    eos_token_id=[151645,151643],pad_token_id=151643)
            assert hf.sequences[0,layout['input_length']:].tolist()==results['Native'][0]['token_ids']
            assert len(hf.logits)==len(results['Native'][1])
            assert all(torch.equal(a,b[0].float().cpu()) for a,b in zip(results['Native'][1],hf.logits))
            del hf
            prefix=results['Dense'][0]['token_ids'][:2]
            assert len(prefix)==2 and prefix[-1] in [151645,151643]
            diagnostic={}
            dtraces={}
            for label,fixed in [('Dense',False),('Fixed',True)]:
                trace=OperatorTrace(model,layout)
                with trace.context():
                    diagnostic[label]=fixed_prefix_diagnostic(model,processor,inputs,layout,prefix,fixed)
                dtraces[label]=trace
                r,v=diagnostic[label]
                assert all(torch.equal(a,b) for a,b in zip(v[:2],results[label][1]))
                write(item/(label+'_two_decode_diagnostic.json'),r)
                np.savez_compressed(item/(label+'_two_decode_logits.npz'),logits=torch.stack(v).numpy(),token_ids=r['token_ids'])
            dcompare=compare(*diagnostic['Dense'],*diagnostic['Fixed'])
            table(item/'two_decode_comparison.csv',dcompare)
            assert len(dcompare)==3 and all(r['pass'] for r in dcompare)
            table(item/'two_decode_operator_comparison.csv',compare_records(dtraces['Dense'],dtraces['Fixed']))
            write(item/'two_decode_semantics.json',audit_semantics(dtraces['Dense'],dtraces['Fixed']))
            entry['two_decode_diagnostic_pass']=True
            entry['second_decode_after_natural_eos']=True
            entry['independent_hf_native_strict']=True
            del diagnostic,dtraces
        rows.append(entry)
        write(item/'validation.json',entry)
        write(folder/'progress.json',rows)
        print('VERIFY',json.dumps(entry),flush=True)
        assert weights==[(n,p.data_ptr(),p._version) for n,p in model.named_parameters()]
        del results,traces,inputs,vectors,result
        gc.collect()
    assert hooks==hook_counts(model) and qm.eager_attention_forward is saved_eager and torch.matmul is saved_matmul
    write(folder/'completion.json',dict(status='PASS',inputs_completed=6,failure_case_pass=True,
          prior_five_regressions_pass=True,new_process=True,pid=os.getpid(),diagnosis_pid=old_pid,
          two_decode_diagnostic_pass=True,second_decode_after_natural_eos=True,
          no_59_boundary_run=True,no_48_pretest=True,no_full_evaluation=True,gate=GATE,rows=rows))


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
    protocol=json.loads((OUT/('protocol.json' if mode=='diagnose' else 'fixed_protocol.json')).read_text())
    assert protocol['source_files']==source_snapshot(), 'Changed diagnostic snapshot'
    write(OUT/(mode+'_gpu_before.json'),preflight())
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    if mode=='isolate':
        isolate()
        torch.cuda.empty_cache()
        return
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=None
    try:
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
               attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        if mode=='diagnose':diagnose(model,processor)
        elif mode=='verify':verify(model,processor)
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
    parser.add_argument('mode',choices=['prepare','prepare-fixed','diagnose','isolate','verify'])
    args=parser.parse_args()
    if args.mode=='prepare':cpu_prepare()
    elif args.mode=='prepare-fixed':cpu_prepare_fixed()
    else:gpu_run(args.mode)
