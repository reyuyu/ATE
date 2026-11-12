"""GPU1 only. Smoke gates checkpointed native collection and fixed generation."""
import argparse
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import time
import traceback

import torch

from .common import OUT, ROOT, MODEL, RAW, CONDITIONS, read, write, sha256, verify, digest, prediction
from .engine import WindowHook, first_metrics, forward, generate
from vlm_research.internvl3_smoke.engine import (MassObserver, LayerTrace, hooks_count,
    official_preprocessor, tensor_hash, replay)
from vlm_research.internvl3_smoke.run import ChatCapture, architecture, expanded_prompt, command, ResourceMonitor


class Monitor(ResourceMonitor):
    def __exit__(self,*args):
        self.sample()
        self.stop.set()
        self.thread.join()
        write(OUT/f'gpu1_resources_{os.getpid()}.json',dict(rows=self.rows,errors=self.errors,pid=os.getpid(),
            peak_process_memory_mib=max((r['memory_mib'] for r in self.rows),default=0),
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))


def native(model,tokenizer,sample,loader):
    assert sha256(sample['image_path'])==sample['image_sha256']
    pixels=loader(sample['image_path'],max_num=1).to('cuda:0',dtype=torch.bfloat16)
    assert pixels.shape[0]==1
    before=tensor_hash(pixels)
    with ChatCapture(model) as capture:
        answer=model.chat(tokenizer,pixels,sample['request'],dict(do_sample=False,num_beams=1,max_new_tokens=32))
    assert tensor_hash(pixels)==before
    assert capture.input_ids[0].tolist()==sample['input_ids']
    assert expanded_prompt(model,sample,1)==sample['prompt']
    assert len(capture.generated_ids)==len(capture.calls)
    assert all(c['cache_length']==len(sample['input_ids'])+i and c['query_length']==(len(sample['input_ids']) if i==0 else 1)
               for i,c in enumerate(capture.calls))
    eos=capture.generated_ids[-1]==tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    value=dict(answer=answer,normalized=prediction(answer),token_ids=capture.generated_ids,
        first_token=capture.generated_ids[0],eos=eos,truncated=len(capture.generated_ids)==32 and not eos,
        empty=not bool(answer),length=len(capture.generated_ids),
        repeated=len(capture.generated_ids)>=6 and any(capture.generated_ids[i:i+3]==capture.generated_ids[i+3:i+6]
            for i in range(len(capture.generated_ids)-5)),cache_trace=capture.calls,
        vision_trace=capture.vision)
    assert int(capture.first_logits.argmax())==capture.generated_ids[0]
    return value,capture.first_logits,capture.saved


@torch.inference_mode()
def smoke(model,tokenizer,samples,loader,protocol):
    checks=[]
    layers=model.language_model.model.layers
    baseline=hooks_count(model)
    for sample in samples:
        print('SMOKE',sample['task'],sample['question_id'],'length',sample['layout']['input_length'],flush=True)
        value,full,saved=native(model,tokenizer,sample,loader)
        repeat,reference,_,_=replay(model,saved)
        assert torch.equal(repeat,full)
        noop,noop_logits=generate(model,tokenizer,saved,CONDITIONS['W2'],noop=True,audit=True)
        plain,plain_logits=generate(model,tokenizer,saved,audit=True)
        assert torch.equal(noop_logits,full) and torch.equal(plain_logits,full)
        assert noop['token_ids']==plain['token_ids']==value['token_ids']
        with MassObserver({i:l.self_attn for i,l in enumerate(layers)},sample['layout']) as observer:
            mass_logits=forward(model,saved)
        assert torch.equal(full,mass_logits) and len(observer.rows)==28
        single=[]
        for index in (2,27):
            off,records,_,_=replay(model,saved,layer=index)
            for a,b in zip(reference,records):
                assert a['attention_prefix_sha256']==b['attention_prefix_sha256']
                assert a['output_prefix_sha256']==b['output_prefix_sha256']
            single.append(dict(layer=index,max_logit_difference=float((full.float()-off.float()).abs().max())))
        with WindowHook(layers,CONDITIONS['W2'],len(sample['input_ids'])):
            with LayerTrace(layers,len(sample['input_ids'])-1) as trace:
                window_prefill=forward(model,saved)
                trace.validate()
        window,first=generate(model,tokenizer,saved,CONDITIONS['W2'],audit=True)
        assert torch.equal(window_prefill,first)
        assert window['prefill_prefix_hashes']==plain['prefill_prefix_hashes']
        for index in range(28):
            assert torch.count_nonzero(trace.records[index]['attention'])==0 if index in CONDITIONS['W2'] else torch.count_nonzero(trace.records[index]['attention'])>0
        assert hooks_count(model)==baseline
        import transformers.models.qwen2.modeling_qwen2 as modeling
        original=modeling.eager_attention_forward
        try:
            with WindowHook(layers,[2],len(sample['input_ids'])):
                with MassObserver({0:layers[0].self_attn},sample['layout']):
                    raise RuntimeError('intentional cleanup check')
        except RuntimeError as error:
            assert str(error)=='intentional cleanup check'
        assert hooks_count(model)==baseline and modeling.eager_attention_forward is original
        checks.append(dict(task=sample['task'],question_id=sample['question_id'],input_length=len(sample['input_ids']),
            noop_max_abs=0.,observer_max_abs=0.,all_28_layers=True,max_mass_row_error=max(observer.errors.values()),
            single_checks=single,window_first_logits_exact=True,non_target_prefill_exact=True,
            all_mlp_residual_valid=True,cache_lengths_correct=True,every_decode_query_hooked=True,
            normal_exception_cleanup=True,native_tokens=value['token_ids'],window_tokens=window['token_ids']))
    write(OUT/'gpu_smoke.json',dict(passed=True,cases=checks,protocol_sha256=sha256(OUT/'protocol.json')))


@torch.inference_mode()
def collect(model,tokenizer,samples,loader,protocol,monitor):
    from safetensors.torch import save_file
    started=time.perf_counter()
    binding=sha256(OUT/'protocol.json')
    layers=model.language_model.model.layers
    baseline=hooks_count(model)
    for number,sample in enumerate(samples,1):
        key=sample['task']+'__'+sample['question_id']
        path=OUT/'checkpoints'/(key+'.json')
        if path.exists():
            previous=read(path)
            assert previous['binding']==binding and previous['input_binding']==digest(sample)
            if previous.get('complete'):
                assert len(previous['sensitivity'])==28 and set(previous['windows'])==set(CONDITIONS)
                continue
        monitor.stage=key
        value,full,saved=native(model,tokenizer,sample,loader)
        record=read(path) if path.exists() else dict(binding=binding,input_binding=digest(sample),task=sample['task'],
            question_id=sample['question_id'],image_sha256=sample['image_sha256'],sensitivity=[],windows={},complete=False)
        if 'Full' in record['windows']:
            assert value==record['windows']['Full'], 'Resumed native generation differs'
            assert tensor_hash(full)==record['first_logits_sha256']
        record['windows']['Full']=value
        record['first_logits_sha256']=tensor_hash(full)
        raw=RAW/(key+'.safetensors')
        RAW.mkdir(parents=True,exist_ok=True)
        if not raw.exists():
            save_file({'native_first_logits':full},str(raw))
        record.update(native_first_logits_path=str(raw),native_first_logits_file_sha256=sha256(raw))
        with MassObserver({i:l.self_attn for i,l in enumerate(layers)},sample['layout']) as observer:
            measured=forward(model,saved)
        assert torch.equal(measured,full) and len(observer.rows)==28
        record['mass']={str(i):dict(head_masses=observer.rows[i],row_sum_max_error=observer.errors[i]) for i in range(28)}
        write(path,record)
        complete_layers={r['layer'] for r in record['sensitivity']}
        for index in range(28):
            if index in complete_layers:
                continue
            with WindowHook(layers,[index],len(sample['input_ids'])) as hook:
                off=forward(model,saved)
            assert hook.calls[index]==[len(sample['input_ids'])]
            record['sensitivity'].append(dict(layer=index,**first_metrics(full,off)))
            write(path,record)
        for name,targets in CONDITIONS.items():
            if name in record['windows']:
                continue
            value,first=generate(model,tokenizer,saved,targets)
            value.update(first_metrics(full,first))
            record['windows'][name]=value
            write(path,record)
        assert hooks_count(model)==baseline
        record['complete']=True
        record['seconds_elapsed']=time.perf_counter()-started
        write(path,record)
        print(f'COMPLETE {number}/120 {key} elapsed={time.perf_counter()-started:.1f}s full={record["windows"]["Full"]["answer"]!r}',flush=True)
        write(OUT/'progress.json',dict(completed=sum(read(p).get('complete',False) for p in (OUT/'checkpoints').glob('*.json')),
            total=120,elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),last_sample=key))
        del saved,full,record


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','run'])
    mode=parser.parse_args().mode
    protocol=verify()
    assert read(OUT/'cpu_validation.json')['passed']
    if mode=='run':
        assert read(OUT/'gpu_smoke.json')['passed']
        assert read(OUT/'gpu_smoke.json')['protocol_sha256']==sha256(OUT/'protocol.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.device_count()==1
    assert not command('nvidia-smi','-i','1','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'), 'GPU1 occupied; stop'
    uuid=command('nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader')
    write(OUT/f'gpu1_preflight_{mode}.json',dict(snapshot=command('nvidia-smi'),uuid=uuid,pid=os.getpid(),physical_gpu=1,device='cuda:0'))
    from transformers import AutoModel,AutoTokenizer
    model=None
    start=time.perf_counter()
    status=dict(passed=False,mode=mode,errors=[])
    try:
        with Monitor() as monitor:
            model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,use_flash_attn=False,device_map={'':'cuda:0'},use_safetensors=True).eval()
            assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
            assert {p.dtype for p in model.parameters()}=={torch.bfloat16}
            assert model.language_model.config._attn_implementation=='eager' and len(model.language_model.model.layers)==28
            assert not any(layer.attn.use_flash_attn for layer in model.vision_model.encoder.layers)
            mapped=command('nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
            assert any(str(os.getpid())==line.split(',')[0].strip() and uuid in line for line in mapped.splitlines())
            tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
            write(OUT/'architecture_bindings.json',architecture(model,tokenizer))
            versions=[(p._version,p.data_ptr()) for p in model.parameters()]
            baseline=hooks_count(model)
            loader,_=official_preprocessor(MODEL)
            samples=read(OUT/'sample_manifest.json')['samples']
            if mode=='smoke':
                wanted=[tuple(p) for p in protocol['smoke_samples']]
                lookup={(s['task'],s['question_id']):s for s in samples}
                smoke(model,tokenizer,[lookup[k] for k in wanted],loader,protocol)
            else:
                collect(model,tokenizer,samples,loader,protocol,monitor)
            assert versions==[(p._version,p.data_ptr()) for p in model.parameters()]
            assert hooks_count(model)==baseline
            status.update(passed=True,no_nan_inf_oom=True,weights_unchanged=True,no_hook_residue=True,
                physical_gpu=1,uuid=uuid,device='cuda:0',code_commit=command('git','rev-parse','HEAD'))
    except Exception:
        status['errors'].append(traceback.format_exc())
        raise
    finally:
        status['seconds']=time.perf_counter()-start
        write(OUT/f'execution_{mode}.json',status)
        model=None
        gc.collect()
        torch.cuda.empty_cache()
        print('END',status['passed'],mode,'seconds',status['seconds'],flush=True)


if __name__=='__main__':
    main()
