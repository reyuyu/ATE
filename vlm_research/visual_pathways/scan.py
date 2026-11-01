"""Resumable per-image single-layer Image OFF; reuse exact historical Qwen240."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import importlib.metadata
import os
import time
import traceback

import torch

from .common import (ROOT,OUT,QH,IMODEL,TASK1_END,read,write,sha,digest,assign,
                     verify,preflight,command)
from .engine import InternPrefill,metrics
from ..internvl3_smoke.engine import tensor_hash,hooks_count,official_preprocessor


def load_intern():
    from transformers import AutoModel,AutoTokenizer
    model=AutoModel.from_pretrained(IMODEL,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,
        device_map={'':'cuda:0'},use_safetensors=True).eval()
    assert len(model.language_model.model.layers)==28
    assert model.language_model.config._attn_implementation=='eager'
    assert not any(l.attn.use_flash_attn for l in model.vision_model.encoder.layers)
    assert {p.dtype for p in model.parameters()}=={torch.bfloat16}
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    return model,AutoTokenizer.from_pretrained(IMODEL,use_fast=False,local_files_only=True),AutoTokenizer.from_pretrained(IMODEL,use_fast=True,local_files_only=True)


@torch.inference_mode()
def intern_smoke(model,tokenizer,fast,sample,loader):
    from ..internvl3_smoke.run import ChatCapture
    runner=InternPrefill(model,tokenizer,fast,sample,loader)
    full,_,reference=runner.forward(trace=True)
    noop,_,_=runner.forward(0,noop=True)
    assert torch.equal(full,noop), 'No-op changed native logits'
    with ChatCapture(model) as capture:
        model.chat(tokenizer,runner.pixels,runner.request,dict(do_sample=False,num_beams=1,max_new_tokens=1))
    assert capture.input_ids[0].tolist()==runner.layout['input_ids']
    assert torch.equal(capture.saved['inputs_embeds'],runner.inputs['inputs_embeds'].cpu())
    assert torch.equal(capture.first_logits,full), 'Official chat and independent native Full differ'
    checks=[]
    for layer in (0,14,27):
        _,audit,trace=runner.forward(layer,trace=True)
        assert trace.rows==reference.rows, 'Non-target Prefill Query hidden states changed'
        assert audit['masked_probability_max']==0 and audit['changed_entries']==256
        checks.append(dict(layer=layer,**audit,non_target_query_states_exact=True))
    again,_,_=runner.forward()
    assert torch.equal(full,again), 'Cross-condition contamination'
    runner.verify()
    return dict(passed=True,question_id=sample['question_id'],native_logits_max_abs=0,
        noop_logits_max_abs=0,official_vision_embedding_exact=True,no_cache=True,
        independent_replay_exact=True,input_binding=digest(runner.layout),checks=checks)


def environment():
    return dict(executable=os.sys.executable,versions={p:importlib.metadata.version(p) for p in ('torch','transformers','tokenizers','Pillow')})


@torch.inference_mode()
def worker(args):
    protocol=verify()
    samples=read(OUT/'sample_manifest.json')['samples']
    if args.cohort=='pilot':
        samples=[s for s in samples if s['pilot']]
    samples=assign(samples,args.shards)[args.shard]
    samples.sort(key=lambda s:(not s['pilot'],s['task'],s['question_id']))
    folder=OUT/args.model/f'{args.cohort}_gpu{args.gpu}_shard{args.shard}of{args.shards}'
    preflight(args.gpu,folder)
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    write(folder/'environment.json',environment())
    if args.model=='internvl':
        model,tokenizer,fast=load_intern()
        loader,_=official_preprocessor(IMODEL)
        smoke=intern_smoke(model,tokenizer,fast,samples[0],loader)
        layers=28
    else:
        from ..qwen3vl_keep_l26.run import load
        from ..qwen3vl_image_source.run import smoke as qsmoke,environment as qenviron
        # Loader's independent preflight is still before any model allocations.
        model,processor=load(args.gpu,folder/'loader')
        assert qenviron()==read(QH/'environment.json')
        historical=read(QH/'sample_manifest.json')['samples']
        oldlookup={(s['task'],s['question_id']):s for s in historical}
        smoke=qsmoke(model,processor,oldlookup[(samples[0]['task'],samples[0]['question_id'])])
        layers=36
    smoke.update(seconds_from_start=time.perf_counter()-started,physical_gpu=args.gpu,
        protocol_sha256=sha(OUT/'protocol.json'),code_commit=command('git','rev-parse','HEAD'))
    write(folder/'validation.json',smoke)
    print('SMOKE PASS',args.model,args.gpu,round(smoke['seconds_from_start'],2),flush=True)
    weights=[(p._version,p.data_ptr()) for p in model.parameters()]
    baseline=hooks_count(model)
    checkpoints=OUT/args.model/'checkpoints'
    reused,new=0,0
    formal=time.perf_counter()
    for index,sample in enumerate(samples):
        if datetime.now().astimezone()>=datetime.fromisoformat(TASK1_END):
            print('TASK1 TIME LIMIT; safe stop between samples',flush=True)
            break
        key=f'{sample["task"]}__{sample["question_id"]}'
        path=checkpoints/f'{key}.json'
        if path.exists():
            prior=read(path)
            assert prior['image_sha256']==sample['image_sha256'] and prior['sample_sha256']==digest(sample)
            assert prior['protocol_sha256']==sha(OUT/'protocol.json')
            if prior.get('complete'):
                assert len(prior['rows'])==layers
                reused+=1
                continue
        if args.model=='qwen':
            from ..qwen3vl_early_windows.experiment import prepare
            from ..qwen3vl_image_source.engine import Prefill
            inputs,layout=prepare(processor,sample)
            if sample['pilot']:
                oldsample=oldlookup[(sample['task'],sample['question_id'])]
                assert layout==oldsample['layout'] and digest(layout)==oldsample['input_binding']
                oldpath=QH/f'shard_{oldsample["shard"]}/checkpoints/{key}.json'
                record=read(oldpath)
                assert record['input_binding']==digest(layout) and len(record['rows'])==36
                assert record['protocol_sha256']==sha(QH/'protocol.json')
                for row in record['rows']:
                    assert row['image_sha256']==sample['image_sha256'] and row['question_id']==sample['question_id']
                record.update(complete=True,image_sha256=sample['image_sha256'],sample_sha256=digest(sample),
                    protocol_sha256=sha(OUT/'protocol.json'),input_layout=layout,
                    historical_protocol_sha256=record['protocol_sha256'],reused_from=str(oldpath),historical_record_sha256=sha(oldpath))
                write(path,record)
                reused+=1
                continue
            runner=Prefill(model,inputs.to('cuda:0'),layout)
            full,_,_=runner.forward(native=True)
        else:
            runner=InternPrefill(model,tokenizer,fast,sample,loader)
            layout=runner.layout
            full,_,_=runner.forward()
        record=dict(complete=False,task=sample['task'],question_id=sample['question_id'],
            image_sha256=sample['image_sha256'],sample_sha256=digest(sample),input_layout=layout,
            input_binding=digest(layout),protocol_sha256=sha(OUT/'protocol.json'),
            code_commit=smoke['code_commit'],full_logits_sha256=tensor_hash(full),rows=[])
        if path.exists():
            prior=read(path)
            assert prior['full_logits_sha256']==record['full_logits_sha256'] and prior['input_binding']==record['input_binding']
            record['rows']=prior['rows']
        for layer in range(layers):
            if any(r['layer']==layer for r in record['rows']):
                continue
            changed,audit,_=runner.forward(layer=layer)
            row=dict(model=args.model,task=sample['task'],question_id=sample['question_id'],
                image_id=sample['image_id'],image_sha256=sample['image_sha256'],physical_gpu=args.gpu,
                layer=layer,image_tokens=layout['image_tokens'],**metrics(full,changed),**audit)
            record['rows'].append(row)
            # Every layer is resumable; a timed-out question is never counted as complete.
            write(path,record)
        runner.verify()
        record['complete']=True
        write(path,record)
        del runner,full
        new+=1
        progress=dict(done=index+1,total=len(samples),new=new,reused=reused,seconds=time.perf_counter()-formal,
            physical_gpu=args.gpu,model=args.model,last_sample=key)
        write(folder/'progress.json',progress)
        if new%10==0 or index==0:
            print('PROGRESS',progress,flush=True)
    assert weights==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==baseline
    write(folder/'complete.json',dict(passed=True,requested=len(samples),complete_count=new+reused,
        status='complete' if new+reused==len(samples) else 'partial',new=new,reused=reused,
        formal_seconds=time.perf_counter()-formal,total_seconds=time.perf_counter()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        no_nan_inf_oom=True,weights_unchanged=True,hooks_clean=True))
    del model
    gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','internvl'],required=True)
    p.add_argument('--cohort',choices=['pilot','all'],required=True)
    p.add_argument('--gpu',type=int,choices=[0,2,3],required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,required=True)
    args=p.parse_args()
    try:
        worker(args)
    except BaseException:
        write(OUT/args.model/f'{args.cohort}_gpu{args.gpu}_shard{args.shard}of{args.shards}'/'error.json',dict(traceback=traceback.format_exc()))
        raise
