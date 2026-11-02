import argparse
import fcntl
import gc
import os
import time
import traceback

import torch

from .common import OUT, MODELS, conditions, sources, freeze, verify, read, write, sha, digest, command, compact_digest
from .engine import prepare, generate, BudgetMask, modules
from .history import token_metrics
from ..internvl3_smoke.engine import hooks_count, tensor_hash, official_preprocessor
from ..visual_pathways.scan import load_intern, environment
from ..visual_pathways.common import IMODEL


def preflight(model):
    gpu=MODELS[model]['gpu'];assert gpu!=1 and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    inventory=command('nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits')
    row=next(x for x in inventory.splitlines() if x.startswith(str(gpu)+', ')).split(', ')
    expected={0:'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',2:'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',3:'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'}
    assert row[1]==expected[gpu] and int(row[3])>=20000 and int(row[4])==0
    assert not command('nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader')
    write(OUT/model/'preflight.json',dict(pid=os.getpid(),physical_gpu=gpu,uuid=row[1],inventory=inventory,time=time.time()))


def load(kind):
    if kind=='qwen':
        from ..qwen3vl_keep_l26.run import load as qload
        model,processor=qload(0,OUT/kind/'loader');return model,processor,None,None
    if kind=='internvl':
        model,tokenizer,fast=load_intern();loader,_=official_preprocessor(IMODEL)
        return model,tokenizer,fast,loader
    from vlm_research.vlm_common import load_model
    path='/data/lk/ATE-resources/models/llava-1.5-7b-hf/'+MODELS[kind]['revision']
    model,processor,env=load_model(path)
    assert env['transformers']=='4.49.0' and env['model_revision']==MODELS[kind]['revision']
    return model,processor,None,None


def check_first(kind, first, history, which='Full'):
    expected=history['full_logits_sha256'] if which=='Full' else history.get('only_logits_sha256')
    if expected is not None:
        assert tensor_hash(first)==expected, f'Historical {kind} {which} first-logit hash mismatch; no tolerance relaxation'
    assert int(first.argmax())==history['Full' if which=='Full' else 'Only_Text'].get('changed_first_token',int(first.argmax()))


@torch.inference_mode()
def smoke(kind,model,processor,fast,loader,sample,history):
    inputs,layout,runner=prepare(kind,model,processor,sample,fast,loader)
    assert (compact_digest(layout['input_mapping']) if kind=='llava' else digest(layout))==history['input_binding']
    tokenizer=processor if kind=='internvl' else processor.tokenizer
    configs,_=conditions(kind)
    native,first,_=generate(kind,model,tokenizer,inputs,layout,native=True)
    full,ff,reference=generate(kind,model,tokenizer,inputs,layout,capture=True)
    noop,nf,_=generate(kind,model,tokenizer,inputs,layout,configs['Only Text']['off'],noop=True)
    assert torch.equal(first,ff) and torch.equal(first,nf)
    assert native['token_ids']==full['token_ids']==noop['token_ids']==history['Full']['token_ids']
    assert native['all_step_logits_sha256']==full['all_step_logits_sha256']==noop['all_step_logits_sha256']
    check_first(kind,first,history)
    if 'full_step_logits_sha256' in history:
        assert native['all_step_logits_sha256']==history['full_step_logits_sha256']
    assert len(native['token_ids'])>=2
    if kind=='internvl':
        from ..internvl3_smoke.run import ChatCapture
        with ChatCapture(model) as official:
            text=model.chat(processor,runner.pixels,runner.request,dict(do_sample=False,num_beams=1,max_new_tokens=32))
        assert torch.equal(official.first_logits,first) and text.strip()==full['text']
    checks=[]
    for name,c in configs.items():
        if name=='Full':continue
        value,f,trace=generate(kind,model,tokenizer,inputs,layout,c['off'],capture=True,limit=32 if name=='Only Text' else 3)
        assert trace['prefix']==reference['prefix'] and trace['deepstack']==reference['deepstack']
        assert value['audit']['blocked_image_probability_max']==0 and value['generated_tokens']>=2
        if c['keep']:assert value['audit']['retained_image_probability_min']>0
        if name=='Only Text':
            assert value['token_ids']==history['Only_Text']['token_ids']
            check_first(kind,f,history,'Only_Text')
            if 'only_step_logits_sha256' in history:
                assert value['all_step_logits_sha256']==history['only_step_logits_sha256']
        checks.append(dict(condition=name,non_target_prefill_exact=True,deepstack_native_exact=True,
            mask_zero=True,retained_path_positive=bool(c['keep']),actual_decode=True))
    repeat,rf,_=generate(kind,model,tokenizer,inputs,layout)
    assert torch.equal(rf,first) and repeat['all_step_logits_sha256']==full['all_step_logits_sha256']
    lm,layers=modules(kind,model);before=hooks_count(model)
    from transformers.cache_utils import DynamicCache
    try:
        with BudgetMask(lm,layers,layout['input_length'],layout['positions']['image'] if kind=='internvl' else layout['image_positions'],
            list(range(len(layers))),DynamicCache()):
            raise RuntimeError('cleanup fixture')
    except RuntimeError as e:assert str(e)=='cleanup fixture'
    assert hooks_count(model)==before
    if runner:runner.verify()
    return dict(passed=True,question_id=sample['question_id'],native_all_tokens_logits_exact=True,
        historical_full_and_only_text_exact=True,non_p1_prefill_exact=True,cache_isolation_exact=True,
        hooks_clean=True,exception_cleanup=True,checks=checks,physical_gpu=MODELS[kind]['gpu'])


@torch.inference_mode()
def worker(kind):
    folder=OUT/kind;folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=verify();binding=sha(OUT/'protocol.json')
    if (folder/'complete.json').exists():return
    preflight(kind)
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();model,processor,fast,loader=load(kind)
    write(folder/'environment.json',environment())
    samples=read(OUT/'sample_manifest.json')['samples'];hist=read(OUT/f'endpoint_bindings/{kind}.json')
    chosen=next(s for s in samples if s['task']=='CompareAttr' and len(hist[s['question_id']]['Full']['token_ids'])>=2
        and len(hist[s['question_id']]['Only_Text']['token_ids'])>=2)
    validated=smoke(kind,model,processor,fast,loader,chosen,hist[chosen['question_id']])
    commit=command('git','rev-parse','HEAD');validated.update(code_commit=commit,protocol_sha256=binding)
    write(folder/'validation.json',validated);print('SMOKE PASS',kind,flush=True)
    while not all((OUT/m/'validation.json').exists() for m in MODELS):
        assert not any((OUT/m/'error.json').exists() for m in MODELS), 'Peer failed; stop before formal experiment'
        time.sleep(2)
    assert all(read(OUT/m/'validation.json')['code_commit']==commit for m in MODELS)
    tokenizer=processor if kind=='internvl' else processor.tokenizer
    configs=read(OUT/'layer_sets.json')['conditions'][kind]
    footprint=hooks_count(model);versions=[(p._version,p.data_ptr()) for p in model.parameters()]
    # Same image always stays on this model's single physical GPU. No sample is filtered.
    samples.sort(key=lambda s:(s['image_sha256'],s['task'],s['question_id']))
    formal=time.perf_counter();new=replayed=0
    for index,s in enumerate(samples,1):
        path=folder/'checkpoints'/f"{s['task']}__{s['question_id']}.json"
        old=hist[s['question_id']]
        record=read(path) if path.exists() else dict(complete=False,task=s['task'],question_id=s['question_id'],
            image_id=s['image_id'],image_sha256=s['image_sha256'],gold=s['answer'],sample_sha256=compact_digest(s),
            protocol_sha256=binding,code_commit=commit,physical_gpu=MODELS[kind]['gpu'],conditions={})
        assert record['sample_sha256']==compact_digest(s) and record['protocol_sha256']==binding and record['code_commit']==commit
        if record['complete']:
            assert len(record['conditions'])==14;continue
        inputs,layout,runner=prepare(kind,model,processor,s,fast,loader)
        actual=compact_digest(layout['input_mapping']) if kind=='llava' else digest(layout)
        assert actual==old['input_binding'], 'Historical Prompt/Processor binding differs; stop rather than silently reuse'
        assert sha(old['source_path'])==old['source_sha256'] and sha(old['protocol_path'])==old['protocol_sha256']
        if 'Full' not in record['conditions']:
            record['conditions']['Full']=dict(old['Full'],source='strict_historical_reuse')
        if 'Only Text' not in record['conditions']:
            off=dict(old['Only_Text'],source='strict_historical_reuse')
            if 'signed_logp_loss' not in off:
                _,f,_=generate(kind,model,tokenizer,inputs,layout,configs['Only Text']['off'],limit=1)
                check_first(kind,f,old,'Only_Text');off.update(token_metrics(f,old['Full']))
                off['first_token_source']='single-prefill replay; complete generation remains historical';replayed+=1
            record['conditions']['Only Text']=off
        record['input_binding']=actual;record['history_binding_sha256']=digest(old)
        write(path,record)
        for name,c in configs.items():
            if name in record['conditions']:continue
            tick=time.perf_counter();g,first,_=generate(kind,model,tokenizer,inputs,layout,c['off'])
            g.update(token_metrics(first,old['Full']),correct=int(g['prediction']==s['answer']),
                seconds=time.perf_counter()-tick,source='new_random_condition')
            record['conditions'][name]=g;new+=1;write(path,record)
        if runner:runner.verify()
        assert hooks_count(model)==footprint
        record['complete']=True;write(path,record)
        elapsed=time.perf_counter()-formal
        write(folder/'progress.json',dict(done=index,total=3000,new_generations=new,first_only_replays=replayed,
            seconds=elapsed,last_sample=s['question_id'],physical_gpu=MODELS[kind]['gpu']))
        if index%25==0 or index==1:print('PROGRESS',kind,index,3000,'new',new,'seconds',round(elapsed,1),flush=True)
    assert versions==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==footprint
    write(folder/'complete.json',dict(passed=True,questions=3000,conditions=14,new_generations=new,
        first_only_replays=replayed,formal_seconds=time.perf_counter()-formal,total_seconds=time.perf_counter()-started,
        model_weights_unchanged=True,hooks_clean=True,no_nan_inf_oom=True,code_commit=commit,
        protocol_sha256=binding,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model;gc.collect();torch.cuda.empty_cache();print('COMPLETE',kind,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker'])
    parser.add_argument('--model',choices=list(MODELS));args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:
        try:worker(args.model)
        except BaseException:
            write(OUT/args.model/'error.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
