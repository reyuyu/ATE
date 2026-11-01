"""Fixed L26 rescue under relay OFF; not a new layer/window search."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import time
import traceback

import torch

from .common import ROOT,OUT,QH,EXPERIMENT_END,read,write,sha,digest,preflight,command
from .relay import Controller
from .engine import metrics
from ..internvl3_smoke.engine import hooks_count,tensor_hash
from ..qwen3vl_early_windows.experiment import PrefillTrace,prepare
from ..qwen3vl_image_source.engine import Prefill

DIR=OUT/'qwen_rescue'
CONDITIONS={'FullImageAccess':[],'AllImageOFF':list(range(36)),
            'KeepSparse':[i for i in range(36) if i!=26]}
GEN_HISTORY=ROOT/'vlm_research/results/qwen3vl_keep_l26_generation_240'


class QController(Controller):
    def __init__(self,model,inputs,layout,image_off=(),noop=False):
        self.model,self.inputs=model,inputs
        self.layers=model.model.language_model.layers
        self.n=layout['input_length'];self.image=layout['image_positions']
        assert inputs['input_ids'][0].tolist()==layout['input_ids']
        self.config=dict(relay_off=True,skip=False,image_off=image_off,noop=noop,tolerance=2**-8+2e-6)
        self.capture=False;self.calls=[];self.audits={};self.state_hashes={}


@torch.inference_mode()
def generate(model,processor,inputs,layout,image_off=(),noop=False,trace=False):
    from transformers.cache_utils import DynamicCache
    before=hooks_count(model)
    fingerprints={k:tensor_hash(v) for k,v in inputs.items()}
    cache=DynamicCache(config=model.config.text_config)
    with ExitStack() as stack:
        controller=stack.enter_context(QController(model,inputs,layout,image_off,noop))
        observer=stack.enter_context(PrefillTrace(model,layout['input_length'],layout['image_positions'])) if trace else None
        result=model.generate(**inputs,past_key_values=cache,use_cache=True,do_sample=False,num_beams=1,
            max_new_tokens=32,logits_to_keep=1,return_dict_in_generate=True,output_scores=True,
            output_attentions=False,output_hidden_states=False)
        ids=result.sequences[0,layout['input_length']:].tolist()
        assert ids and len(ids)==len(result.scores) and result.past_key_values is cache
        assert all(torch.isfinite(score).all() for score in result.scores)
        length=layout['input_length']+len(ids)-1
        assert all(cache.get_seq_length(i)==length for i in range(36))
        assert controller.calls==[layout['input_length']]+[1]*(len(ids)-1)
        assert len(controller.audits)==36
        first=result.scores[0][0].detach().float().cpu().clone()
        text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        value=dict(text=text,prediction=text.rstrip('.').lower(),token_ids=ids,
            eos=ids[-1] in eos,truncated=len(ids)==32 and ids[-1] not in eos,empty=not bool(text),generated_tokens=len(ids),
            prefill_mask_audits=controller.audits,decode_hooks_removed=True,fresh_cache=True,
            final_cache_length=length,deepstack='Native throughout')
        if observer:
            observer.validate();assert observer.mlp_calls==[len(ids)]*36
    assert hooks_count(model)==before and fingerprints=={k:tensor_hash(v) for k,v in inputs.items()}
    return value,first,observer


def history(sample):
    oldpath=QH/f'shard_{sample["shard"]}/checkpoints/{sample["task"]}__{sample["question_id"]}.json'
    genpath=GEN_HISTORY/f'shard_{sample["shard"]}/checkpoints/{sample["task"]}__{sample["question_id"]}.json'
    old,gen=read(oldpath),read(genpath)
    assert old['input_binding']==gen['input_binding']==sample['input_binding']
    assert old['protocol_sha256']==sha(QH/'protocol.json') and gen['protocol_sha256']==sha(GEN_HISTORY/'protocol.json')
    assert gen['image_sha256']==sample['image_sha256']
    return old,gen


def freeze():
    assert not (DIR/'protocol.json').exists()
    decision=read(OUT/'llava_relay/extension_decision.json')
    assert not decision['extend_to_other_models'], 'Do not silently omit a required positive-interaction extension'
    samples=read(QH/'sample_manifest.json')['samples']
    original={(s['task'],s['question_id']):s for s in read(OUT/'sample_manifest.json')['samples'] if s['pilot']}
    assert len(samples)==240
    for s in samples:
        assert original[(s['task'],s['question_id'])]['image_sha256']==s['image_sha256']
        history(s)
    write(DIR/'sample_manifest.json',dict(samples=samples,unique_images=len({s['image_sha256'] for s in samples})))
    runtime={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'vlm_research/visual_pathways/qwen_rescue.py',ROOT/'vlm_research/visual_pathways/relay.py',ROOT/'vlm_research/visual_pathways/engine.py']}
    write(DIR/'protocol.json',dict(source_sha256=runtime,manifest_sha256=sha(DIR/'sample_manifest.json'),
        model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',dtype='bfloat16',attention='eager',transformers='4.57.6',
        keep_layers=[26],conditions=CONDITIONS,relay_off=True,decode='Entirely native, own cache per condition',
        deepstack='Native visual token injection after L0/L1/L2 preserved',max_new_tokens=32,
        task2_extension='Not performed: negative LLaVA interaction under frozen positive gate',
        target='Historical native Full top1, exact current last-only native logits SHA verified per question',
        bootstrap=read(OUT/'protocol.json')['bootstrap'],scope='Only predeclared Task3 three conditions, no layer search'))


@torch.inference_mode()
def smoke(model,processor,sample):
    from ..qwen3vl_keep_l26.engine import generate as native_generate
    inputs,layout=prepare(processor,sample)
    assert layout==sample['layout']
    old,gen=history(sample)
    native,first,native_trace=native_generate(model,processor,inputs,layout,native=True,trace=True)
    assert native['token_ids']==gen['conditions']['Full']['token_ids']
    noop,noop_first,_=generate(model,processor,inputs,layout,range(36),noop=True)
    assert native['token_ids']==noop['token_ids'] and torch.equal(first,noop_first)
    assert tensor_hash(first.to(torch.bfloat16))==old['full_logits_sha256']
    reference=None;checks=[]
    for name,layers in CONDITIONS.items():
        result,_,trace=generate(model,processor,inputs,layout,layers,trace=True)
        assert all(a['masked_probability_max']==0 for a in result['prefill_mask_audits'].values())
        for i in range(3):
            assert torch.equal(trace.features[i],native_trace.features[i]), 'DeepStack changed'
        if reference is not None:
            for layer in range(36):
                assert torch.equal(reference.records[layer]['output'][:,:-1],trace.records[layer]['output'][:,:-1])
        else:
            reference=trace
        checks.append(dict(condition=name,all_target_probabilities_zero=True,deepstack_exact=True,
            other_prefill_queries_unchanged_across_direct_access_conditions=True))
    return dict(passed=True,question_id=sample['question_id'],noop_logits_max_abs=0,
        historical_native_generation_exact=True,historical_first_logits_exact=True,checks=checks)


@torch.no_grad()
def worker(gpu):
    p=read(DIR/'protocol.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    assert sha(DIR/'sample_manifest.json')==p['manifest_sha256']
    folder=DIR/f'gpu{gpu}';preflight(gpu,folder)
    from ..qwen3vl_keep_l26.run import load
    model,processor=load(gpu,folder/'loader')
    samples=read(DIR/'sample_manifest.json')['samples']
    validation=smoke(model,processor,samples[0]);validation['code_commit']=command('git','rev-parse','HEAD')
    write(folder/'validation.json',validation);print('QWEN RESCUE SMOKE PASS',flush=True)
    versions=[(p._version,p.data_ptr()) for p in model.parameters()];footprint=hooks_count(model)
    started=time.perf_counter();done=0
    for s in samples:
        if datetime.now().astimezone()>=datetime.fromisoformat(EXPERIMENT_END):break
        key=f'{s["task"]}__{s["question_id"]}';path=DIR/'checkpoints'/f'{key}.json'
        if path.exists() and read(path).get('complete'):
            r=read(path);assert r['sample_sha256']==digest(s) and r['protocol_sha256']==sha(DIR/'protocol.json')
            done+=1;continue
        inputs,layout=prepare(processor,s);assert layout==s['layout'] and digest(layout)==s['input_binding']
        old,gen=history(s)
        runner=Prefill(model,inputs.to('cuda:0'),layout)
        full,_,_=runner.forward(native=True)
        assert tensor_hash(full)==old['full_logits_sha256'], 'Current native Full differs from history'
        conditions={}
        for name,layers in CONDITIONS.items():
            result,changed,_=generate(model,processor,inputs.to('cuda:0'),layout,layers)
            result.update(metrics(full,changed),correct=int(result['prediction']==s['answer']),physical_gpu=gpu)
            conditions[name]=result
        runner.verify()
        write(path,dict(complete=True,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
            image_sha256=s['image_sha256'],sample_sha256=digest(s),protocol_sha256=sha(DIR/'protocol.json'),
            input_binding=s['input_binding'],full_logits_sha256=old['full_logits_sha256'],
            native_full_accuracy=gen['conditions']['Full']['correct'],conditions=conditions))
        done+=1
        write(folder/'progress.json',dict(done=done,total=240,seconds=time.perf_counter()-started))
        if done%20==0:print('QWEN RESCUE',done,240,round(time.perf_counter()-started,1),flush=True)
    assert versions==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==footprint
    write(folder/'complete.json',dict(passed=True,done=done,total=240,seconds=time.perf_counter()-started,
        weights_unchanged=True,hooks_clean=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker']);parser.add_argument('--gpu',type=int,choices=[0,2,3],default=3);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:
        try:worker(args.gpu)
        except BaseException:
            write(DIR/f'gpu{args.gpu}'/'error.json',dict(traceback=traceback.format_exc()));raise
