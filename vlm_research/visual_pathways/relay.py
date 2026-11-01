"""Fixed 2x2 relay/attention-output experiment, with native cached decoding."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import os
import time
import traceback

import torch

from .common import ROOT,OUT,EXPERIMENT_END,TASKS,read,write,sha,digest,assign,verify,preflight,command
from .engine import PrefillIntervention,metrics,PrefixTrace
from ..internvl3_smoke.engine import hooks_count,tensor_hash

RDIR=OUT/'llava_relay'
CONDITIONS={'A':(False,False),'B':(False,True),'C':(True,False),'D':(True,True)}
MODEL='/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9'


class Controller:
    def __init__(self,model,inputs,relay_off=False,skip=False,image_off=(),noop=False,capture=False):
        self.model,self.inputs=model,inputs
        self.layers=model.language_model.model.layers
        self.n=inputs['input_ids'].shape[1]
        self.image=(inputs['input_ids'][0]==model.config.image_token_index).nonzero().flatten().tolist()
        assert len(self.image)==576 and self.image[-1]<self.n-1
        self.config=dict(relay_off=relay_off,skip=skip,image_off=image_off,noop=noop)
        self.capture=capture
        self.calls=[];self.audits={};self.state_hashes={}

    def __enter__(self):
        self.stack=ExitStack()
        self.active=ExitStack()
        def before(module,args,kwargs):
            ids=kwargs.get('input_ids')
            length=ids.shape[1] if ids is not None else kwargs['inputs_embeds'].shape[1]
            assert length==(self.n if not self.calls else 1)
            self.calls.append(length)
            if len(self.calls)==1:
                self.edit=self.active.enter_context(PrefillIntervention(self.layers,self.n,self.image,**self.config))
                if self.capture:
                    for i,layer in enumerate(self.layers):
                        def observe(module,args,out,i=i):
                            value=out[0]
                            self.state_hashes[i]=dict(prefix=tensor_hash(value[:,:-1]),image=tensor_hash(value[:,self.image]))
                        self.active.callback(layer.register_forward_hook(observe).remove)
            else:
                assert all(not edit.handles for edit in self.edit.masks.values())
        def after(module,args,output):
            if len(self.calls)==1:
                assert all(m.calls==1 for m in self.edit.masks.values())
                self.audits={i:m.audit for i,m in self.edit.masks.items()}
                if self.config['skip']:
                    assert self.edit.events==[2,3,4,5]
                self.active.close()
        self.stack.callback(self.model.register_forward_pre_hook(before,with_kwargs=True).remove)
        self.stack.callback(self.model.register_forward_hook(after).remove)
        return self

    def __exit__(self,*args):
        self.active.close();self.stack.close()


@torch.inference_mode()
def generate(model,processor,inputs,relay_off=False,skip=False,image_off=(),noop=False,native=False,capture=False):
    from transformers.cache_utils import DynamicCache
    cache=DynamicCache()
    footprint=hooks_count(model)
    n=inputs['input_ids'].shape[1]
    input_hashes={k:tensor_hash(v) for k,v in inputs.items()}
    with ExitStack() as stack:
        control=None if native else stack.enter_context(Controller(model,inputs,relay_off,skip,image_off,noop,capture))
        output=model.generate(**inputs,past_key_values=cache,use_cache=True,do_sample=False,num_beams=1,
            max_new_tokens=32,logits_to_keep=1,return_dict_in_generate=True,output_scores=True,
            output_attentions=False,output_hidden_states=False)
        ids=output.sequences[0,n:].tolist()
        assert ids and len(output.scores)==len(ids) and output.past_key_values is cache
        assert all(torch.isfinite(score).all() for score in output.scores)
        assert all(cache.get_seq_length(i)==n+len(ids)-1 for i in range(32))
        if control:
            assert control.calls==[n]+[1]*(len(ids)-1)
        first=output.scores[0][0].detach().cpu().clone()
        text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        result=dict(text=text,prediction=text.rstrip('.').lower(),token_ids=ids,
            eos=ids[-1] in eos,truncated=len(ids)==32 and ids[-1] not in eos,empty=not bool(text),
            generated_tokens=len(ids),prefill_mask_audits=control.audits if control else {},
            decode_hooks_removed=True,fresh_cache=True,final_cache_length=n+len(ids)-1)
    assert hooks_count(model)==footprint and input_hashes=={k:tensor_hash(v) for k,v in inputs.items()}
    return result,first,control.state_hashes if control else {}


def prepare(model,processor,sample):
    from ..vlm_common import prepare_inputs
    assert sha(sample['image_path'])==sample['image_sha256']
    inputs,_=prepare_inputs(model,processor,sample['image_path'],sample['prompt_text'])
    assert inputs['input_ids'][0].tolist()==sample['input_mapping']['prompt_ids']
    return inputs


@torch.inference_mode()
def smoke(model,processor,sample):
    inputs=prepare(model,processor,sample)
    native,native_first,_=generate(model,processor,inputs,native=True)
    full,first,full_states=generate(model,processor,inputs,capture=True)
    noop,noop_first,_=generate(model,processor,inputs,True,True,noop=True)
    assert torch.equal(native_first,first) and torch.equal(first,noop_first)
    assert native['token_ids']==full['token_ids']==noop['token_ids']
    checks=[]
    for name,(relay_off,skip) in list(CONDITIONS.items())[1:]:
        result,_,states=generate(model,processor,inputs,relay_off,skip,capture=True)
        assert all(states[i]['image']==full_states[i]['image'] for i in range(32)), 'Native visual computation changed'
        if name=='B':
            assert states==full_states, 'P-1 skip changed non-P-1 Prefill states'
        assert all(a['masked_probability_max']==0 for a in result['prefill_mask_audits'].values())
        assert len(result['prefill_mask_audits'])==(32 if relay_off else 0)
        checks.append(dict(condition=name,image_states_exact=True,mask_layers=len(result['prefill_mask_audits']),
            target_probability_zero=True,native_decode=True,independent_cache=True))
    footprint=hooks_count(model)
    try:
        with Controller(model,inputs,True,True):
            raise RuntimeError('cleanup fixture')
    except RuntimeError:
        pass
    assert footprint==hooks_count(model)
    return dict(passed=True,question_id=sample['question_id'],no_op_logits_max_abs=0,
        native_generation_exact=True,exception_cleanup=True,checks=checks)


def freeze():
    verify()
    assert not (RDIR/'protocol.json').exists()
    samples=[s for s in read(OUT/'sample_manifest.json')['samples'] if s['pilot']]
    assert len(samples)==240
    write(RDIR/'sample_manifest.json',dict(samples=samples,unique_images=len({s['image_sha256'] for s in samples})))
    runtime={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'vlm_research/visual_pathways/relay.py',ROOT/'vlm_research/visual_pathways/engine.py',ROOT/'vlm_research/visual_pathways/test_relay.py',ROOT/'vlm_research/vlm_common.py']}
    write(RDIR/'protocol.json',dict(parent_protocol_sha256=sha(OUT/'protocol.json'),source_sha256=runtime,
        manifest_sha256=sha(RDIR/'sample_manifest.json'),conditions=CONDITIONS,model_path=MODEL,
        model_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',dtype='float16',transformers='4.49.0',attention='eager',
        relay='All nonvisual Prefill queries before P-1; all layers Image-source hard mask on causal-legal edges; no Decode mask',
        skip='P-1 L2-L5 o_proj only, removed before Decode',max_new_tokens=32,use_cache=True,
        first_projection='Identical logits_to_keep=1 for native, no-op and every condition',
        bootstrap=read(OUT/'protocol.json')['bootstrap'],interaction_gate=read(OUT/'protocol.json')['task2']['extension_gate']))


@torch.no_grad()
def worker(args):
    protocol=read(RDIR/'protocol.json')
    assert all(sha(ROOT/p)==v for p,v in protocol['source_sha256'].items())
    assert sha(RDIR/'sample_manifest.json')==protocol['manifest_sha256']
    samples=assign(read(RDIR/'sample_manifest.json')['samples'],args.shards)[args.shard]
    folder=RDIR/f'gpu{args.gpu}_shard{args.shard}of{args.shards}'
    preflight(args.gpu,folder)
    from ..vlm_common import load_model
    model,processor,environment=load_model(MODEL)
    write(folder/'environment.json',environment)
    validation=smoke(model,processor,samples[0])
    validation['code_commit']=command('git','rev-parse','HEAD')
    write(folder/'validation.json',validation)
    print('RELAY SMOKE PASS',args.gpu,flush=True)
    versions=[(p._version,p.data_ptr()) for p in model.parameters()]
    before=hooks_count(model)
    started=time.perf_counter();done=0
    for sample in samples:
        if datetime.now().astimezone()>=datetime.fromisoformat(EXPERIMENT_END):
            break
        key=f'{sample["task"]}__{sample["question_id"]}'
        path=RDIR/'checkpoints'/f'{key}.json'
        record=read(path) if path.exists() else dict(complete=False,task=sample['task'],question_id=sample['question_id'],
            image_id=sample['image_id'],image_sha256=sample['image_sha256'],answer=sample['answer'],
            sample_sha256=digest(sample),protocol_sha256=sha(RDIR/'protocol.json'),conditions={})
        assert record['sample_sha256']==digest(sample) and record['protocol_sha256']==sha(RDIR/'protocol.json')
        if record.get('complete'):
            done+=1;continue
        inputs=prepare(model,processor,sample)
        full=None
        for name,(relay_off,skip) in CONDITIONS.items():
            result,first,_=generate(model,processor,inputs,relay_off,skip)
            if name=='A':
                full=first
                record['full_logits_sha256']=tensor_hash(full)
            result.update(metrics(full,first))
            result.update(correct=int(result['prediction']==sample['answer']),physical_gpu=args.gpu,
                input_binding=digest({k:tensor_hash(v) for k,v in inputs.items()}))
            if name in record['conditions']:
                assert record['conditions'][name]['token_ids']==result['token_ids']
            record['conditions'][name]=result
            write(path,record)
        record['complete']=True
        write(path,record)
        done+=1
        write(folder/'progress.json',dict(done=done,total=len(samples),seconds=time.perf_counter()-started,last_sample=key))
        if done%10==0:
            print('RELAY',done,len(samples),round(time.perf_counter()-started,1),flush=True)
    assert versions==[(p._version,p.data_ptr()) for p in model.parameters()] and before==hooks_count(model)
    write(folder/'complete.json',dict(passed=True,done=done,total=len(samples),seconds=time.perf_counter()-started,
        weights_unchanged=True,hooks_clean=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker'])
    parser.add_argument('--gpu',type=int,choices=[0,2,3]);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
    if args.action=='freeze':
        freeze()
    else:
        try:
            worker(args)
        except BaseException:
            write(RDIR/f'gpu{args.gpu}_shard{args.shard}of{args.shards}'/'error.json',dict(traceback=traceback.format_exc()))
            raise
