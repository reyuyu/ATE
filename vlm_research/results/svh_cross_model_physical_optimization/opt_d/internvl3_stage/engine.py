"""Minimal query-local multi-layer hook; exact native embedding replay."""
from contextlib import ExitStack
import torch
from transformers.cache_utils import DynamicCache

from .common import prediction
from vlm_research.internvl3_smoke.engine import hooks_count, cache_audit, tensor_hash


class WindowHook:
    def __init__(self, layers, targets, prompt_length, noop=False):
        self.layers, self.targets, self.length, self.noop = layers, list(targets), prompt_length, noop
        self.calls = {i:[] for i in targets}

    def __enter__(self):
        self.stack = ExitStack()
        for index in self.targets:
            def hook(module,args,output,index=index):
                assert output.ndim==3 and output.shape[0]==1
                n=output.shape[1]
                assert n in (self.length,1), ('Not original prefill or incremental decode',n)
                prior=len(self.calls[index])
                assert n==self.length if prior==0 else n==1
                self.calls[index].append(n)
                if self.noop:
                    return output
                result=output.clone()
                result[:,-1,:]=0
                assert torch.count_nonzero(result[:,-1])==0
                if n>1:
                    assert torch.equal(result[:,:-1],output[:,:-1])
                return result
            self.stack.callback(self.layers[index].self_attn.o_proj.register_forward_hook(hook).remove)
        return self

    def __exit__(self,*args):
        self.stack.close()


def first_metrics(full, off):
    full,off=full.float(),off.float()
    assert torch.isfinite(full).all() and torch.isfinite(off).all()
    a,b=full.log_softmax(-1),off.log_softmax(-1)
    token=int(full.argmax())
    midpoint=torch.logaddexp(a,b)-torch.log(torch.tensor(2.))
    js=float(.5*((a.exp()*(a-midpoint)).sum()+(b.exp()*(b-midpoint)).sum()))
    loss=float(a[token]-b[token])
    return dict(full_first_token=token,off_first_token=int(off.argmax()),signed_logp_loss=loss,
                absolute_logp_change=abs(loss),top1_flip=int(full.argmax()!=off.argmax()),js_divergence=js)


@torch.inference_mode()
def forward(model,saved):
    inputs={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in saved.items() if k!='past_key_values'}
    inputs['past_key_values']=DynamicCache()
    output=model.language_model(**inputs)
    assert torch.isfinite(output.logits).all()
    assert output.past_key_values.get_seq_length()==inputs['inputs_embeds'].shape[1]
    return output.logits[0,-1].detach().cpu().clone()


@torch.no_grad()
def generate(model,tokenizer,saved,targets=(),noop=False,audit=False):
    n=saved['inputs_embeds'].shape[1]
    baseline=hooks_count(model)
    calls=[]
    def inspect(module,args,kwargs,out):
        logits=out.logits[0,-1]
        assert torch.isfinite(out.logits).all()
        q=kwargs['inputs_embeds'].shape[1] if kwargs.get('inputs_embeds') is not None else kwargs['input_ids'].shape[1]
        assert q==n if not calls else q==1
        expected=n+len(calls)
        assert all(out.past_key_values.get_seq_length(i)==expected for i in range(len(model.language_model.model.layers)))
        calls.append(dict(query_length=q,cache_length=expected,first_logits=logits.detach().cpu().clone() if not calls else None))
    with ExitStack() as stack:
        hook=stack.enter_context(WindowHook(model.language_model.model.layers,targets,n,noop))
        prefixes={}
        mlp_counts={i:0 for i in range(len(model.language_model.model.layers))}
        if audit:
            for index,layer in enumerate(model.language_model.model.layers):
                def layer_output(module,args,out,index=index):
                    if out[0].shape[1]==n:
                        prefixes[index]=tensor_hash(out[0][:,:-1])
                def mlp_output(module,args,out,index=index):
                    mlp_counts[index]+=1
                    assert torch.isfinite(out).all()
                stack.callback(layer.register_forward_hook(layer_output).remove)
                stack.callback(layer.mlp.register_forward_hook(mlp_output).remove)
        stack.callback(model.language_model.register_forward_hook(inspect,with_kwargs=True).remove)
        result=model.language_model.generate(inputs_embeds=saved['inputs_embeds'].to('cuda:0'),
            attention_mask=saved['attention_mask'].to('cuda:0'),use_cache=True,do_sample=False,num_beams=1,
            max_new_tokens=32,eos_token_id=tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip()))
        ids=result[0].detach().cpu().tolist()
        assert len(calls)==len(ids) and ids
        for index in targets:
            assert hook.calls[index]==[n]+[1]*(len(ids)-1), 'Mask must affect every real decode query'
    assert hooks_count(model)==baseline
    text=tokenizer.batch_decode(result,skip_special_tokens=True)[0].split(model.conv_template.sep.strip())[0].strip()
    eos=ids[-1]==tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    value=dict(answer=text,normalized=prediction(text),token_ids=ids,first_token=ids[0],eos=eos,
               truncated=len(ids)==32 and not eos,empty=not bool(text),length=len(ids),
               repeated=len(ids)>=6 and any(ids[i:i+3]==ids[i+3:i+6] for i in range(len(ids)-5)),
               hook_calls=hook.calls,cache_trace=[{k:v for k,v in c.items() if k!='first_logits'} for c in calls])
    if audit:
        assert all(count==len(ids) for count in mlp_counts.values())
        value.update(prefill_prefix_hashes=prefixes,all_mlp_calls=mlp_counts)
    return value,calls[0]['first_logits']
