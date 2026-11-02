from contextlib import ExitStack

import torch

from ..visual_pathways.sparse_accuracy import GenerationMask, prepare as previous_prepare
from ..internvl3_smoke.engine import tensor_hash, hooks_count


def modules(kind, model):
    lm = model.model.language_model if kind=='qwen' else model.language_model
    layers = lm.layers if kind=='qwen' else lm.model.layers
    return lm,layers


class BudgetMask(GenerationMask):
    """Existing native source mask plus read-only audit of retained image paths."""
    def __enter__(self):
        self.rows=[]
        try:
            super().__enter__()
            for i,layer in enumerate(self.layers):
                def observe(module,args,output,i=i):
                    step=len(self.calls)-1; row=output[1][0,:,-1,:]
                    assert row.shape[-1]==self.n+step and torch.isfinite(row).all()
                    maximum=float(row[:,self.image].max())
                    active=i in self.off and not self.noop
                    assert maximum==0. if active else maximum>0.
                    error=float((row.float().sum(-1)-1).abs().max())
                    assert error <= self.tolerance
                    self.rows.append(dict(step=step,layer=i,blocked=active,image_max=maximum,row_error=error))
                    return (output[0],None)+tuple(output[2:])
                self.stack.callback(layer.self_attn.register_forward_hook(observe).remove)
            return self
        except BaseException:
            self.stack.close();raise


@torch.inference_mode()
def prepare(kind,model,processor,sample,fast=None,loader=None):
    if kind!='llava':return previous_prepare(kind,model,processor,sample,fast,loader)
    from vlm_research.vlm_common import prepare_inputs
    inputs,counts=prepare_inputs(model,processor,sample['image_path'],sample['prompt_text'])
    ids=inputs['input_ids'][0].tolist();mapping=sample['input_mapping']
    assert ids==mapping['prompt_ids'] and counts['visual_tokens']==576
    positions=[i for i,t in enumerate(ids) if t==model.config.image_token_index]
    assert positions==mapping['visual_positions'] and mapping['p_minus_one']==len(ids)-1
    return inputs,dict(input_length=len(ids),image_positions=positions,input_mapping=mapping),None


@torch.inference_mode()
def generate(kind,model,tokenizer,inputs,layout,off=(),*,noop=False,native=False,capture=False,limit=32):
    from transformers.cache_utils import DynamicCache
    lm,layers=modules(kind,model);n=layout['input_length']
    image=layout['positions']['image'] if kind=='internvl' else layout['image_positions']
    cache=DynamicCache(config=model.config.text_config) if kind=='qwen' else DynamicCache()
    footprint=hooks_count(model);fingerprints={k:tensor_hash(v) for k,v in inputs.items()}
    with ExitStack() as stack:
        control=None if native else stack.enter_context(BudgetMask(lm,layers,n,image,off,cache,noop,capture))
        if control:control.tolerance=.002 if kind=='llava' else 2**-8+2e-6
        options=dict(past_key_values=cache,use_cache=True,do_sample=False,num_beams=1,
            max_new_tokens=limit,return_dict_in_generate=True,output_scores=True,
            output_attentions=kind=='llava' and not native,output_hidden_states=False)
        if kind!='llava':options['logits_to_keep']=1
        if kind=='internvl':
            end=tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip());eos=[end]
            output=lm.generate(**inputs,**options,eos_token_id=end,pad_token_id=end)
            ids=output.sequences[0].tolist()
        else:
            if kind=='llava':options['eos_token_id']=2
            output=model.generate(**inputs,**options);ids=output.sequences[0,n:].tolist()
            eos=[2] if kind=='llava' else model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos
        assert ids and len(ids)==len(output.scores) and output.past_key_values is cache
        assert all(torch.isfinite(v).all() for v in output.scores)
        assert all(cache.get_seq_length(i)==n+len(ids)-1 for i in range(len(layers)))
        first=output.scores[0][0].detach().cpu().clone()
        text=tokenizer.decode(ids,skip_special_tokens=True)
        if kind=='internvl':text=text.split(model.conv_template.sep.strip())[0]
        text=text.strip();rows=control.rows if control else []
        if control:
            assert len(control.calls)==len(ids) and len(rows)==len(ids)*len(layers)
            assert sum(r['blocked'] for r in rows)==(0 if noop else len(off)*len(ids))
        value=dict(text=text,prediction=text.rstrip('.').lower(),token_ids=ids,eos=ids[-1] in eos,
            truncated=len(ids)==limit and ids[-1] not in eos,empty=not bool(text),generated_tokens=len(ids),
            fresh_cache=True,final_cache_length=n+len(ids)-1,first_logits_sha256=tensor_hash(first),
            all_step_logits_sha256=[tensor_hash(v) for v in output.scores],
            audit=dict(rows=len(rows),masked_rows=sum(r['blocked'] for r in rows),
                blocked_image_probability_max=max((r['image_max'] for r in rows if r['blocked']),default=0.),
                retained_image_probability_min=min((r['image_max'] for r in rows if not r['blocked']),default=None),
                row_sum_max_error=max((r['row_error'] for r in rows),default=0.)))
        trace=dict(prefix=control.prefix,deepstack=control.deepstack) if control else {}
    assert hooks_count(model)==footprint
    assert fingerprints=={k:tensor_hash(v) for k,v in inputs.items()}
    return value,first,trace
