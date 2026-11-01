"""Reuse RQ2.1 partitions and RQ2.2 empty-cache hard masks, with native Decode."""
from contextlib import nullcontext

import torch

from knockout_probe import model_layers
from paper_rq1.engine import step_inputs
from rq2_source_knockout.engine import Observer
from rq2_visual_sufficiency.engine import MultiImageKnockout

S = [1,13,14,15,16]
CONDITIONS = dict(A=[1], B=[13,14,15,16], C=S,
                  D=[k for k in range(32) if k not in S], E=list(range(32)))
LABELS = dict(Full='Full',A='Drop L1',B='Drop L13-L16',C='Drop five',D='Keep five',E='All Prompt OFF')


class MultiPromptKnockout(MultiImageKnockout):
    def __init__(self,model,inputs,bos_id,blocked,noop=False,expected_visual=576):
        super().__init__(model,inputs,bos_id,blocked,noop,expected_visual)
        # Exact RQ2.1 partition already constructed by the parent SourceKnockout.
        self.layout=dict(self.layout,visual=self.groups['Prompt'])
        self.source='Prompt'


@torch.inference_mode()
def native_reference(model,inputs,bos_id,blocked,limit=32,expected_visual=576):
    """Independent HF greedy driver; remove masks before it enters Decode."""
    mask=MultiPromptKnockout(model,inputs,bos_id,blocked,expected_visual=expected_visual)
    n=inputs['input_ids'].shape[1]
    def release(module,args,output):
        if mask.handles:
            assert args[0].shape[1]==n
            mask.assert_one_forward()
            mask.__exit__(None,None,None)
    handle=model_layers(model)[-1].register_forward_hook(release)
    try:
        with mask:
            driver=model if hasattr(model,'generate') else model.language_model
            result=driver.generate(**inputs,do_sample=False,num_beams=1,
                max_new_tokens=limit,use_cache=True,eos_token_id=2)
        assert not mask.handles and len(mask.records)==len(model_layers(model))
        return result[0,n:].tolist()
    finally:
        handle.remove()
        mask.__exit__(None,None,None)


@torch.inference_mode()
def generate(model,inputs,bos_id,blocked,limit=32,noop=False,audit=False,
             collect=False,observe=False,expected_visual=576):
    if limit<1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('Positive limit and unpadded original prompt required')
    n=inputs['input_ids'].shape[1]
    mask=MultiPromptKnockout(model,inputs,bos_id,blocked,noop,expected_visual)
    observer=Observer(model,inputs,bos_id,audit,expected_visual) if audit or observe else None
    tokens,history,trace,debug=[],None,[],{}
    first=None
    while len(tokens)<limit and (not tokens or tokens[-1]!=2):
        if history is not None and any(history.get_seq_length(k)!=n+len(tokens)-1 for k in range(len(model_layers(model)))):
            raise AssertionError('Own-prefix cache lengths differ')
        with mask if not tokens else nullcontext():
            with observer if not tokens and observer is not None else nullcontext():
                output=model(**step_inputs(inputs,tokens),past_key_values=history,use_cache=True,
                    output_attentions=not tokens and observer is not None,return_dict=True)
        if not tokens:
            mask.assert_one_forward()
            active=[r for r in mask.records if r['applied']]
            expected=[] if noop else sorted(blocked)
            if [r['layer'] for r in active]!=expected:
                raise AssertionError('Wrong layer scope')
            if any(r['blocked_connections_per_head']!=int(mask.groups['Prompt'].sum()) for r in active):
                raise AssertionError('Wrong Prompt key count or query')
            if bool((mask.layout['visual']&(mask.groups['Image']|mask.groups['BOS']|mask.groups['Self'])).any()):
                raise AssertionError('Self, image or BOS incorrectly masked')
            first=output.logits[0,n-1].detach().float().cpu().clone()
            if observer is not None:
                if [r['layer'] for r in observer.rows]!=list(range(len(model_layers(model)))):
                    raise AssertionError('Missing P-1 rows')
                if any(observer.rows[k]['Prompt_max']!=0 for k in expected):
                    raise AssertionError('Prompt probability nonzero after hard mask')
                debug['rows']=observer.rows
            if audit:
                debug.update(prefill_logits=output.logits.detach().float().cpu().clone(),observer=observer)
        history=output.past_key_values
        logits=output.logits[0,-1]
        if not torch.isfinite(logits).all() or history is None:
            raise AssertionError('Invalid prediction or missing native cache')
        if mask.handles or len(mask.records)!=len(model_layers(model)):
            raise AssertionError('Prompt masks leaked into Decode')
        if collect:trace.append(logits.detach().float().cpu().clone())
        tokens.append(int(logits.argmax()))
    return dict(token_ids=tokens,length=len(tokens),eos=tokens[-1]==2,
        truncated=len(tokens)==limit and tokens[-1]!=2,decode_steps=len(tokens)-1,
        masked_layers=[] if noop else sorted(blocked),mask_calls=len(mask.records),
        own_prefill=True,own_prefix=True,decode_native=True),first,trace,debug
