"""Only native language-attention input masks change; generation stays native."""
from contextlib import AbstractContextManager

import torch

from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_query_role.engine import generate as native_generate


def roles(layout,vision_end_id):
    image = layout['image_positions']
    n = layout['input_length']
    assert image and image==list(range(image[0],image[-1]+1))
    assert layout['input_ids'][image[-1]+1]==vision_end_id and layout['p_minus_1']==n-1
    # The closing visual marker can relay image information, so it belongs to R.
    relay = list(range(image[-1]+1,n-1))
    assert relay and set(layout['question_positions'])<=set(relay)
    return dict(I=image,R=relay,A=[n-1])


def blocked_edges(mapping,writers,n,past,device):
    assert set(writers)<=set(mapping['R']) and (past==0 or past>=n)
    blocked = torch.zeros((1 if past else n,past+1 if past else n),dtype=torch.bool,device=device)
    queries = [0] if past else [q for q in mapping['R'] if q not in set(writers)]+[n-1]
    q = torch.tensor(queries,device=device)
    k = torch.tensor(mapping['I'],device=device)
    blocked[q[:,None],k[None,:]] = True
    return blocked


def apply_mask(old,blocked,past,noop=False,dtype=torch.bfloat16):
    q,k = blocked.shape
    if old is None:
        base = torch.zeros((1,1,q,k),dtype=dtype,device=blocked.device)
        if not past:
            base.masked_fill_(torch.ones(q,k,dtype=torch.bool,device=blocked.device).triu(1),-torch.inf)
    else:
        assert old.shape==(1,1,q,k) and old.is_floating_point()
        assert ((old==0)|(old==torch.finfo(old.dtype).min)|torch.isneginf(old)).all()
        base = old
    legal = base[0,0]==0
    assert not (blocked & ~legal).any()
    if not past:
        assert not legal.triu(1).any()
    if noop:
        return old
    changed = base.clone()
    changed[0,0].masked_fill_(blocked,-torch.inf)
    assert torch.equal(changed[0,0][~blocked],base[0,0][~blocked])
    assert (changed[0,0]==0).any(-1).all()
    assert (changed[0,0,:,0]==0).all()
    if not past:
        assert (changed[0,0].diagonal()==0).all()
    else:
        assert changed[0,0,0,past]==0
    return changed


class WriterMask(AbstractContextManager):
    def __init__(self,model,spec,writers,noop=False):
        self.model,self.spec,self.writers,self.noop = model,spec,list(writers),noop
        self.handles,self.events,self.positions,self.state = [],[],[],{}

    def __enter__(self):
        try:
            def position_audit(module,args,kwargs):
                position = kwargs['position_ids']
                cp = kwargs['cache_position']
                t = len(self.positions)
                n = self.spec['layout']['input_length']
                if t==0:
                    assert tensor_digest(position)==self.spec['mrope']['position_ids_sha256']
                    assert cp.tolist()==list(range(n))
                else:
                    expected = n+t-1+self.spec['mrope']['delta']
                    assert position.shape==(3,1,1) and (position==expected).all()
                    assert cp.tolist()==[n+t-1]
                self.positions.append(dict(step=t,position_ids_sha256=tensor_digest(position),
                    shape=list(position.shape),cache_position=cp.tolist() if t else [0,n-1],native_mrope=True))
            self.handles.append(self.model.model.language_model.register_forward_pre_hook(position_audit,with_kwargs=True))
            for i,layer in enumerate(self.model.model.language_model.layers):
                def before(module,args,kwargs,i=i):
                    hidden,cache,cp = kwargs['hidden_states'],kwargs.get('past_key_values'),kwargs['cache_position']
                    past = cache.get_seq_length(i) if cache is not None else 0
                    n = self.spec['layout']['input_length']
                    assert hidden.shape[:2]==(1,1 if past else n)
                    assert cp.tolist()==([past] if past else list(range(n)))
                    blocked = blocked_edges(self.spec['roles'],self.writers,n,past,hidden.device)
                    changed = apply_mask(kwargs['attention_mask'],blocked,past,self.noop,hidden.dtype)
                    self.state[i] = past,blocked
                    return args,dict(kwargs,attention_mask=changed)
                def after(module,args,output,i=i):
                    past,blocked = self.state[i]
                    weights = output[1]
                    assert weights.shape[0]==1 and weights.shape[2:]==blocked.shape
                    rows = weights[0,:,([0] if past else self.spec['roles']['R']+[self.spec['layout']['p_minus_1']]),:].float()
                    assert torch.isfinite(rows).all() and (rows>=0).all()
                    error = float((rows.sum(-1)-1).abs().max())
                    assert error<=torch.finfo(weights.dtype).eps
                    maximum = float(weights[0,:,blocked].max())
                    if not self.noop:
                        assert maximum==0, 'Prohibited attention probability is nonzero'
                    self.events.append(dict(layer=i,step=len(self.positions)-1,phase='decode' if past else 'prefill',
                        absolute_start=past,query_length=blocked.shape[0],key_length=blocked.shape[1],
                        blocked_connections_per_head=int(blocked.sum()),blocked_probability_max=maximum,
                        row_sum_max_error=error,writer_queries=len(self.writers),non_target_mask_exact=True,
                        prefix_self_legal=True,noop=self.noop))
                    return (output[0],None)
                self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def generate(model,processor,inputs,spec,condition,noop=False,audit=False,trace=False):
    before = hook_counts(model)
    writers = spec['selection']['writers'][condition]
    try:
        with WriterMask(model,spec,writers,noop) as ctl:
            result,scores,observer,cache = native_generate(model,processor,inputs,spec['roles'],
                condition='Full',native=True,audit=audit,trace=trace)
        assert len(ctl.events)==36*len(result['token_ids']) and not ctl.handles
        result.update(source_events=ctl.events,mrope_events=ctl.positions,writer_positions=writers,
            writer_count=len(writers),relay_count=len(spec['roles']['R']),noop=noop,
            logits_to_keep=1,all_36_answer_image_blocked=not noop,independent_native_cache=True)
        return result,scores,observer,cache
    finally:
        assert hook_counts(model)==before, 'Mask hook leaked'
