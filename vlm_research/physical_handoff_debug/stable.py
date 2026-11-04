"""Temporary, matched numerical kernels; no persistent FP32 weights or extra visual rows."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import importlib

import torch
from torch.nn import functional as F


def linear_fp32(module, x):
    return F.linear(x.float(), module.weight.float(),
                    None if module.bias is None else module.bias.float()).to(x.dtype)


def linear_tiles(module, x, tile=32):
    flat=x.reshape(-1,x.shape[-1])
    if flat.shape[0]==1:
        return F.linear(x,module.weight,module.bias)
    result=[]
    for chunk in flat.split(tile):
        padded=F.pad(chunk,(0,0,0,tile-len(chunk)))
        result.append(F.linear(padded,module.weight,module.bias)[:len(chunk)])
    return torch.cat(result).reshape(*x.shape[:-1],module.out_features)


@contextmanager
def numerical_kernels(adapter, candidate):
    assert candidate in ('native', 'fp32_linears', 'fp32_linears_attention', 'canonical_tiles')
    cutoff = 20 if adapter.kind == 'llava' else 22
    with ExitStack() as stack:
        if candidate != 'native':
            modules = [m for layer in adapter.layers[cutoff:] for m in layer.modules()
                       if isinstance(m, torch.nn.Linear)]
            modules.append(adapter.model.get_output_embeddings())
            for module in modules:
                stack.enter_context(patch.object(module, 'forward',
                    lambda x, module=module: (linear_tiles if candidate=='canonical_tiles' else linear_fp32)(module, x)))
        if candidate in ('fp32_linears_attention','canonical_tiles'):
            source = importlib.import_module(adapter.layers[-1].self_attn.__class__.__module__)
            native = source.eager_attention_forward
            caches={}
            if candidate=='canonical_tiles':
                for layer in adapter.layers[cutoff:]:
                    def remember(module,args,kwargs):
                        caches[id(module)]=kwargs.get('past_key_values',kwargs.get('past_key_value'))
                    handle=layer.self_attn.register_forward_pre_hook(remember,with_kwargs=True)
                    stack.callback(handle.remove)

            def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
                if getattr(module,'layer_idx',-1) < cutoff:
                    return native(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)
                assert dropout == 0.
                if candidate=='canonical_tiles':
                    cache=caches[id(module)]
                    keys=cache.entries[module.layer_idx][2]
                    queries=cache.current_positions
                    image_id=adapter.model.config.image_token_index if adapter.kind=='llava' else adapter.model.config.image_token_id
                    image=torch.where(adapter.inputs['input_ids'][0]==image_id)[0]
                    tq=torch.where(~torch.isin(queries,image))[0]
                    vq=torch.where(torch.isin(queries,image))[0]
                    tk=torch.where(~torch.isin(keys,image))[0]
                    # Every removed edge has zero probability in the dense graph after cutoff.
                    ik=torch.where(torch.isin(keys,image))[0]
                    if len(ik):
                        forbidden=attention_mask.index_select(2,tq).index_select(3,ik)
                        assert (forbidden<0).all()
                    output=query.new_empty(query.shape[0],query.shape[2],query.shape[1],query.shape[3])
                    weights=query.new_zeros(query.shape[0],query.shape[1],query.shape[2],key.shape[2])
                    tm=attention_mask.index_select(2,tq).index_select(3,tk)
                    text,tw=native(module,query.index_select(2,tq),key.index_select(2,tk),value.index_select(2,tk),
                                   tm,scaling=scaling,dropout=dropout,**kwargs)
                    output.index_copy_(1,tq,text)
                    weights[:,:,tq[:,None],tk[None,:]]=tw
                    if len(vq):
                        visual,vw=native(module,query.index_select(2,vq),key,value,attention_mask.index_select(2,vq),
                                         scaling=scaling,dropout=dropout,**kwargs)
                        output.index_copy_(1,vq,visual)
                        weights.index_copy_(2,vq,vw)
                    return output,weights
                k = source.repeat_kv(key, module.num_key_value_groups)
                v = source.repeat_kv(value, module.num_key_value_groups)
                # Keep the frozen low-precision boundaries, changing GEMM accumulation only.
                scores = (query.float() @ k.float().transpose(2, 3)).to(query.dtype) * scaling
                if attention_mask is not None:
                    scores = scores + attention_mask[..., :k.shape[-2]]
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                output = (weights.float() @ v.float()).to(query.dtype).transpose(1, 2).contiguous()
                return output, weights

            stack.enter_context(patch.object(source, 'eager_attention_forward', attention))
        yield
