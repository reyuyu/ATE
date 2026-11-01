"""Small hooks around native eager attention; no replacement forward kernels."""
from contextlib import ExitStack
import torch

from ..internvl3_smoke.engine import hooks_count, tensor_hash


def masked(mask, queries, keys):
    assert mask is not None and mask.ndim == 4 and mask.shape[:2] == (1,1)
    assert queries and keys and len(queries) == len(set(queries)) and len(keys) == len(set(keys))
    q = torch.tensor(queries, device=mask.device)
    k = torch.tensor(keys, device=mask.device)
    assert int(q.max()) < mask.shape[-2] and int(k.max()) < mask.shape[-1]
    result = mask.clone()
    # Only causal-legal edges are edited, preserving all pre-existing causal/pad mask entries.
    selected = result[0,0,q[:,None],k[None,:]]
    legal = selected == 0
    selected = selected.masked_fill(legal, -torch.inf)
    result[0,0,q[:,None],k[None,:]] = selected
    assert (result[0,0] == 0).any(-1).all(), 'A query lost all legal keys'
    return result, int(legal.sum())


class SourceMask:
    def __init__(self, module, queries, keys, noop=False, row_tolerance=2**-8+2e-6):
        self.module, self.queries, self.keys = module, queries, keys
        self.noop, self.tolerance = noop, row_tolerance
        self.calls, self.audit = 0, {}

    def __enter__(self):
        def before(module, args, kwargs):
            self.calls += 1
            assert self.calls == 1, 'One original Prefill, no Decode mask'
            original = kwargs['attention_mask']
            changed, count = (original,0) if self.noop else masked(original,self.queries,self.keys)
            self.audit['changed_entries'] = int((original != changed).sum())
            assert self.audit['changed_entries'] == count
            return args, dict(kwargs,attention_mask=changed)
        def after(module,args,output):
            weights = output[1]
            assert weights is not None
            rows = weights[0,:,self.queries,:]
            assert torch.isfinite(rows).all() and (rows >= 0).all()
            maximum = float(rows[...,self.keys].max())
            error = float((rows.float().sum(-1)-1).abs().max())
            assert self.noop or maximum == 0
            assert error <= self.tolerance
            self.audit.update(masked_probability_max=maximum,row_sum_max_error=error)
        self.handles = [self.module.register_forward_pre_hook(before,with_kwargs=True),self.module.register_forward_hook(after)]
        return self

    def __exit__(self,*args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class PrefillIntervention:
    """Only active around one prefill; native decode sees no intervention hooks."""
    def __init__(self,layers,n,image,relay_off=False,skip=False,image_off=(),noop=False,tolerance=.002):
        self.layers,self.n,self.image = layers,n,image
        self.relay_off,self.skip,self.image_off = relay_off,skip,list(image_off)
        self.noop,self.tolerance = noop,tolerance
        self.masks,self.events = {},[]

    def __enter__(self):
        self.stack = ExitStack()
        visual = set(self.image)
        relay = [i for i in range(self.n-1) if i not in visual]
        for i,layer in enumerate(self.layers):
            queries = (relay if self.relay_off else []) + ([self.n-1] if i in self.image_off else [])
            if queries:
                self.masks[i] = self.stack.enter_context(SourceMask(layer.self_attn,queries,self.image,self.noop,self.tolerance))
            if self.skip and i in (2,3,4,5):
                def output(module,args,value,i=i):
                    assert value.shape[1] == self.n
                    self.events.append(i)
                    if self.noop:
                        return value
                    result = value.clone()
                    result[:,-1] = 0
                    assert torch.equal(result[:,:-1],value[:,:-1])
                    return result
                self.stack.callback(layer.self_attn.o_proj.register_forward_hook(output).remove)
        return self

    def __exit__(self,*args):
        self.stack.close()


class PrefixTrace:
    def __init__(self,layers,n):
        self.layers,self.n,self.rows = layers,n,{}

    def __enter__(self):
        self.stack = ExitStack()
        for i,layer in enumerate(self.layers):
            def observe(module,args,out,i=i):
                value = out[0] if isinstance(out,tuple) else out
                assert value.shape[1] == self.n and torch.isfinite(value).all()
                self.rows[i] = tensor_hash(value[:,:-1])
            self.stack.callback(layer.register_forward_hook(observe).remove)
        return self

    def __exit__(self,*args):
        self.stack.close()


def metrics(full,changed):
    full,changed = full.float(),changed.float()
    assert torch.isfinite(full).all() and torch.isfinite(changed).all()
    a,b = full.log_softmax(-1),changed.log_softmax(-1)
    token,other = int(full.argmax()),int(changed.argmax())
    loss = float(a[token]-b[token])
    middle = torch.logaddexp(a,b)-__import__('math').log(2.)
    js = float(.5*((a.exp()*(a-middle)).sum()+(b.exp()*(b-middle)).sum()))
    return dict(full_first_token=token,changed_first_token=other,full_top1_logp=float(a[token]),
        off_full_top1_logp=float(b[token]),signed_logp_loss=loss,absolute_logp_change=abs(loss),
        top1_flip=int(token!=other),js_divergence=js)


class InternPrefill:
    def __init__(self,model,tokenizer,fast,sample,loader):
        from ..internvl3_smoke.engine import partitions
        from .common import sha
        assert sha(sample['image_path']) == sample['image_sha256']
        self.model = model
        self.request = '<image>\n'+sample['question']+'\nAnswer using a single word or short phrase.'
        chat = model.conv_template.copy()
        chat.system_message = model.system_message
        chat.append_message(chat.roles[0],self.request)
        chat.append_message(chat.roles[1],None)
        prompt = chat.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1)
        ids = tokenizer(prompt,return_tensors='pt')['input_ids'].to('cuda:0')
        image_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.img_context_token_id = image_id
        self.layout = partitions(tokenizer,fast,prompt,sample['question'],ids[0].tolist(),image_id)
        self.layout.update(prompt=prompt,input_ids=ids[0].tolist())
        assert self.layout['image_tokens'] == 256
        self.pixels = loader(sample['image_path'],max_num=1).to('cuda:0',dtype=torch.bfloat16)
        assert self.pixels.shape[0] == 1
        visual = model.extract_feature(self.pixels)
        embed = model.language_model.get_input_embeddings()(ids).clone()
        embed[ids == image_id] = visual.reshape(-1,embed.shape[-1])
        self.inputs = dict(inputs_embeds=embed,attention_mask=torch.ones_like(ids))
        self.fingerprint = tensor_hash(embed)
        self.layers = model.language_model.model.layers
        self.hooks = hooks_count(model)

    @torch.inference_mode()
    def forward(self,layer=None,noop=False,trace=False):
        n = self.layout['input_length']
        with ExitStack() as stack:
            edit = stack.enter_context(SourceMask(self.layers[layer].self_attn,[n-1],self.layout['positions']['image'],noop)) if layer is not None else None
            states = stack.enter_context(PrefixTrace(self.layers,n)) if trace else None
            out = self.model.language_model(**self.inputs,use_cache=False,past_key_values=None,output_attentions=False)
            assert out.past_key_values is None and torch.isfinite(out.logits).all()
            logits = out.logits[0,-1].detach().cpu().clone()
        assert hooks_count(self.model) == self.hooks
        if edit:
            assert edit.calls == 1 and edit.audit['changed_entries'] == (0 if noop else 256)
        return logits,edit.audit if edit else {},states

    def verify(self):
        assert tensor_hash(self.inputs['inputs_embeds']) == self.fingerprint
        assert hooks_count(self.model) == self.hooks
