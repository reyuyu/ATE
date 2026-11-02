from contextlib import ExitStack

import torch

from .common import PROMPT, QMODEL, IMODEL, sha, digest
from ..visual_pathways.sparse_accuracy import GenerationMask
from ..internvl3_smoke.engine import tensor_hash, hooks_count, partitions


class DecodeMask(GenerationMask):
    """Reuse the audited persistent controller, but make all Prefill hooks no-ops."""
    def __enter__(self):
        self.requested_noop = self.noop
        self.rows, self.full_prefill = [], {}
        try:
            super().__enter__()
            def phase(module, args, kwargs):
                self.noop = self.requested_noop or len(self.calls) == 1
            self.stack.callback(self.lm.register_forward_pre_hook(phase, with_kwargs=True).remove)
            for i, layer in enumerate(self.layers):
                def after(module, args, output, i=i):
                    step = len(self.calls)-1
                    row = output[1][0, :, -1, :]
                    assert row.shape[-1] == self.n+step and torch.isfinite(row).all()
                    mx = float(row[:, self.image].max())
                    error = float((row.float().sum(-1)-1).abs().max())
                    assert error <= 2**-8+2e-6
                    active = step > 0 and i in self.off and not self.requested_noop
                    assert mx == 0. if active else mx > 0.
                    self.rows.append(dict(step=step, layer=i, masked=active, image_max=mx,
                        image_mass=float(row[:,self.image].float().sum(-1).mean()), row_error=error))
                self.stack.callback(layer.self_attn.register_forward_hook(after).remove)
                if self.capture:
                    def prefill(module,args,output,i=i):
                        if len(self.calls) == 1:
                            v = output[0] if isinstance(output,tuple) else output
                            self.full_prefill[i] = tensor_hash(v)
                    self.stack.callback(layer.register_forward_hook(prefill).remove)
            return self
        except BaseException:
            self.stack.close()
            raise


def intern_layout(model, tokenizer, fast, sample):
    request = '<image>\n'+PROMPT
    chat = model.conv_template.copy()
    chat.system_message = model.system_message
    chat.append_message(chat.roles[0], request)
    chat.append_message(chat.roles[1], None)
    prompt = chat.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1)
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    image_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    layout = partitions(tokenizer, fast, prompt, PROMPT, ids[0].tolist(), image_id)
    layout.update(prompt=prompt, input_ids=ids[0].tolist(), request=request)
    assert layout['image_tokens'] == 256
    return ids, layout


def qwen_inputs(processor, sample):
    from PIL import Image
    assert sha(sample['image_path']) == sample['image_sha256']
    with Image.open(sample['image_path']) as f:
        image = f.convert('RGB')
    messages = [{'role':'user','content':[{'type':'image','image':sample['image_path']},
                {'type':'text','text':PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt')
    ids = inputs['input_ids'][0].tolist()
    positions = [i for i,t in enumerate(ids) if t == processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    grid = inputs['image_grid_thw'][0].tolist()
    assert len(positions) == __import__('math').prod(grid)//processor.image_processor.merge_size**2
    assert positions == list(range(positions[0], positions[-1]+1))
    return inputs, dict(prompt=prompt, input_ids=ids, input_length=len(ids),
        image_positions=positions,image_tokens=len(positions),image_grid_thw=grid,p_minus_1=len(ids)-1,
        tensor_hashes={k:tensor_hash(v) for k,v in inputs.items()})


@torch.inference_mode()
def prepare(kind, model, processor, sample, fast=None, loader=None):
    if kind == 'qwen':
        inputs, layout = qwen_inputs(processor, sample)
        return inputs.to('cuda:0'), layout, None
    ids, layout = intern_layout(model, processor, fast, sample)
    ids = ids.to('cuda:0')
    model.img_context_token_id = processor.convert_tokens_to_ids('<IMG_CONTEXT>')
    pixels = loader(sample['image_path'], max_num=1).to('cuda:0',dtype=torch.bfloat16)
    assert pixels.shape[0] == 1
    features = model.extract_feature(pixels)
    embed = model.language_model.get_input_embeddings()(ids).clone()
    embed[ids == model.img_context_token_id] = features.reshape(-1, embed.shape[-1])
    return dict(inputs_embeds=embed,attention_mask=torch.ones_like(ids)), layout, pixels


@torch.inference_mode()
def generate(kind, model, tokenizer, inputs, layout, keep, *, noop=False, native=False, capture=False, limit=96):
    from transformers.cache_utils import DynamicCache
    lm = model.model.language_model if kind == 'qwen' else model.language_model
    layers = lm.layers if kind == 'qwen' else lm.model.layers
    n = layout['input_length']
    image = layout['image_positions'] if kind == 'qwen' else layout['positions']['image']
    off = sorted(set(range(len(layers)))-set(keep))
    cache = DynamicCache(config=model.config.text_config) if kind == 'qwen' else DynamicCache()
    footprint = hooks_count(model)
    hashes = {k:tensor_hash(v) for k,v in inputs.items()}
    with ExitStack() as stack:
        controller = None if native else stack.enter_context(DecodeMask(lm,layers,n,image,off,cache,noop,capture))
        options = dict(past_key_values=cache, use_cache=True, do_sample=False,num_beams=1,
            max_new_tokens=limit,logits_to_keep=1,return_dict_in_generate=True,output_scores=True,
            output_attentions=False,output_hidden_states=False)
        if kind == 'qwen':
            out = model.generate(**inputs, **options)
            ids = out.sequences[0,n:].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos,int) else eos
        else:
            end = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
            out = lm.generate(**inputs, **options, eos_token_id=end,pad_token_id=end)
            ids = out.sequences[0].tolist(); eos = [end]
        assert ids and len(ids) == len(out.scores) and out.past_key_values is cache
        assert all(torch.isfinite(v).all() for v in out.scores)
        assert all(cache.get_seq_length(i) == n+len(ids)-1 for i in range(len(layers)))
        first = out.scores[0][0].detach().cpu().clone()
        text = tokenizer.decode(ids,skip_special_tokens=True)
        if kind == 'internvl': text = text.split(model.conv_template.sep.strip())[0]
        rows = controller.rows if controller else []
        if controller:
            assert len(controller.calls) == len(ids)
            assert len(rows) == len(ids)*len(layers)
            assert sum(r['masked'] for r in rows) == (0 if noop else len(off)*(len(ids)-1))
            assert not any(r['masked'] for r in rows if r['step'] == 0)
        value = dict(text=text.strip(), token_ids=ids, eos=ids[-1] in eos,
            truncated=len(ids)==limit and ids[-1] not in eos, empty=not bool(text.strip()),
            decode_steps=len(ids)-1,fresh_cache=True,cache_length=n+len(ids)-1,
            logits_sha256=[tensor_hash(v) for v in out.scores],prefill_blocked=[],decode_blocked=off if not noop else [],
            audit=dict(rows=len(rows),decode_masked_rows=sum(r['masked'] for r in rows),
                row_sum_max_error=max((r['row_error'] for r in rows),default=0),
                blocked_image_probability_max=max((r['image_max'] for r in rows if r['masked']),default=0.),
                retained_image_probability_min=min((r['image_max'] for r in rows if not r['masked']),default=1.)))
        trace = dict(full_prefill=controller.full_prefill,deepstack=controller.deepstack,
                     rows=rows if capture else []) if controller else {}
    assert hooks_count(model) == footprint
    assert hashes == {k:tensor_hash(v) for k,v in inputs.items()}
    return value, first, trace
