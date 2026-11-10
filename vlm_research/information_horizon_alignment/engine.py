"""Author native Flash2 and separate, uniformly eager HF physical conditions."""
from contextlib import nullcontext, ExitStack
from dataclasses import replace
import gc
import os
import sys
from types import SimpleNamespace, MethodType
from unittest.mock import patch

import torch
from PIL import Image

from .common import *

PRUNING_ENV = ('SAVE_NUM_LIST', 'PRUNED_LAYER_LIST', 'METHOD_PRE', 'METHOD_LLM',
               'PRUNE_STG', 'DATASET', 'SIZE', 'X', 'Y', 'LAYER_INDEX')


def image_ids(text, tokenizer, marker):
    """Same image placeholder boundary treatment as author's tokenizer_image_token."""
    chunks = [tokenizer(v).input_ids for v in text.split('<image>')]
    offset = int(bool(chunks[0]) and chunks[0][0] == tokenizer.bos_token_id)
    result = [tokenizer.bos_token_id] if offset else []
    for i, chunk in enumerate(chunks):
        if i:
            result.append(marker)
        result.extend(chunk[offset:])
    return result


def paper_pixels(path, processor):
    with Image.open(path) as original:
        image = original.convert('RGB')
    w, h = image.size
    if w != h:
        bg = tuple(int(v * 255) for v in processor.image_mean)
        padded = Image.new(image.mode, (max(w, h), max(w, h)), bg)
        padded.paste(image, ((max(w,h)-w)//2, (max(w,h)-h)//2))
        image = padded
    return processor.preprocess(image, return_tensors='pt')['pixel_values']


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load(stage):
    setup()
    if stage == 'B':
        from ..mme_seed_quality.engine import load as historical_load
        a = historical_load('llava')
        a.stage = stage
        a.tokenizer = a.processor.tokenizer
        a.image_processor = a.processor.image_processor
        return a
    for key in PRUNING_ENV:
        assert key not in os.environ, f'Pruning environment must be absent: {key}'
    sys.path.insert(0, str(AUTHOR))
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.modeling_llama_self import LlamaModel
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer, model, processor, context = load_pretrained_model(
        str(NATIVE_MODEL), None, 'llava-v1.5-7b', device_map='auto',
        attn_implementation='flash_attention_2')
    model.eval()
    assert all(p.device.type=='cuda' for p in model.parameters()), 'No CPU offload in native baseline'
    model.config.pruned_config = None
    # The author's PrunedLLM.forward uses unbound K when pruned_config=None.
    # Select the author's existing non-pruning parent forward, without editing
    # its checkout, attention kernels, weights, generation or preprocessing.
    model.model.forward = MethodType(LlamaModel.forward, model.model)
    assert len(model.model.layers) == 32 and model.model._use_flash_attention_2
    assert all(type(layer.self_attn).__name__ == 'LlamaFlashAttention2' for layer in model.model.layers)
    return SimpleNamespace(stage=stage, model=model, tokenizer=tokenizer,
        image_processor=processor, layers=model.model.layers, context_length=context)


def prepare(a, sample):
    text = sample['prompt']
    raw_ids = image_ids(text, a.tokenizer, -200)
    assert raw_ids.count(-200) == 1 and raw_ids[0] == a.tokenizer.bos_token_id
    pixels = paper_pixels(sample['image_path'], a.image_processor)
    assert pixels.shape == (1, 3, 336, 336)
    binding = dict(prompt=text, raw_image_placeholder_ids=raw_ids,
                   pixel_values_sha256=tensor_sha(pixels), pixel_shape=list(pixels.shape))
    if a.stage == 'A':
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.conversation import conv_templates
        ref = tokenizer_image_token(text, a.tokenizer)
        assert ref == raw_ids
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], '<image>\n' + sample['user_text'])
        conv.append_message(conv.roles[1], None)
        assert conv.get_prompt() == text
        with Image.open(sample['image_path']) as image:
            reference = process_images([image.convert('RGB')], a.image_processor, a.model.config)
            image_size = image.size
        assert torch.equal(reference, pixels)
        a.inputs = dict(input_ids=torch.tensor([raw_ids], device='cuda:0'),
                        images=pixels.to('cuda:0', dtype=torch.float16), image_sizes=[image_size])
        binding.update(expanded_length=len(raw_ids)+575,
            image_positions=list(range(raw_ids.index(-200),raw_ids.index(-200)+576)))
        return None, binding
    marker = a.model.config.image_token_index
    ids = []
    for token in raw_ids:
        ids.extend([marker]*576 if token == -200 else [token])
    assert ids.count(marker) == 576
    image = tuple(i for i,t in enumerate(ids) if t == marker)
    assert image == tuple(range(image[0],image[0]+576))
    text_positions = tuple(i for i,t in enumerate(ids) if t != marker)
    relay = tuple(range(image[-1]+1,len(ids)-1))
    from ..physical_handoff_fastpath.engine import Layout
    layout = Layout(len(ids),image,text_positions,relay,relay,20)
    a.inputs = dict(input_ids=torch.tensor([ids],device='cuda:0'),
        attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device='cuda:0'),
        pixel_values=pixels.to('cuda:0',dtype=torch.float16))
    binding.update(input_ids=ids,expanded_length=len(ids),image_positions=list(image),
                   Pminus1=len(ids)-1,R=list(relay))
    return layout,binding


def result(a, ids, cap):
    eos = a.model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos)
    text = a.tokenizer.decode(ids,skip_special_tokens=True).strip()
    grams = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
    return dict(text=text,token_ids=ids,eos=ids[-1] in eos,empty=not bool(text),
        truncated=len(ids)==cap and ids[-1] not in eos,first_token_id=ids[0],
        repeated_3gram=len(grams)!=len(set(grams)),all_effective_logits_finite=True,
        independent_cache=True)


@torch.inference_mode()
def author_generate(a, sample, capture=False, noop=False, trace=False, inactive_prune=False):
    cap = sample['max_new_tokens']
    before = hook_counts(a.model)
    logits, first, observed, handles = [], [], [], []
    calls = [0]
    original = a.model.model.forward
    if inactive_prune:
        from llava.model.language_model.modeling_llama_self import PrunedLLM
        a.model.config.pruned_config = dict(K=-1,image_token_start_index=0,image_token_length=576,
            max_num_trunction=576)
        a.model.model.forward = MethodType(PrunedLLM.forward,a.model.model)
    def head(m,args,out):
        calls[0] += 1
        v = out[:, -1].float()
        assert torch.isfinite(v).all()
        if not first:
            first.append(tensor_sha(v))
        if capture:
            logits.append(v.cpu().clone())
    handles.append(a.model.lm_head.register_forward_hook(head))
    if trace:
        for i,layer in enumerate(a.layers):
            def seen(m,args,i=i):
                observed.append(dict(layer=i,rows=args[0].shape[1]))
            handles.append(layer.register_forward_pre_hook(seen))
    try:
        with ExitStack() as stack:
            if noop:
                for layer in a.layers:
                    method = layer.self_attn.forward
                    def identity(*args,method=method,**kw):
                        return method(*args,**kw)
                    stack.enter_context(patch.object(layer.self_attn,'forward',identity))
            generated = a.model.generate(a.inputs['input_ids'],images=a.inputs['images'],
                image_sizes=a.inputs['image_sizes'],do_sample=False,temperature=0,
                num_beams=1,top_p=None,max_new_tokens=cap,use_cache=True,
                return_dict_in_generate=True,output_scores=False)
        sequence = generated.sequences[0].tolist()
        # Inputs-embeds generation may return a synthetic BOS (not an answer).
        nsteps = calls[0]
        if sequence and sequence[0] == a.tokenizer.bos_token_id and len(sequence)==nsteps+1:
            sequence = sequence[1:]
        assert len(sequence) == nsteps
        record = result(a,sequence,cap)
        record.update(first_logits_sha256=first[0],native_unpruned_parent_forward=True)
        if trace:
            assert len(observed) == 32*len(sequence)
            length = len(a.inputs['input_ids'][0])+575
            for j,row in enumerate(observed):
                assert row == dict(layer=j%32,rows=length if j<32 else 1)
            record['native_layer_shapes'] = observed
        return record,logits
    finally:
        for handle in handles:
            handle.remove()
        a.model.model.forward = original
        a.model.config.pruned_config = None
        assert hook_counts(a.model) == before


@torch.inference_mode()
def hf_generate(a, layout, condition, cap, capture=False, noop=False, audit=False,
                dense=False, fixed_prefix=None):
    from ..c3v2_middle_reaccess.engine import MiddleWriter
    from ..physical_handoff_fastpath.engine import FastHandoff
    ctl = None
    if condition != 'HF-Native':
        keep = () if condition == 'Physical-C3' else tuple(range(11,25))
        layout = replace(layout,cutoff=20 if not keep else 25)
        ctl = MiddleWriter(a,layout,keep,cap,audit,dense)
    elif noop:
        ctl = FastHandoff(a,layout,'Native-Noop',cap,audit)
    before = hook_counts(a.model)
    n = layout.length
    positions = torch.arange(n+cap-1,device='cuda:0')
    mask = torch.ones((1,n+cap-1),dtype=torch.long,device='cuda:0')
    cache,token = None,None
    ids,logits = [],[]
    try:
        with ctl.context() if ctl else nullcontext():
            for step in range(cap):
                kw = dict(a.inputs) if step == 0 else dict(input_ids=token.reshape(1,1),
                    attention_mask=mask[:,:n+step])
                kw.update(use_cache=True,return_dict=True,past_key_values=cache,
                    cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],
                    logits_to_keep=1)
                output = a.model(**kw)
                assert output.logits.shape[:2] == (1,1)
                v = output.logits[:,-1]
                assert torch.isfinite(v).all()
                if not step:
                    first = tensor_sha(v)
                if capture:
                    logits.append(v.detach().float().cpu().clone())
                token = v.argmax(-1)
                if fixed_prefix is not None:
                    token = torch.tensor([fixed_prefix[step]],device='cuda:0')
                ids.append(int(token))
                cache = output.past_key_values
                del output
                eos = a.model.generation_config.eos_token_id
                if fixed_prefix is None and ids[-1] in ([eos] if isinstance(eos,int) else eos):
                    break
        record = result(a,ids,cap)
        record.update(first_logits_sha256=first,logits_to_keep=1)
        if audit and ctl:
            record.update(attention_audit=ctl.audit_rows,visual_audit=getattr(ctl,'visual_events',[]),
                          kv_audit=ctl.cache.audit(),_hidden=ctl.hidden,_rotary=ctl.rotary)
        return record,logits
    finally:
        del ctl,cache
        assert hook_counts(a.model) == before
