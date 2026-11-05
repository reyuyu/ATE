"""Untimed quality adapter around unchanged historical physical execution classes.

Candidate answers are always consumed as incremental Decode, never appended to
Prefill. Every candidate forks a real independent prompt cache. Fork equivalence
to four cold requests is a strict Smoke gate, including all valid logits.
"""
from contextlib import contextmanager, ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import copy
import torch
from PIL import Image
from .common import MODELS, tensor_sha, hook_counts


def load(kind):
    from transformers import AutoProcessor, AutoConfig, LlavaForConditionalGeneration
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    processor=AutoProcessor.from_pretrained(MODELS[kind],local_files_only=True)
    if kind=='llava':
        config=AutoConfig.from_pretrained(MODELS[kind],local_files_only=True)
        processor.patch_size=config.vision_config.patch_size
        processor.num_additional_image_tokens=1
        processor.vision_feature_select_strategy=config.vision_feature_select_strategy
        processor.tokenizer.padding_side='left'
        model=LlavaForConditionalGeneration.from_pretrained(MODELS[kind],local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='eager',device_map={'':'cuda:0'},
            use_safetensors=True,low_cpu_mem_usage=True).eval()
        layers=model.language_model.model.layers
    else:
        from transformers import Qwen3VLForConditionalGeneration
        from .common import read,OUT
        processor.image_processor.size={**dict(processor.image_processor.size),'longest_edge':2048*32*32}
        assert processor.image_processor.to_dict()==read(OUT/'qwen_processor_2048.json')['capped']
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODELS[kind],local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},
            use_safetensors=True,low_cpu_mem_usage=True).eval()
        layers=model.model.language_model.layers
    return SimpleNamespace(model=model,processor=processor,layers=layers,kind=kind)


def prepare(adapter,sample):
    p=adapter.processor;kind=adapter.kind
    with Image.open(sample['image_path']) as im:image=im.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':sample['question']}]}]
    prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=p(images=image,text=prompt,return_tensors='pt')
    ids=inputs['input_ids'][0].tolist()
    image_id=adapter.model.config.image_token_index if kind=='llava' else adapter.model.config.image_token_id
    visual=[i for i,v in enumerate(ids) if v==image_id]
    assert visual and visual==list(range(visual[0],visual[-1]+1))
    if kind=='llava':assert len(visual)==576
    text=[i for i in range(len(ids)) if i not in set(visual)]
    relay=list(range(visual[-1]+1,len(ids)-1))
    assert relay and set(relay).issubset(text)
    binding=dict(prompt=prompt,input_ids=ids,image_positions=visual,input_length=len(ids),
        image_tokens=len(visual),p_minus_1=len(ids)-1,R=relay,
        input_hashes={k:tensor_sha(v) for k,v in inputs.items()})
    if 'image_grid_thw' in inputs:binding['image_grid_thw']=inputs['image_grid_thw'].tolist()
    if kind=='qwen':assert len(visual)<=2048
    adapter.inputs=inputs.to('cuda:0',dtype=next(adapter.model.parameters()).dtype)
    if kind=='llava':
        from ..physical_handoff_fastpath.engine import Layout
        layout=Layout(len(ids),tuple(visual),tuple(text),tuple(relay),tuple(relay),20)
    else:layout=binding
    return layout,binding


@contextmanager
def identity_attention(adapter):
    with ExitStack() as st:
        for layer in adapter.layers:
            original=layer.self_attn.forward
            def noop(*args,original=original,**kw):return original(*args,**kw)
            st.enter_context(patch.object(layer.self_attn,'forward',noop))
        yield


def control(adapter,layout,condition,capacity,audit=False):
    if condition=='Native':return None
    if adapter.kind=='llava':
        if condition=='Dense-F':
            from ..c3v2_middle_reaccess.engine import MiddleWriter
            return MiddleWriter(adapter,layout,(),capacity,audit,dense=True)
        from ..accuracy_autoresearch.engine import ResearchWriter
        return ResearchWriter(adapter,layout,{'answer_attention':'both'},list(layout.relay),capacity,audit)
    from ..qwen3vl_c3v2_physical.engine import Execution
    return Execution(adapter.model,layout,'F' if condition=='Dense-F' else 'C',capacity,audit)


def call(adapter,layout,cache,t,token=None):
    n=layout.length if adapter.kind=='llava' else layout['input_length']
    device=adapter.inputs['input_ids'].device
    kw=dict(adapter.inputs) if t==0 else dict(input_ids=torch.tensor([[token]],device=device),
        attention_mask=torch.ones((1,n+t),dtype=torch.long,device=device))
    kw.update(past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1,
        cache_position=torch.arange(n,device=device) if t==0 else torch.tensor([n+t-1],device=device))
    out=adapter.model(**kw)
    assert out.logits.shape[:2]==(1,1) and bool(torch.isfinite(out.logits).all())
    return out.logits[0,-1],out.past_key_values


def get_audit(c):
    if c is None:return None
    return dict(kv=c.cache.audit(),events=getattr(c,'events',[]),
        layers=getattr(c,'layers',[]),deepstack=getattr(c,'deepstack',[]),
        probability=getattr(c,'probability',[]))


@torch.inference_mode()
def greedy(adapter,layout,condition,capacity=32,capture=False,audit=False,noop=False):
    before=hook_counts(adapter.model);c=control(adapter,layout,condition,capacity,audit)
    ids=[];vectors=[];cache=None
    eos=adapter.model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    with (c.context() if c else nullcontext()),(identity_attention(adapter) if noop else nullcontext()):
        for t in range(capacity):
            v,cache=call(adapter,layout,cache,t,ids[-1] if t else None)
            if capture:vectors.append(v.detach().cpu().clone())
            tid=int(v.argmax());ids.append(tid)
            if tid in eos:break
    assert hook_counts(adapter.model)==before
    text=adapter.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
    result=dict(token_ids=ids,text=text,eos=ids[-1] in eos,empty=not bool(text),
        truncated=len(ids)==capacity and ids[-1] not in eos,first_token=ids[0],
        repeated_3gram=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2)))
    if audit:result['audit']=get_audit(c)
    del cache,c
    return result,vectors


def candidate_tokens(adapter,binding,choices):
    """Mean conditional NLL of choice content with explicit native end marker.

    LLaVA's official answer-ranking adapter includes its </s> turn terminator.
    Qwen's official adapter scores content only. Modern Qwen3 chat framing is
    retained and its candidate content is tokenized without BOS or end marker.
    Full concatenation must retain the exact prompt prefix, otherwise STOP.
    """
    tok=adapter.processor.tokenizer;prompt=binding['prompt'];out=[]
    for choice in choices:
        continuation=(' '+choice+tok.eos_token) if adapter.kind=='llava' else choice
        # Image expansion is already frozen by the Processor. Text suffix must
        # match both raw chat concatenation and the expanded input prefix.
        if adapter.kind=='llava':
            expanded=prompt.replace(adapter.processor.image_token,
                adapter.processor.image_token*binding['image_tokens'])
        else:
            expanded=prompt.replace(adapter.processor.image_token,
                adapter.processor.image_token*binding['image_tokens'])
        full=tok(expanded+continuation,add_special_tokens=True)['input_ids']
        prefix=binding['input_ids']
        assert full[:len(prefix)]==prefix,'Candidate boundary changes frozen prompt tokenization'
        ids=full[len(prefix):]
        assert ids and (adapter.kind!='llava' or ids[-1]==tok.eos_token_id)
        out.append(ids)
    return out


def clone_cache(cache):
    # Cache objects contain tensors and scalar counters, never the model.
    # Independent storages forbid candidate cross-contamination.
    return copy.deepcopy(cache)


@torch.inference_mode()
def rank_choices(adapter,layout,condition,candidates,capture=False,audit=False,fork=True):
    before=hook_counts(adapter.model);capacity=max(map(len,candidates))+1
    c=control(adapter,layout,condition,capacity,audit)
    values=[];scores=[];valid=[]
    if not fork:
        for ids in candidates:
            cc=control(adapter,layout,condition,capacity,audit);cache=None;logps=[];vectors=[]
            with cc.context() if cc else nullcontext():
                for t,target in enumerate(ids):
                    v,cache=call(adapter,layout,cache,t,ids[t-1] if t else None)
                    logps.append(float(v.float().log_softmax(-1)[target]))
                    if capture:vectors.append(v.detach().cpu().clone())
            values.append(logps);scores.append(float(-torch.tensor(logps,dtype=torch.float32).mean()));valid.append(vectors)
            del cache,cc
    else:
        with c.context() if c else nullcontext():
            first,prompt_cache=call(adapter,layout,None,0)
            # Physical prefill buffers are no longer modified after cloning.
            # DynamicCache clone also duplicates its complete tensor storage.
            for ids in candidates:
                cache=clone_cache(prompt_cache)
                if c:
                    c.cache=cache;c.step=1
                    if hasattr(cache,'prefill'):cache.prefill=False
                logps=[float(first.float().log_softmax(-1)[ids[0]])]
                vectors=[first.detach().cpu().clone()] if capture else []
                for t,target in enumerate(ids[1:],1):
                    v,cache=call(adapter,layout,cache,t,ids[t-1])
                    logps.append(float(v.float().log_softmax(-1)[target]))
                    if capture:vectors.append(v.detach().cpu().clone())
                values.append(logps);scores.append(float(-torch.tensor(logps,dtype=torch.float32).mean()));valid.append(vectors)
                del cache
            del prompt_cache
    assert hook_counts(adapter.model)==before
    result=dict(choice_token_ids=candidates,token_logps=values,mean_nll=scores,
        prediction='ABCD'[min(range(4),key=lambda i:(scores[i],i))],
        scoring='official answer-ranking; mean conditional NLL; lower is better',
        candidate_cache_independent=True)
    if audit:result['audit']=get_audit(c)
    del c
    return result,valid
