"""InternVL-specific native eager mask hooks; no replacement forward or cache."""
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import torch

TASKS = ["ChooseAttr","ChooseCat","ChooseRel","CompareAttr","LogicalObj","QueryAttr"]
CONDITIONS = ["All-Writer","No-Writer"]+[f"Drop-L{i}" for i in range(28)]+["Keep-L0-L16"]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def hooks_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())

def writers(condition):
    if condition=="All-Writer": return list(range(28))
    if condition=="No-Writer": return []
    if condition=="Keep-L0-L16": return list(range(17))
    assert condition.startswith("Drop-L")
    layer=int(condition[6:]);assert 0<=layer<28
    return [i for i in range(28) if i!=layer]

def official_preprocessor(directory):
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    text=(directory/"README.md").read_text(encoding="utf-8")
    text=text[text.index("IMAGENET_MEAN ="):];text=text[:text.index("def split_model(model_name):")]
    tree=ast.parse(text);names={"build_transform","find_closest_aspect_ratio","dynamic_preprocess","load_image"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    scope=dict(Image=Image,T=T,InterpolationMode=InterpolationMode,torch=torch,
               IMAGENET_MEAN=(.485,.456,.406),IMAGENET_STD=(.229,.224,.225))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(directory/"README.md"),"exec"),scope)
    return scope["load_image"]

def role_layout(ids,image_id):
    image=[i for i,x in enumerate(ids) if x==image_id]
    assert len(image)==256 and image==list(range(image[0],image[-1]+1))
    n=len(ids);r=list(range(image[-1]+1,n-1))
    assert r and max(image)<n-1
    return dict(n=n,image=image,R=r,A_prefill=[n-1],image_token_id=image_id)

def prepare_cpu(model,tokenizer,sample):
    request="<image>\n"+sample["question"]+"\nAnswer using a single word or short phrase."
    conv=model.conv_template.copy();conv.system_message=model.system_message
    conv.append_message(conv.roles[0],request);conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
    ids=tokenizer(prompt,return_tensors="pt")["input_ids"]
    roles=role_layout(ids[0].tolist(),tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
    return request,prompt,ids,roles

@torch.inference_mode()
def prepare(model,tokenizer,loader,sample):
    request,prompt,ids,roles=prepare_cpu(model,tokenizer,sample)
    pixels=loader(sample["actual_image_path"],max_num=1).to("cuda:0",dtype=torch.bfloat16)
    assert pixels.shape==(1,3,448,448)
    ids=ids.to("cuda:0");image_id=roles["image_token_id"];model.img_context_token_id=image_id
    visual=model.extract_feature(pixels)
    assert visual.shape[:2]==(1,256) and torch.isfinite(visual).all()
    embed=model.language_model.get_input_embeddings()(ids).clone()
    embed[ids==image_id]=visual.reshape(-1,embed.shape[-1])
    inputs=dict(inputs_embeds=embed,attention_mask=torch.ones_like(ids))
    binding=dict(prompt=prompt,input_ids=ids[0].tolist(),roles=roles,
        pixels_sha256=tensor_hash(pixels),visual_sha256=tensor_hash(visual),embeddings_sha256=tensor_hash(embed))
    return dict(inputs=inputs,roles=roles,pixels=pixels,request=request,prompt=prompt,binding=binding,
                input_binding=digest(binding))

def edit_mask(original,roles,prefill,r_off):
    assert original is not None and original.ndim==4 and original.shape[:2]==(1,1)
    image=roles["image"];lo,hi=image[0],image[-1]+1
    result=original.clone();q=roles["n"]-1 if prefill else 0
    result[:,:,q,lo:hi]=-torch.inf
    if prefill and r_off: result[:,:,hi:q,lo:hi]=-torch.inf
    return result

class WriterMask:
    def __init__(self,model,roles,condition,cache,noop=False,audit=False,trace=False):
        self.model=model;self.lm=model.language_model;self.layers=self.lm.model.layers
        self.roles=roles;self.condition=condition;self.allowed=set(writers(condition))
        self.cache=cache;self.noop=noop;self.audit=audit;self.trace=trace
        self.steps=[];self.events=[];self.prefix={};self.mask_checks=[]
        self.max_probability=torch.zeros((),device="cuda:0")
        self.finite=torch.ones((),device="cuda:0",dtype=torch.bool)
        self.row_error=torch.zeros((),device="cuda:0")
        self.open_R_max=torch.zeros((),device="cuda:0")
        self.states={}
    def __enter__(self):
        self.stack=ExitStack()
        def begin(module,args,kwargs):
            step=len(self.steps);n=self.roles["n"];start=0 if step==0 else n+step-1;qlen=n if step==0 else 1
            value=kwargs.get("inputs_embeds")
            if value is None:value=kwargs["input_ids"]
            assert value.shape[1]==qlen and kwargs["past_key_values"] is self.cache
            assert kwargs.get("logits_to_keep")==1
            assert kwargs["cache_position"].tolist()==list(range(start,start+qlen))
            assert kwargs["position_ids"].tolist()==[list(range(start,start+qlen))]
            assert all(self.cache.get_seq_length(i)==start for i in range(28))
            self.steps.append(dict(step=step,qlen=qlen,start=start))
            self.native_mask=None;self.masks={}
        self.stack.callback(self.lm.register_forward_pre_hook(begin,with_kwargs=True).remove)
        for index,layer in enumerate(self.layers):
            def before(module,args,kwargs,i=index):
                step=len(self.steps)-1;prefill=step==0;qlen=self.roles["n"] if prefill else 1
                assert kwargs["hidden_states"].shape[1]==qlen
                assert kwargs["past_key_value"] is self.cache
                native=kwargs["attention_mask"]
                assert native.shape[-2]==qlen and native.shape[-1]>=self.roles["n"]
                assert kwargs["cache_position"].tolist()==list(range(self.steps[-1]["start"],self.steps[-1]["start"]+qlen))
                if self.native_mask is None:self.native_mask=native
                else:assert native is self.native_mask
                off=prefill and i not in self.allowed
                if self.noop:changed=native
                else:
                    if off not in self.masks:self.masks[off]=edit_mask(native,self.roles,prefill,off)
                    changed=self.masks[off]
                if self.audit:
                    expected=torch.zeros_like(native,dtype=torch.bool)
                    if not self.noop:
                        lo,hi=self.roles["image"][0],self.roles["image"][-1]+1
                        q=self.roles["n"]-1 if prefill else 0
                        expected[:,:,q,lo:hi]=True
                        if off:expected[:,:,hi:q,lo:hi]=True
                    assert torch.equal(native!=changed,expected)
                    assert (changed==0).any(-1).all()
                    assert torch.equal(kwargs["position_embeddings"][0],self.position_embeddings[0]) if i else True
                    assert torch.equal(kwargs["position_embeddings"][1],self.position_embeddings[1]) if i else True
                    if i==0:self.position_embeddings=kwargs["position_embeddings"]
                    self.mask_checks.append(dict(step=step,layer=i,changed=int(expected.sum()),R_off=off,non_target_exact=True))
                self.events.append((step,i))
                return args,dict(kwargs,attention_mask=changed)
            def after(module,args,out,i=index):
                weights=out[1];assert weights is not None
                prefill=len(self.steps)==1;lo,hi=self.roles["image"][0],self.roles["image"][-1]+1
                q=self.roles["n"]-1 if prefill else 0
                row=weights[:,:,q:q+1,:]
                self.finite=self.finite & torch.isfinite(row).all() & (row>=0).all()
                self.row_error=torch.maximum(self.row_error,(row.float().sum(-1)-1).abs().max())
                if not self.noop:
                    self.max_probability=torch.maximum(self.max_probability,row[:,:,:,lo:hi].abs().max())
                    if prefill and i not in self.allowed:
                        blocked=weights[:,:,hi:q,:]
                        self.finite=self.finite & torch.isfinite(blocked).all() & (blocked>=0).all()
                        self.max_probability=torch.maximum(self.max_probability,blocked[:,:,:,lo:hi].abs().max())
                        self.row_error=torch.maximum(self.row_error,(blocked.float().sum(-1)-1).abs().max())
                    elif prefill:
                        self.open_R_max=torch.maximum(self.open_R_max,weights[:,:,hi:q,lo:hi].max())
                return out
            self.stack.callback(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True).remove)
            self.stack.callback(layer.self_attn.register_forward_hook(after).remove)
            if self.trace:
                def output(module,args,value,i=index):
                    if len(self.steps)!=1:return
                    x=value[0] if isinstance(value,tuple) else value
                    self.prefix[i]=tensor_hash(x[:,:self.roles["image"][-1]+1])
                    assert torch.isfinite(x).all()
                    record=self.states[i]
                    assert torch.equal(record["input"]+record["attn"],record["mid"])
                    assert torch.equal(record["mid"]+record["mlp"],x)
                    self.states[i]={}
                self.stack.callback(layer.register_forward_hook(output).remove)
                def state(name,pre=False,i=index):
                    def record(module,args,out=None):
                        if len(self.steps)!=1:return
                        value=args[0] if pre else out
                        self.states.setdefault(i,{})[name]=value.detach().clone()
                    return record
                self.stack.callback(layer.register_forward_pre_hook(state("input",True)).remove)
                self.stack.callback(layer.self_attn.o_proj.register_forward_hook(state("attn")).remove)
                self.stack.callback(layer.post_attention_layernorm.register_forward_pre_hook(state("mid",True)).remove)
                self.stack.callback(layer.mlp.register_forward_hook(state("mlp")).remove)
        return self
    def __exit__(self,*args):
        self.stack.close()
    def validate(self,tokens):
        assert len(self.steps)==tokens and self.events==[(s,i) for s in range(tokens) for i in range(28)]
        assert bool(self.finite) and float(self.row_error)<=2**-8+2e-6
        assert self.noop or float(self.max_probability)==0
        return dict(steps=tokens,prefill=1,real_decode=tokens-1,layers=28,
            blocked_probability_max=float(self.max_probability),row_sum_max_error=float(self.row_error),
            open_R_image_probability_max=float(self.open_R_max),native_positions=True,non_target_mask_exact=self.audit)

@torch.inference_mode()
def generate(model,tokenizer,prepared,condition="All-Writer",native=False,noop=False,audit=False,trace=False):
    from transformers.cache_utils import DynamicCache
    cache=DynamicCache();before=hooks_count(model);lm=model.language_model
    fingerprints={k:tensor_hash(v) for k,v in prepared["inputs"].items()}
    with ExitStack() as stack:
        control=None if native else stack.enter_context(WriterMask(model,prepared["roles"],condition,cache,noop,audit,trace))
        eos=tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
        result=lm.generate(**prepared["inputs"],past_key_values=cache,use_cache=True,do_sample=False,num_beams=1,
            max_new_tokens=32,logits_to_keep=1,eos_token_id=eos,pad_token_id=eos,
            return_dict_in_generate=True,output_scores=True,output_attentions=False,output_hidden_states=False)
        ids=result.sequences[0].tolist();assert ids and len(ids)==len(result.scores) and result.past_key_values is cache
        scores=[s[0].detach().cpu().clone() for s in result.scores]
        assert all(torch.isfinite(v).all() for v in scores)
        n=prepared["roles"]["n"];lengths=[cache.get_seq_length(i) for i in range(28)]
        assert lengths==[n+len(ids)-1]*28
        cache_finite=torch.ones((),device="cuda:0",dtype=torch.bool)
        for k,v in zip(cache.key_cache,cache.value_cache):
            assert k.shape[-2]==v.shape[-2]==lengths[0] and k.dtype==v.dtype==torch.bfloat16
            cache_finite=cache_finite & torch.isfinite(k).all() & torch.isfinite(v).all()
        assert bool(cache_finite)
        text=tokenizer.decode(ids,skip_special_tokens=True).split(model.conv_template.sep.strip())[0].strip()
        audit_row=control.validate(len(ids)) if control else dict(native=True)
        value=dict(text=text,prediction=text.rstrip(".").lower(),token_ids=ids,
            generated_tokens=len(ids),eos=ids[-1]==eos,truncated=len(ids)==32 and ids[-1]!=eos,empty=not text,
            repeated_trigram=len(ids)>=6 and len({tuple(ids[j:j+3]) for j in range(len(ids)-2)})<len(ids)-2,
            all_step_logits_sha256=[tensor_hash(v) for v in scores],final_cache_lengths=lengths,
            cache_shapes=[list(x.shape) for x in cache.key_cache],independent_fresh_cache=True,
            writer_layers=writers(condition),P1_Decode_image_off=not(native or noop),logits_to_keep=1,
            mask_audit=audit_row,input_binding=prepared["input_binding"])
        traces=dict(prefix=control.prefix,mask_checks=control.mask_checks) if control else {}
    assert hooks_count(model)==before
    assert fingerprints=={k:tensor_hash(v) for k,v in prepared["inputs"].items()}
    return value,scores,traces

class OfficialChatCapture:
    def __init__(self,model):self.model=model;self.scores=[];self.ids=None
    def __enter__(self):
        from unittest.mock import patch
        self.stack=ExitStack();original=self.model.generate
        def generate(*args,**kwargs):
            self.input_ids=kwargs["input_ids"].detach().cpu().clone()
            out=original(*args,**kwargs);self.ids=out[0].detach().cpu().tolist();return out
        self.stack.enter_context(patch.object(self.model,"generate",generate))
        def before(module,args,kwargs):
            if not self.scores:self.embeddings=kwargs["inputs_embeds"].detach().cpu().clone()
            assert kwargs.get("logits_to_keep")==1
        def after(module,args,out):self.scores.append(out.logits[0,-1].float().detach().cpu().clone())
        self.stack.callback(self.model.language_model.register_forward_pre_hook(before,with_kwargs=True).remove)
        self.stack.callback(self.model.language_model.register_forward_hook(after).remove)
        return self
    def __exit__(self,*args):self.stack.close()
