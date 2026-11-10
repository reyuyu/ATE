"""Independent HF 4.49 bridges. Official functions are AST-loaded unchanged.

ApET: official 288 -> 96 after layer 16, FPS basis=10 (average 192).
DUET: official current C+all code, dominant=300 including CLS, contextual=7,
      307 -> 153 after layer 16 -> 0 after layer 24 (average 191.75).
This does not claim DUET's last-text-token C variant or trained weights.
"""
from pathlib import Path
from types import SimpleNamespace,MethodType
from contextlib import contextmanager
from typing import *
import ast,copy,hashlib,math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary
import einops as ein

ROOT=Path('/home/featurize/work/ATE/vlm_research/external_baselines')
KERNELS={}

def extract(path,name,namespace,owner=None):
    """Compile precisely the official AST, retaining its source line metadata."""
    tree=ast.parse(Path(path).read_text());nodes=tree.body
    if owner:nodes=next(n.body for n in nodes if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    module=ast.Module(body=[copy.deepcopy(node)],type_ignores=[])
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name],dict(path=str(path),function=(owner+'.' if owner else '')+name,start=node.lineno,end=node.end_lineno,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),AST_unmodified=True)

def sparse_pruning_block(path,namespace):
    """Wrap the official inference pruning AST verbatim, without rewriting it."""
    tree=ast.parse(Path(path).read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlamaDynamicvitModel')
    forward=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    branch=next(n for n in ast.walk(forward) if isinstance(n,ast.If) and 'layer_idx in self.pruning_loc' in ast.unparse(n.test))
    inference=next(n for n in branch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.training').orelse
    first=next(i for i,n in enumerate(inference) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('attn_logits = layer_outputs[2]'))
    body=copy.deepcopy(inference[first:])
    wrapper=ast.parse('''def official_sparse_prune(hidden_states, attn, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx, position_ids):
 B=1
 retained_tokens=192
 pre_prompt_length_list=[v_token_start]
 image_shape=576
 layer_outputs=(hidden_states,None,attn)
''').body[0]
    wrapper.body+=body+ast.parse('return layer_outputs[0],v_token_num,position_ids').body
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));exec(compile(module,str(path),'exec'),namespace)
    return namespace['official_sparse_prune'],dict(path=str(path),function='LlamaDynamicvitModel.forward inference prune block',start=body[0].lineno,end=body[-1].end_lineno,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),AST_unmodified=True)

class LegacyRotary:
    """Old seq_len API over the frozen HF rotary implementation, no new weights."""
    def __init__(self,rotary):self.rotary=rotary
    def __call__(self,value,seq_len):
        pos=torch.arange(seq_len,device=value.device).unsqueeze(0)
        cos,sin=self.rotary(value,pos)
        return cos.squeeze(0),sin.squeeze(0)

class OfficialCompat:
    def __init__(self,a,layout,method,trace=False):
        assert method in ('ApET','DUET','SparseVLM','BridgeNative')
        self.a=a;self.model=a.model;self.decoder=a.model.language_model.model
        self.layout=layout;self.method=method;self.trace=trace;self.step=0;self.events=[];self.shapes=[]
        self.provenance=[];self.handles=[];self.layer_visual=[None]*32;self.forward_calls=0
        self.namespace=dict(globals());self.namespace.update(IGNORE_INDEX=-100)
        if method in KERNELS:
            bundle=KERNELS[method]
            self.namespace=bundle['namespace'];self.provenance=bundle['provenance']
            for name,fn in bundle['functions'].items():setattr(self,name,fn)
        elif method=='ApET':
            base=ROOT/'apet/official_source/llava/model'
            for name in ('fps',):self.load(base/'utils.py',name)
            self.vision_merge=self.load(base/'llava_arch.py','token_merging','LlavaMetaForCausalLM')
            self.vision_encode=self.load(base/'llava_arch.py','encode_images','LlavaMetaForCausalLM')
            self.load(base/'modeling_llama_x.py','token_merging')
            self.prune=self.load(base/'modeling_llama_x.py','layer_prune','LlamaModel')
        elif method=='DUET':
            base=ROOT/'duet_vlm/official_source'
            for name in ('CLIPAttention_forward','CLIP_EncoderLayer_forward'):
                self.load(base/'visionzip/utils.py',name)
            self.vision_select=self.load(base/'visionzip/clip_encoder_cw.py','forward','CLIPVisionTower_VisionZip')
            self.load(base/'llava/model/modeling_llama_pdrop.py','rotate_half')
            self.load(base/'llava/model/modeling_llama_pdrop.py','apply_rotary_pos_emb')
            self.prune=self.load(base/'llava/model/modeling_llama_pdrop.py','pdrop_rank_drop','LlamaModel')
        elif method=='SparseVLM':
            base=ROOT/'sparsevlm_icml2025/official_source/llava/model/language_model'
            for name in ('scaled_dot_product_attention','batch_index_select','index_points','cluster_and_merge'):self.load(base/'utils.py',name)
            self.namespace.update(layer_dict={2:0,6:1,15:2},sparse_token_dict={192:[300,200,110]})
            self.load(base/'score.py','attn_postprocess_topk')
            self.prune,p=sparse_pruning_block(base/'modelling_sparse_llama.py',self.namespace);self.provenance.append(p)
        if method not in KERNELS:
            KERNELS[method]=dict(namespace=self.namespace,provenance=self.provenance,functions={name:getattr(self,name) for name in ('vision_merge','vision_encode','vision_select','prune') if hasattr(self,name)})
        if method=='ApET':self.initial_visual=288;self.stage_visual={16:96}
        elif method=='DUET':self.initial_visual=307;self.stage_visual={16:153,24:0}
        elif method=='SparseVLM':self.initial_visual=576;self.stage_visual={3:309,7:204,16:113}
        else:self.initial_visual=576;self.stage_visual={}
        # Proxies bridge removed attributes/API only. All Q/K/V/Norm modules are
        # the SAME frozen model objects. No alternative attention implementation.
        layers=[]
        for l in a.layers:
            att=l.self_attn;c=att.config
            layers.append(SimpleNamespace(input_layernorm=l.input_layernorm,self_attn=SimpleNamespace(
                q_proj=att.q_proj,k_proj=att.k_proj,v_proj=att.v_proj,
                num_heads=c.num_attention_heads,num_key_value_heads=c.num_key_value_heads,
                head_dim=att.head_dim,rotary_emb=LegacyRotary(self.decoder.rotary_emb))))
        config=SimpleNamespace(tokenizer_padding_side='right',tokenizer_model_max_length=2048,use_salient_tokens=False)
        self.proxy=SimpleNamespace(config=config,layers=layers,training=False,image_tokens=[self.initial_visual],
            image_token_posi=[layout.image[0]],image_token_list=[288,96],basis_token_num=10,
            image_token_ratio_list=[1,.5,0],prompt_len=[layout.length-576],top_rank_idxs=[])

    def load(self,path,name,owner=None):
        fn,p=extract(path,name,self.namespace,owner);self.provenance.append(p);return fn

    def vision(self,pixel_values,*args,**kwargs):
        self.vision_calls+=1
        if self.method in ('BridgeNative','SparseVLM'):return self.original_vision(pixel_values,*args,**kwargs)
        if self.method=='ApET':
            def tower(images):
                out=self.model.vision_tower(images,output_hidden_states=True)
                return out.hidden_states[-2],None
            model_proxy=SimpleNamespace(basis_token_num=10,mm_projector=self.model.multi_modal_projector,get_vision_tower=lambda:tower)
            proxy=SimpleNamespace(model=model_proxy,get_model=lambda:model_proxy,get_visual_token_num=lambda:288)
            proxy.token_merging=MethodType(self.vision_merge,proxy)
            full,mask=self.vision_encode(proxy,pixel_values)
            assert full.shape[1]==576 and mask.shape==(1,576)
            return full[mask].view(1,288,full.shape[-1])
        out,indices=self.vision_select(SimpleNamespace(vision_tower=self.model.vision_tower,device=pixel_values.device,dtype=pixel_values.dtype),pixel_values)
        assert out.shape[1]==307 and indices.shape[1]==300
        self.dominant_indices=indices
        return self.model.multi_modal_projector(out)

    def top_forward(self,model,input_ids=None,pixel_values=None,attention_mask=None,position_ids=None,past_key_values=None,inputs_embeds=None,cache_position=None,**kwargs):
        assert inputs_embeds is None
        if pixel_values is not None:
            assert past_key_values is None and input_ids.shape[1]==self.layout.length
            embeds=self.decoder.embed_tokens(input_ids)
            v=self.model.get_image_features(pixel_values,vision_feature_layer=-2,vision_feature_select_strategy='default')
            left=self.layout.image[0];right=self.layout.image[-1]+1
            h=torch.cat((embeds[:,:left],v,embeds[:,right:]),dim=1)
            return self.model.language_model(inputs_embeds=h,past_key_values=None,attention_mask=torch.ones((1,h.shape[1]),device=h.device,dtype=torch.long),**kwargs)
        return self.model.language_model(input_ids=input_ids,past_key_values=past_key_values,attention_mask=None,**kwargs)

    def decoder_forward(self,decoder,input_ids=None,attention_mask=None,position_ids=None,past_key_values=None,inputs_embeds=None,use_cache=True,output_attentions=False,output_hidden_states=False,return_dict=True,cache_position=None,**kwargs):
        assert use_cache and not output_hidden_states and not output_attentions
        h=decoder.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cache=DynamicCache() if past_key_values is None else past_key_values
        prefill=h.shape[1]>1;self.step=0 if prefill else self.step+1;self.forward_calls+=1
        visual=self.initial_visual
        if self.method=='SparseVLM' and prefill:
            start=self.layout.image[0];v_t=h[:,start:start+576];t_t=h[:,start+576:]
            m_v_t=(v_t@t_t.transpose(1,2)).softmax(2).mean(1)
            self.t_token_idx=torch.where(m_v_t>m_v_t.mean())
            assert self.t_token_idx[1].numel()>0,'SparseVLM text-rater selection empty'
        original_mask=None;last_shape=None;pos=rope=mask=None
        for i,layer in enumerate(decoder.layers):
            q=h.shape[1];past=cache.get_seq_length(i);size=(q,past)
            # RoPE and causal mask follow the official compact position convention
            # at every physical pruning stage. Decode uses each layer's real cache.
            if size!=last_shape:
                cp=torch.arange(past,past+q,device=h.device);pos=cp.unsqueeze(0)
                rope=decoder.rotary_emb(h,pos)
                mask=decoder._update_causal_mask(torch.ones((1,past+q),dtype=torch.long,device=h.device),h,cp,cache,False)
                if mask is not None:mask=mask[:,:,:q,:past+q]
                last_shape=size
            self.layer_visual[i]=visual
            if self.trace and self.step in (0,1,2,3,4,5,6,7,8,31):
                self.shapes.append(dict(step=self.step,layer=i,hidden_shape=list(h.shape),mask_shape=None if mask is None else list(mask.shape),RoPE_shape=[list(x.shape) for x in rope],position_first=int(pos[0,0]),position_last=int(pos[0,-1]),cache_before=past,visual=visual))
                assert q==(self.layout.length-576+visual if prefill else 1)
                assert int(pos[0,0])==(0 if prefill else past)
            h=layer(h,attention_mask=mask,position_ids=pos,past_key_value=cache,output_attentions=False,use_cache=True,cache_position=cp,position_embeddings=rope)[0]
            rank=i+1
            if rank in self.stage_visual:
                next_visual=self.stage_visual[rank]
                if prefill:
                    stage=list(self.stage_visual).index(rank)
                    if self.method=='SparseVLM':
                        new_h,vnum,p=self.prune(h,self.last_relation,self.layout.image[0],visual,self.layout.image[0]+visual,self.t_token_idx,i,pos)
                        assert int(vnum)==next_visual
                        self.last_relation=None
                    elif self.method=='ApET':
                        p,m,new_h,lab=self.prune(self.proxy,stage,rank,h,pos,None,None,None,None,None)
                    else:p,m,new_h,lab=self.prune(self.proxy,stage,rank,h,pos,None,None,None)
                    assert new_h.shape[1]==h.shape[1]-(visual-next_visual)
                    assert p is not None and torch.equal(p,torch.arange(new_h.shape[1],device=h.device).unsqueeze(0))
                    self.events.append(dict(after_layer=i,before_layer=rank,visual_before=visual,visual_after=next_visual,hidden_before=list(h.shape),hidden_after=list(new_h.shape),official_AST_unmodified=True))
                    h=new_h;last_shape=None
                visual=next_visual
        h=decoder.norm(h)
        result=BaseModelOutputWithPast(last_hidden_state=h,past_key_values=cache)
        return result if return_dict else result.to_tuple()

    def sparse_attention(self,att,hidden_states,position_embeddings,attention_mask=None,past_key_value=None,cache_position=None,**kwargs):
        """HF interfaces around the OFFICIAL manual eager attention function.

        Despite its class name 'Sdpa', the pinned official inference function
        is explicit QK matmul, FP16 softmax, dropout(0), AV matmul. We call this
        function unchanged; no fused SDPA/Flash kernel or ranking approximation.
        """
        b,q,_=hidden_states.shape;nh=att.config.num_attention_heads;nk=att.config.num_key_value_heads;d=att.head_dim
        query=att.q_proj(hidden_states).view(b,q,nh,d).transpose(1,2)
        key=att.k_proj(hidden_states).view(b,q,nk,d).transpose(1,2)
        value=att.v_proj(hidden_states).view(b,q,nk,d).transpose(1,2)
        query,key=hf_apply_rotary(query,key,*position_embeddings)
        if past_key_value is not None:key,value=past_key_value.update(key,value,att.layer_idx,dict(cos=position_embeddings[0],sin=position_embeddings[1],cache_position=cache_position))
        assert nk==nh==32 and b==1
        output,relation=self.namespace['scaled_dot_product_attention'](query,key,value,attn_mask=None,dropout_p=0.0,is_causal=q>1)
        if q>1 and att.layer_idx in (2,6,15):self.last_relation=relation
        output=output.transpose(1,2).contiguous().reshape(b,q,nh*d)
        return att.o_proj(output),None

    @contextmanager
    def context(self):
        self.vision_calls=0;self.original_vision=self.model.get_image_features
        original_top=self.model.forward;original_decoder=self.decoder.forward;vision_saved=[];attention_saved=[]
        try:
            if self.method=='DUET':
                vt=self.model.vision_tower;self.vision_info=dict(r=[0]*22+[1,0],dominant=299,contextual=7,cluster_width=4)
                original_info=getattr(vt,'_info',None);vt._info=self.vision_info
                for layer in vt.vision_model.encoder.layers:
                    vision_saved.append((layer,layer.forward,layer.self_attn.forward,getattr(layer,'_info',None)))
                    layer.forward=MethodType(self.namespace['CLIP_EncoderLayer_forward'],layer)
                    layer.self_attn.forward=MethodType(self.namespace['CLIPAttention_forward'],layer.self_attn)
                    layer._info=self.vision_info
            if self.method=='SparseVLM':
                for layer in self.a.layers:
                    attention_saved.append((layer.self_attn,layer.self_attn.forward))
                    layer.self_attn.forward=MethodType(self.sparse_attention,layer.self_attn)
            self.model.get_image_features=self.vision
            self.model.forward=MethodType(self.top_forward,self.model)
            self.decoder.forward=MethodType(self.decoder_forward,self.decoder)
            yield self
        finally:
            self.model.get_image_features=self.original_vision;self.model.forward=original_top;self.decoder.forward=original_decoder
            for att,forward in attention_saved:att.forward=forward
            for layer,lf,af,info in vision_saved:
                layer.forward=lf;layer.self_attn.forward=af
                if info is None:delattr(layer,'_info')
                else:layer._info=info
                if hasattr(layer,'metric'):delattr(layer,'metric')
            if self.method=='DUET':
                if original_info is None:delattr(vt,'_info')
                else:vt._info=original_info
