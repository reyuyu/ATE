"""Observe actual native eager probabilities; release every full layer matrix."""
from contextlib import AbstractContextManager
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from semantic_visual_writers.engine import apply_mask
from paper_rq1_single_layer_gqa6.run import tensor_hashes
from benchmarks.smoke import hook_footprint
from .common import *
class Capture(AbstractContextManager):
    def __init__(self,model,inputs,allwriter=False,keep_image=False):
        self.model=model;self.layout=roles(inputs,model.config.image_token_index);self.allwriter=allwriter;self.keep_image=keep_image;self.handles=[];self.state={};self.mass=[];self.legal=[];self.probs=[];self.audit=[];self.MLP=[0]*32;self.visual=[]
        self.queries=self.layout['R']+[self.layout['prompt_length']-1]
    def __enter__(self):
        try:
            def visual(m,args,out):self.visual.append(tensor_hashes({'visual':out})['visual'])
            self.handles.append(self.model.multi_modal_projector.register_forward_hook(visual))
            for k,layer in enumerate(model_layers(self.model)):
                def mlp(m,args,out,k=k):assert torch.isfinite(out).all();self.MLP[k]+=1
                def before(module,args,kw,k=k):
                    n=self.layout['prompt_length'];old=kw['attention_mask'];cp=kw['cache_position'];cache=kw.get('past_key_value')
                    assert kw['hidden_states'].shape[:2]==(1,n) and old.shape==(1,1,n,n) and kw.get('output_attentions') is True
                    assert cache is None or cache.get_seq_length(k)==0
                    assert torch.equal(cp,torch.arange(n,device=cp.device));allowed=old[0,0]==0
                    assert torch.equal(allowed,torch.ones(n,n,device=old.device,dtype=torch.bool).tril())
                    original=old.clone()
                    if self.allwriter:
                        changed,blocked=apply_mask(old,self.layout,self.layout['R'],0)
                        assert blocked.sum()==576 and not blocked[self.layout['R']].any()
                        assert torch.equal(changed[0,0][~blocked],old[0,0][~blocked])
                    else:changed=old
                    assert torch.equal(old,original)
                    # Keep only the immutable native/actual legality and small hashes.
                    self.state[k]=(changed[0,0]==0,sha256_tensor(original),sha256_tensor(changed),int((changed!=old).sum()))
                    return (args,{**kw,'attention_mask':changed}) if self.allwriter else None
                def after(module,args,out,k=k):
                    w=out[1];legal,h0,h1,modified=self.state.pop(k);n=self.layout['prompt_length'];assert w is not None and w.shape==(1,32,n,n)
                    assert torch.isfinite(w).all() and (w>=0).all();err=float((w.float().sum(-1)-1).abs().max());assert err<=ROW_ATOL
                    assert torch.count_nonzero(w[0,:,~legal])==0
                    if not self.allwriter:assert h0==h1 and modified==0
                    rows=w[0,:,self.queries,:].detach().cpu().numpy().copy()
                    image=self.layout['I'];mass=rows[:,:,image].astype(np.float64).sum(-1);counts=legal[self.queries].sum(-1).cpu().numpy();assert (counts>0).all()
                    self.mass.append(mass);self.legal.append(counts)
                    if self.keep_image:self.probs.append(rows[:,:-1,image].copy())
                    self.audit.append(dict(layer=k,all_prefill_rows_finite_nonnegative=True,all_prefill_row_sum_max_error=err,illegal_probability_exact_zero=True,native_mask_sha256=h0,actual_mask_sha256=h1,mask_modified_edges=modified))
                    # Actual attention output is untouched; only suppress retained matrix.
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(layer.mlp.register_forward_hook(mlp));self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True));self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*_):
        for h in self.handles:h.remove()
        self.handles=[]
    def finish(self):
        assert not self.handles and len(self.mass)==32 and [a['layer'] for a in self.audit]==list(range(32));assert self.MLP==[1]*32 and len(self.visual)==1
        m=np.stack(self.mass);legal=np.stack(self.legal);assert m.shape==(32,32,len(self.queries)) and legal.shape==(32,len(self.queries))
        return m,legal,np.stack(self.probs) if self.keep_image else None,dict(passed=True,queries=self.queries,layer_audits=self.audit,MLP_calls=self.MLP,visual_features_sha256=self.visual[0],all32_actual_eager_attentions=True,full_matrices_not_retained=True,p_minus_one_separate=True,row_sum_fp16_tolerance=ROW_ATOL)
def sha256_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
@torch.inference_mode()
def observed(model,inputs,allwriter=False,keep_image=False):
    before=tensor_hashes(inputs);fp=hook_footprint(model);versions={n:v._version for n,v in model.named_parameters()}
    with Capture(model,inputs,allwriter,keep_image) as cap:
        result=model(**inputs,past_key_values=None,use_cache=True,output_attentions=True,return_dict=True,logits_to_keep=1)
    assert result.logits.shape==(1,1,model.config.text_config.vocab_size) and torch.isfinite(result.logits).all();assert all(x is None for x in result.attentions)
    assert fp==hook_footprint(model) and versions=={n:v._version for n,v in model.named_parameters()} and before==tensor_hashes(inputs)
    cache=result.past_key_values;assert all(cache.get_seq_length(k)==inputs['input_ids'].shape[1] for k in range(32))
    m,legal,probs,audit=cap.finish();audit.update(input_hashes=before,weights_unchanged=True,hook_cleanup=True,independent_single_prefill=True)
    return result.logits.detach().cpu().clone(),m,legal,probs,audit,cache
@torch.inference_mode()
def plain(model,inputs):
    visual=[]
    def f(m,args,out):visual.append(sha256_tensor(out))
    h=model.multi_modal_projector.register_forward_hook(f)
    try:r=model(**inputs,past_key_values=None,use_cache=True,output_attentions=False,return_dict=True,logits_to_keep=1)
    finally:h.remove()
    assert len(visual)==1
    return r.logits.detach().cpu().clone(),visual[0],r.past_key_values
