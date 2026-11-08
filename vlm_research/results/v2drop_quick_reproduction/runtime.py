"""One cold-request loop and actual-operator counter for all four methods."""
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import time
import torch
from torch.utils.flop_counter import FlopCounterMode
from hf_adapter import V2Drop
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
from vlm_research.physical_handoff_fastpath.engine import Layout,native_cache_info

HERE=Path(__file__).resolve().parent


class Shapes:
    def __init__(self,a):self.a=a;self.step=0;self.rows={};self.handles=[]
    def add(self,layer,op,args):
        x=args[0];key=(self.step,layer);r=self.rows.setdefault(key,{'step':self.step,'layer':layer})
        assert op not in r,(key,op)
        r[op]=int(x.shape[-2])
    def __enter__(self):
        for i,layer in enumerate(self.a.layers):
            self.handles.append(layer.register_forward_pre_hook(lambda m,args,kw,i=i:self.add(i,'hidden',args),with_kwargs=True))
            for op in ('q_proj','k_proj','v_proj'):
                self.handles.append(getattr(layer.self_attn,op).register_forward_pre_hook(lambda m,args,i=i,op=op:self.add(i,op,args)))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
    def finish_step(self,cache):
        for i in range(32):
            r=self.rows[self.step,i];q=r.get('q_proj',0)
            live=cache.used[i] if hasattr(cache,'used') else cache.get_seq_length(i)
            r['attention_Q']=q;r['attention_K']=r.get('k_proj',0) if self.step==0 else live if q else 0
            r['persistent_KV_tokens']=int(live)
        self.step+=1
    def snapshot(self,limit):
        out=[]
        for i in range(32):
            rr=[self.rows[s,i] for s in range(limit)]
            out.append({'layer':i,'prefill':rr[0],'first_decode':rr[1] if limit>1 else None,'last_decode':rr[-1],
                        'total_Q':sum(r['attention_Q'] for r in rr),'total_QK':sum(r['attention_Q']*r['attention_K'] for r in rr),
                        'q_projection_tokens':sum(r.get('q_proj',0) for r in rr),'k_projection_tokens':sum(r.get('k_proj',0) for r in rr),
                        'v_projection_tokens':sum(r.get('v_proj',0) for r in rr)})
        return out


def cache_snapshot(cache,layout,method,ctrl,steps):
    n=layout.length;image=set(layout.image);rows=[]
    if hasattr(cache,'audit'):
        info=cache.audit()
        for r in info['layers']:
            r=dict(r);r['visual_tokens']=sum(p in image for p in r['original_positions']);rows.append(r)
    else:
        info=native_cache_info(cache)
        for i,r in enumerate(info['layers']):
            r=dict(r)
            if method in ('V2Drop','V2Drop-NoPrune'):
                count=576 if method=='V2Drop-NoPrune' or i<3 else 288 if i<17 else 144 if i<22 else 1
                # Actual cache length plus observed hidden rows checks retention;
                # trace mode records the exact original selected visual positions.
                assert r['used_tokens']==ctrl.prefill_rows[i]+steps-1
                assert ctrl.prefill_rows[i]==n-576+count
                if ctrl.trace:positions=list(ctrl.kv_positions[i])
                else:positions=None
            else:count=576;positions=list(range(n+steps-1))
            r.update(visual_tokens=count,original_positions=positions);rows.append(r)
    unit=2*32*128*2
    for r in rows:
        r['visual_live_bytes']=r['visual_tokens']*unit
        assert r['live_bytes']==r['used_tokens']*unit
        if r['original_positions'] is not None:assert len(r['original_positions'])==r['used_tokens']
    return {'layers':rows,'live_bytes':sum(r['live_bytes'] for r in rows),'storage_bytes':sum(r['storage_bytes'] for r in rows),
            'visual_live_bytes':sum(r['visual_live_bytes'] for r in rows),
            'visual_KV_residency_percent':100*sum(r['visual_live_bytes'] for r in rows)/(576*32*unit),
            'allocation':'Selective in-place ATE KV' if hasattr(cache,'audit') else 'Native DynamicCache; unequal per-layer lengths permitted',
            'terminal_decode_inputs_cached':steps-1}


@torch.inference_mode()
def generate(a,layout,method,length,fixed=False,capture=False,trace=False,count=False,reference=False):
    assert method in ('Native','V2Drop','V2Drop-NoPrune','C3-V2','C3-V2+L11-L24')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated();start=time.perf_counter()
    ctrl=None
    if method.startswith('V2Drop'):
        ctrl=V2Drop(a,layout.image,HERE/'official_sources/V2Drop.py',enabled=method=='V2Drop',trace=trace,reference=reference,selection_timing=count or not capture)
    elif method.startswith('C3-V2'):
        from dataclasses import replace
        keep=tuple(range(11,25)) if '+' in method else ()
        layout=replace(layout,cutoff=25 if keep else 20)
        ctrl=MiddleWriter(a,layout,keep,length,trace,False)
    n=layout.length;device=a.inputs['input_ids'].device
    positions=torch.arange(n+length-1,device=device);mask=torch.ones((1,n+length-1),dtype=torch.long,device=device)
    cache=None;tokens=[];finite=[];logits_capture=[];events=[];snapshots={};first_wall=None
    counter=FlopCounterMode(display=False) if count else None;shape=Shapes(a) if count else None
    vision_flops=[0];vision_calls=[0]
    original=a.model.get_image_features
    def vision(*args,**kwargs):
        before=counter.get_total_flops() if counter else 0
        result=original(*args,**kwargs);vision_calls[0]+=1
        assert result.is_cuda
        if counter:vision_flops[0]+=counter.get_total_flops()-before
        return result
    try:
        with ctrl.context() if ctrl else nullcontext():
            with shape if shape else nullcontext(),counter if counter else nullcontext(),patch.object(a.model,'get_image_features',vision):
                for step in range(length):
                    kw=dict(a.inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=mask[:,:n+step])
                    kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],logits_to_keep=1)
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                    out=a.model(**kw);last=out.logits[:,-1];token=last.argmax(-1);tokens.append(token)
                    finite.append(torch.isfinite(last).all());end.record();events.append((begin,end));cache=out.past_key_values
                    if capture:logits_capture.append(last.detach().float().cpu())
                    if shape:shape.finish_step(cache)
                    if step==0:
                        if not count:torch.cuda.synchronize();first_wall=(time.perf_counter()-start)*1000
                        prefill_flops=counter.get_total_flops() if counter else None
                    if count and step+1 in (32,128):
                        snapshots[str(step+1)]={'full_request_flops':counter.get_total_flops(),'prefill_flops':prefill_flops,'vision_flops':vision_flops[0],
                            'language_prefill_flops':prefill_flops-vision_flops[0],'decode_flops':counter.get_total_flops()-prefill_flops,
                            'shape_audit':shape.snapshot(step+1),'cache':cache_snapshot(cache,layout,method,ctrl,step+1),
                            'operator_flops':{str(k):v for k,v in counter.get_flop_counts()['Global'].items()}}
                    del out,last
                    if not fixed and int(token)==2:break
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        ids=torch.cat(tokens).tolist();assert torch.stack(finite).all().item(),'Nonfinite logits'
        assert vision_calls[0]==1,'Every request must perform real vision encoding'
        kv=cache_snapshot(cache,layout,method,ctrl,len(ids));step_ms=[a.elapsed_time(b) for a,b in events]
        text=a.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        result={'condition':method,'token_ids':ids,'text':text,'eos':ids[-1]==2,'empty':not bool(text),'truncated':not fixed and len(ids)==length and ids[-1]!=2,
                'fixed':fixed,'output_tokens':len(ids),'decode_steps':len(ids)-1,'all_logits_finite':True,'vision_forward_calls':1,
                'ttft_ms':first_wall,'gpu_prefill_ms':step_ms[0],'decode_ms':sum(step_ms[1:]),'decode_TPOT_ms':sum(step_ms[1:])/max(1,len(ids)-1),
                'E2E_ms':elapsed,'peak_allocated_MiB':peak/2**20,'peak_reserved_MiB':reserved/2**20,'initial_allocated_MiB':initial/2**20,
                'step_gpu_ms':step_ms,'cache':kv,'counter_enabled':count,'counter_checkpoints':snapshots}
        if isinstance(ctrl,V2Drop):
            result.update(pruning_events=ctrl.events,selection_overhead_gpu_ms=ctrl.timing(),layer_hidden_trace=ctrl.layers,selector_provenance=ctrl.provenance)
        return result,logits_capture
    finally:
        del ctrl,cache
