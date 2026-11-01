"""Read-only original-query states around the unchanged RQ2.5 hard mask."""
from contextlib import AbstractContextManager

import torch

from knockout_probe import model_layers
from rq2_source_knockout.engine import Observer
from rq2_prompt_five.engine import MultiPromptKnockout,generate

EARLY=list(range(2,13))
MIDDLE=list(range(13,17))
CONDITIONS={'E':EARLY,'EM':list(range(2,17)),'L':list(range(17,32))}
ALL={'F':[],'E':EARLY,'M':MIDDLE,'EM':CONDITIONS['EM'],'L':CONDITIONS['L']}
CHECKPOINTS=[2,6,9,13,17,31]


class StateCapture(AbstractContextManager):
    def __init__(self,model,inputs,layers=CHECKPOINTS):
        self.blocks=model_layers(model);self.layers=list(layers)
        if len(set(self.layers))!=len(self.layers) or any(k<0 or k>=len(self.blocks) for k in self.layers):
            raise ValueError('Invalid fixed state checkpoints')
        mask=inputs['attention_mask'][0]
        if not bool(mask.all()):raise ValueError('Unpadded original input required')
        self.n=int(mask.numel());self.query=self.n-1;self.vectors={};self.handles=[]

    def __enter__(self):
        try:
            for layer in self.layers:
                def capture(module,args,kwargs,layer=layer):
                    h=args[0] if args else kwargs['hidden_states']
                    if h.shape[1]!=self.n:return
                    if layer in self.vectors:raise AssertionError('Repeated original Prefill')
                    vector=h[0,self.query].detach().float().cpu().clone()
                    if not torch.isfinite(vector).all():raise AssertionError('Nonfinite P-1 state')
                    self.vectors[layer]=vector
                self.handles.append(self.blocks[layer].register_forward_pre_hook(capture,with_kwargs=True))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles.clear()

    def states(self):
        if set(self.vectors)!=set(self.layers) or self.handles:raise AssertionError('Incomplete state capture or hook leak')
        return torch.stack([self.vectors[k] for k in self.layers])


def captured_generate(model,inputs,bos_id,blocked,capture_layers=CHECKPOINTS,**kwargs):
    with StateCapture(model,inputs,capture_layers) as capture:
        result=generate(model,inputs,bos_id,blocked,**kwargs)
    return (*result,capture.states())


@torch.inference_mode()
def prefill(model,inputs,bos_id,blocked,capture_layers=CHECKPOINTS,audit=False,expected_visual=576):
    mask=MultiPromptKnockout(model,inputs,bos_id,blocked,expected_visual=expected_visual)
    observer=Observer(model,inputs,bos_id,audit,expected_visual)
    with StateCapture(model,inputs,capture_layers) as capture,mask,observer:
        output=model(**inputs,use_cache=True,output_attentions=True,return_dict=True)
    mask.assert_one_forward()
    if [r['layer'] for r in mask.records if r['applied']]!=sorted(blocked):raise AssertionError('Wrong mask scope')
    if any(observer.rows[k]['Prompt_max']!=0 for k in blocked):raise AssertionError('Prompt probability nonzero')
    if not torch.isfinite(output.logits).all() or mask.handles or observer.handles:raise AssertionError('Invalid Prefill')
    logits=output.logits.detach().float().cpu().clone() if audit else output.logits[0,-1].detach().float().cpu().clone()
    return logits,capture.states(),observer.rows


def state_metrics(full,changed):
    a=torch.as_tensor(full,dtype=torch.float32);b=torch.as_tensor(changed,dtype=torch.float32)
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Invalid states')
    na=torch.linalg.vector_norm(a);nb=torch.linalg.vector_norm(b)
    if na<=0:raise ValueError('Zero Full state denominator')
    exact=torch.equal(a,b)
    cosine=(0. if exact else float(1-(torch.dot(a,b)/(na*nb)).clamp(-1,1))) if nb>0 else None
    return dict(full_norm=float(na),state_norm=float(nb),norm_change=float(nb-na),
        relative_norm_change=float((nb-na)/na),l2_distance=float(torch.linalg.vector_norm(b-a)),
        relative_l2_distance=float(torch.linalg.vector_norm(b-a)/na),cosine_distance=cosine,exact_full=exact)
