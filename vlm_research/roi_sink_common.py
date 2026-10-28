"""Phase 3.4 predeclared geometry, spatial controls, statistics and safe hooks."""
from contextlib import AbstractContextManager, nullcontext
import math
import random
import time
import numpy as np
from PIL import Image
from vlm_common import ROOT, gpu_memory
from knockout_probe import VisualAttentionKnockout, model_layers, ROW_ATOL
import torch

OUTPUT=ROOT/'results/phase3_4_roi_sink_diagnostic'
SEED=20261030
NEUTRAL=['Describe this image.', 'What can be seen in the image?', 'Summarize the scene.']
RANDOM_SEEDS=[3401,3402,3403]
WINDOWS={'middle':list(range(8,16)),'deep':list(range(24,32))}
PROTOCOL=dict(seed=SEED,neutral_prompts=NEUTRAL,discovery_prompt_indices=[0,1],holdout_prompt_index=2,
 sink_rule='For each patch and each of four discovery views (two prompts x mean real Question queries / P-1), count all 32 layers where head-mean visual-conditional probability >=5/576 AND patch is in top58 (stable index ties). A view is persistent at >=8 layers. Candidate requires >=3/4 persistent views. No ROI, labels, margins or depth-specific preference used. Holdout uses both Q/A views persistent at >=8 layers. Empty sets remain empty.',
 sink_enrichment_threshold=5.,sink_min_layers=8,sink_min_discovery_views=3,top_k=58,
 geometry='Half-open original xywh boxes; actual CLIP resize integer output and floor center crop. A patch belongs to ROI iff positive continuous intersection area; row-major r*24+c. Also save overlap fractions and center-inside sensitivity.',
 selection='Exclude designated five historical case IDs from main. Both objects >=2 patches; union 4..192; max/min object patch count<=16. Sort qid, Python Random(20261030), shuffle left then right and take24 each. Geometry only, no predictions. Cases reported separately even if not eligible.',
 random_seeds=RANDOM_SEEDS,random_rule='Separate controls matched to ROI union and non-ROI sink counts. Pool excludes ROI union and all sink candidates. Greedy one-to-one nearest available 6x6-patch spatial bin, ties shuffled by fixed seed; record bin Manhattan displacement. Exact patch count, no replacements. Unavailable controls explicitly skipped, never substitute targets.',
 windows=WINDOWS,query_groups=['question','answer'],causal_conditions=['roi','sink_nonroi','random_roi','random_sink'],
 precision='FP16 eager unchanged. Native Full and historical scores exact. Empty key set is an explicit sham. Smoke mask/causal/cleanup checks and attention on/off exact. No tolerance changes; weak margins |Full or intervened|<=0.15 flagged, not excluded.',
 interpretation='ROI is box coverage, not segmentation. Sink candidates are neutral-prompt persistent hotspots, not proven irrelevant noise. Attention association, mask effects and physical exit effects are distinct.')


def geometry(image,ip):
    if not ip.do_resize or not ip.do_center_crop or ip.size!={'shortest_edge':336} or ip.crop_size!={'height':336,'width':336}:
        raise ValueError('Unexpected actual CLIP geometry; investigate')
    arr=np.asarray(image.convert('RGB'))
    resized=ip.resize(arr,size=ip.size,resample=ip.resample,input_data_format='channels_last',data_format='channels_last')
    rh,rw=resized.shape[:2];top,left=(rh-336)//2,(rw-336)//2
    cropped=ip.center_crop(resized,size=ip.crop_size,input_data_format='channels_last',data_format='channels_last')
    if not np.array_equal(cropped,resized[top:top+336,left:left+336]):raise AssertionError('Actual center crop differs')
    actual=ip(images=image,return_tensors='np')['pixel_values'][0]
    reconstructed=ip.normalize(ip.rescale(cropped,scale=ip.rescale_factor,input_data_format='channels_last'),mean=ip.image_mean,std=ip.image_std,input_data_format='channels_last',data_format='channels_first')
    if not np.array_equal(actual,reconstructed):raise AssertionError('Pixel reconstruction not exact actual Processor')
    w,h=image.size
    return dict(original_size=[w,h],resized_size=[rw,rh],crop_left=left,crop_top=top,
                scale_x=rw/w,scale_y=rh/h,pixel_values_exact=True),cropped


def map_box(box,g):
    x,y,w,h=map(float,box)
    if w<=0 or h<=0:raise ValueError('Nonpositive official box')
    x0=x*g['scale_x']-g['crop_left'];x1=(x+w)*g['scale_x']-g['crop_left']
    y0=y*g['scale_y']-g['crop_top'];y1=(y+h)*g['scale_y']-g['crop_top']
    clip=[max(0.,min(336.,x0)),max(0.,min(336.,y0)),max(0.,min(336.,x1)),max(0.,min(336.,y1))]
    area=max(0.,clip[2]-clip[0])*max(0.,clip[3]-clip[1])
    fractions=np.zeros((24,24),dtype=float);centers=[]
    for r in range(24):
        for c in range(24):
            a=max(0.,min((c+1)*14,clip[2])-max(c*14,clip[0]))*max(0.,min((r+1)*14,clip[3])-max(r*14,clip[1]))
            fractions[r,c]=a/196
            if clip[0]<=(c+.5)*14<clip[2] and clip[1]<=(r+.5)*14<clip[3]:centers.append(r*24+c)
    if not np.isclose(fractions.sum()*196,area):raise AssertionError('Patch areas do not tile clipped box')
    return dict(original_xywh=list(map(float,box)),unclipped_crop_xyxy=[x0,y0,x1,y1],crop_xyxy=clip,
                visible_fraction=area/((x1-x0)*(y1-y0)),patch_indices=np.flatnonzero(fractions.ravel()>0).tolist(),
                patch_count=int((fractions>0).sum()),center_patch_indices=centers,patch_overlap_fraction=fractions.ravel().tolist())


def sink_candidates(maps):
    # [prompts, layers, groups, patches], head-mean raw Attention, no outcomes.
    maps=np.asarray(maps,dtype=np.float64)
    if maps.shape[1:]!=(32,2,576) or maps.shape[0]<2 or not np.isfinite(maps).all():raise ValueError('Invalid neutral maps')
    cond=maps/np.maximum(maps.sum(-1,keepdims=True),1e-30)
    top=np.zeros_like(cond,dtype=bool)
    idx=np.argsort(-cond,axis=-1,kind='stable')[...,:58];np.put_along_axis(top,idx,True,axis=-1)
    high=(cond>=5/576)&top
    persistent=high.sum(axis=1)>=8  # prompt,group,patch
    selected=np.flatnonzero(persistent[:2].sum(axis=(0,1))>=3).tolist()
    holdout=np.flatnonzero(persistent[2].sum(axis=0)>=2).tolist() if len(maps)>2 else []
    union=set(selected)|set(holdout);intersection=set(selected)&set(holdout)
    return dict(indices=selected,holdout_indices=holdout,holdout_jaccard=len(intersection)/len(union) if union else None,
                holdout_retention=len(intersection)/len(selected) if selected else None,
                persistent_view_counts=persistent[:2].sum(axis=(0,1)).tolist())


def matched_background(target,pool,seed):
    target=sorted(set(target));pool=sorted(set(pool))
    if len(pool)<len(target):return dict(available=False,reason='Not enough disjoint background patches',target_count=len(target),pool_count=len(pool))
    rng=random.Random(seed);rng.shuffle(pool);order=target.copy();rng.shuffle(order)
    result=[];distances=[]
    def loc(i):return (i//24//6,i%24//6)
    for t in order:
        tr,tc=loc(t);d=[abs(loc(i)[0]-tr)+abs(loc(i)[1]-tc) for i in pool]
        j=int(np.argmin(d));result.append(pool.pop(j));distances.append(d[j])
    return dict(available=True,indices=sorted(result),target_count=len(target),seed=seed,
                exact_count=True,mean_bin_manhattan=float(np.mean(distances)) if distances else 0.,
                max_bin_manhattan=max(distances,default=0),exact_bin_fraction=float(np.mean(np.array(distances)==0)) if distances else 1.)


class PatchKnockout(VisualAttentionKnockout):
    def __init__(self,layers,inputs,image_token_id,queries,patches,active,expected_visual=576):
        super().__init__(layers,inputs,image_token_id,0,expected_visual,only_layers=active)
        q=list(queries);p=list(patches)
        if len(q)!=len(set(q)) or len(p)!=len(set(p)) or any(type(i)is not int or not 0<=i<expected_visual for i in p):raise ValueError('Invalid target indices')
        qm=torch.zeros_like(self.layout['text_queries'])
        if not q or any(type(i)is not int or not 0<=i<len(qm) for i in q):raise ValueError('Invalid query')
        qm[q]=True
        if (qm&~self.layout['text_queries']).any():raise ValueError('Non-postimage query')
        vm=torch.zeros_like(self.layout['visual']);vm[self.layout['visual_indices'][p]]=True
        self.layout=dict(self.layout,text_queries=qm,visual=vm)


class AttentionRows(AbstractContextManager):
    """Stream needed rows; discard returned full attention weights after each layer."""
    def __init__(self,model,inputs,qpos,apos,patch_hook=None,norms=False,expected_visual=576):
        self.model=model;self.inputs=inputs;self.qpos=qpos;self.apos=apos;self.patch_hook=patch_hook;self.norms=norms
        self.visual=torch.where(inputs['input_ids'][0]==model.config.image_token_index)[0]
        if len(self.visual)!=expected_visual:raise ValueError('Wrong visual indices')
        self.handles=[];self.rows=[];self.audit=[];self.k=[];self.v=[];self.activation=[];self.phi=[];self.final=None
    def __enter__(self):
        try:
            self.handles.append(self.model.language_model.model.norm.register_forward_hook(lambda m,a,o:setattr(self,'final',o[:,self.qpos].detach().cpu())))
            for i,layer in enumerate(model_layers(self.model)):
                self.handles.append(layer.self_attn.register_forward_hook(self.capture(i)))
                if self.norms:
                    self.handles.append(layer.register_forward_pre_hook(self.capture_phi,with_kwargs=True))
                    for name,dest in [('k_proj',self.k),('v_proj',self.v)]:
                        self.handles.append(getattr(layer.self_attn,name).register_forward_hook(self.capture_norm(dest)))
                    self.handles.append(layer.self_attn.register_forward_pre_hook(self.capture_activation,with_kwargs=True))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def capture_norm(self,dest):
        def hook(m,args,out):
            heads=self.model.config.text_config.num_attention_heads
            t=out[0,self.visual].float().reshape(len(self.visual),heads,-1).norm(dim=-1).T
            dest.append(t.cpu().numpy())
        return hook
    def capture_activation(self,m,args,kwargs):
        x=kwargs['hidden_states'][0,self.visual].float();self.activation.append(x.abs().max(-1).values.cpu().numpy())
    def capture_phi(self,m,args,kwargs):
        x=(args[0] if args else kwargs['hidden_states'])[0,self.visual].float()
        if x.shape[-1]!=4096:raise ValueError('Published sink dimensions require 4096 hidden size')
        self.phi.append((x[:,[1415,2533]].max(-1).values/x.square().mean(-1).sqrt().clamp_min(1e-30)).cpu().numpy())
    def capture(self,index):
        def hook(m,args,out):
            attn=out[1];n=self.inputs['input_ids'].shape[1];heads=self.model.config.text_config.num_attention_heads
            if attn is None or tuple(attn.shape)!=(1,heads,n,n):raise AssertionError('Expected eager attention matrix')
            selected=attn[0,:,self.qpos+[self.apos],:]
            if not torch.isfinite(selected).all() or (selected<0).any():raise AssertionError('Invalid selected attention')
            error=float((selected.float().sum(-1)-1).abs().max())
            if error>ROW_ATOL:raise AssertionError('Selected attention row sum')
            q=selected[:,:len(self.qpos),:].index_select(-1,self.visual).float().mean(1)
            a=selected[:,-1,:].index_select(-1,self.visual).float()
            blocked=0.
            if self.patch_hook is not None and index in self.patch_hook.active:
                target=attn[0,:,self.patch_hook.layout['text_queries'],:][:,:,self.patch_hook.layout['visual']]
                blocked=float(target.max()) if target.numel() else 0.
                if blocked!=0:raise AssertionError('Target attention not zero')
            self.rows.append(torch.stack([q,a],dim=1).cpu().numpy().astype(np.float16))
            self.audit.append(dict(layer=index,shape=list(attn.shape),row_sum_error=error,target_attention_max=blocked))
            return (out[0],None)+tuple(out[2:])
        return hook
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[]


def forward(model,inputs,qpos,apos,queries=None,patches=None,active=None,observe=False,norms=False,expected_visual=576):
    layers=model_layers(model);before=[(len(l._forward_pre_hooks),len(l.self_attn._forward_pre_hooks),len(l.self_attn._forward_hooks),len(l.self_attn.k_proj._forward_hooks),len(l.self_attn.v_proj._forward_hooks)) for l in layers]
    final=[];handle=None;cuda=inputs['input_ids'].is_cuda
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();patch=None;stream=None
    try:
        context=nullcontext()
        if queries is not None:
            patch=PatchKnockout(layers,inputs,model.config.image_token_index,queries,patches,active,expected_visual);context=patch
        if not observe:handle=model.language_model.model.norm.register_forward_hook(lambda m,a,o:final.append(o[:,qpos].detach().cpu()))
        with torch.inference_mode(),context:
            if observe:
                with AttentionRows(model,inputs,qpos,apos,patch,norms,expected_visual) as stream:
                    output=model(**inputs,output_attentions=True,use_cache=False,return_dict=True)
            else:output=model(**inputs,output_attentions=False,use_cache=False,return_dict=True)
        if patch:patch.assert_one_forward()
        if cuda:torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        if output.past_key_values is not None or not torch.isfinite(output.logits).all():raise AssertionError('Invalid logits/cache')
        qhidden=stream.final if stream else final[0]
        if not torch.isfinite(qhidden).all():raise AssertionError('Invalid final question hidden')
        result=dict(prediction_logits=output.logits[0,apos].detach().cpu(),question_hidden=qhidden,
                    measurement=dict(forward_seconds=seconds,gpu_memory=gpu_memory() if cuda else None),
                    hook_audit=patch.records if patch else [],attention_audit=stream.audit if stream else [])
        if stream:
            if len(stream.rows)!=len(layers):raise AssertionError('Missing layer rows')
            result['rows']=np.stack(stream.rows)
            if norms:result.update(key_norm=np.stack(stream.k),value_norm=np.stack(stream.v),activation_max=np.stack(stream.activation),sink_phi=np.stack(stream.phi))
        return result
    finally:
        if handle:handle.remove()
        after=[(len(l._forward_pre_hooks),len(l.self_attn._forward_pre_hooks),len(l.self_attn._forward_hooks),len(l.self_attn.k_proj._forward_hooks),len(l.self_attn.v_proj._forward_hooks)) for l in layers]
        if after!=before:raise AssertionError('Hook leak')
