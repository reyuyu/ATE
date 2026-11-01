"""Read-only native attention rows; no mask, replay, generation, or intervention."""
import numpy as np
import torch


class AttentionRows:
    def __init__(self, model, layout):
        self.model,self.layout=model,layout
        self.handles,self.records=[],{}

    def observe(self, layer, module, args, output):
        weights=output[1]
        n=self.layout['input_length']
        assert weights.ndim==4 and weights.shape[0]==1 and weights.shape[-2:]==(n,n)
        assert layer not in self.records
        image=self.layout['image_positions']
        q=self.layout['question_positions']
        assert q and not set(q)&set(image) and self.layout['p_minus_1']==n-1
        vectors,errors=[],[]
        for positions in ([n-1],q):
            selected=weights[0,:,positions,:].float()
            assert torch.isfinite(selected).all() and (selected>=0).all()
            error=float((selected.sum(-1)-1).abs().max())
            assert error<=2**-8+2e-6, 'Existing BF16 row-sum audit threshold exceeded'
            mean=selected.mean((0,1))
            vectors.append(mean[image].cpu().numpy().copy())
            errors.append(error)
        self.records[layer]=dict(visual=np.stack(vectors),row_sum_errors=errors)
        # Returning None preserves the native self-attention output object.

    def __enter__(self):
        try:
            for i,layer in enumerate(self.model.model.language_model.layers):
                self.handles.append(layer.self_attn.register_forward_hook(
                    lambda m,a,o,j=i:self.observe(j,m,a,o)))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self,*_):
        for h in self.handles:h.remove()
        self.handles.clear()

    def arrays(self):
        assert set(self.records)==set(range(36))
        return np.stack([self.records[i]['visual'] for i in range(36)])


class NativeDeepStack:
    def __init__(self,model,visual):
        self.model,self.visual=model,visual
        self.handles,self.features,self.before,self.after=[],{},{},{}

    def __enter__(self):
        for i,merger in enumerate(self.model.model.visual.deepstack_merger_list):
            self.handles.append(merger.register_forward_hook(
                lambda m,a,o,j=i:self.features.__setitem__(j,o.detach().cpu().clone())))
        for i,layer in enumerate(self.model.model.language_model.layers[:4]):
            if i>0:
                self.handles.append(layer.register_forward_pre_hook(
                    lambda m,a,j=i:self.before.__setitem__(j,a[0][:,self.visual].detach().cpu().clone())))
            if i<3:
                self.handles.append(layer.register_forward_hook(
                    lambda m,a,o,j=i:self.after.__setitem__(j,o[:,self.visual].detach().cpu().clone())))
        return self

    def validate(self):
        assert len(self.features)==len(self.after)==len(self.before)==3
        for i in range(3):
            assert self.features[i].shape[0]==len(self.visual)
            assert torch.isfinite(self.features[i]).all() and torch.count_nonzero(self.features[i])>0
            assert torch.equal(self.after[i]+self.features[i].unsqueeze(0),self.before[i+1])

    def __exit__(self,*_):
        for h in self.handles:h.remove()
        self.handles.clear()


def summarize(visual, mapping, repetitions=128):
    assert visual.ndim==3 and visual.shape[:2]==(36,2) and np.isfinite(visual).all()
    n=visual.shape[-1];indices=mapping['roi_token_indices'];k=len(indices)
    assert k and len(set(indices))==k and min(indices)>=0 and max(indices)<n
    rng=np.random.Generator(np.random.PCG64(mapping['random_seed']))
    random=np.stack([rng.choice(n,k,replace=False) for _ in range(repetitions)])
    # Common random patch sets across all layers and both query types, with ROI overlap allowed.
    mass=visual.sum(-1,dtype=np.float64)
    assert (mass>0).all(), 'Zero Image Mass: undefined ROI Hit Ratio, do not silently impute'
    roi=visual[:,:,indices].sum(-1,dtype=np.float64)
    random_mass=visual[:,:,random].sum(-1,dtype=np.float64)
    fraction=k/n
    hit=roi/mass
    return dict(image_mass=mass,roi_mass=roi,roi_hit_ratio=hit,roi_enrichment=hit/fraction,
        random_roi_mass=random_mass.mean(-1),random_roi_hit_ratio=(random_mass/mass[:,:,None]).mean(-1),
        random_roi_enrichment=(random_mass/mass[:,:,None]/fraction).mean(-1),
        random_roi_mass_std=random_mass.std(-1,ddof=1),
        expected_random_roi_mass=mass*fraction,expected_random_roi_hit_ratio=np.full_like(mass,fraction),
        expected_random_roi_enrichment=np.ones_like(mass))
