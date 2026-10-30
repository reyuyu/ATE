"""Online features accept only consumed prefix and pre-Attention hidden state."""
import math
import numpy as np
import torch

PREFIX_DIM=1284


def prefix_features(consumed):
    t=len(consumed)
    if t<1:raise ValueError('Only true Decode, never Prefill')
    out=np.zeros(PREFIX_DIM,dtype=np.float32)
    for lag in range(3):
        token=consumed[-1-lag] if len(consumed)>lag else 0
        out[lag*256+(int(token)*2654435761)%256]=1
    a=consumed[-1];b=consumed[-2] if t>1 else 0;c=consumed[-3] if t>2 else 0
    out[768+((b*65599+a)*2654435761)%256]=1
    out[1024+((c*65599+b)*65599+a)%256]=1
    out[1280:]=[t/64.,math.log1p(t),float(t==1),float(t>=32)]
    return out


def hidden_features(hidden,t):
    h=torch.as_tensor(hidden).float();norm=h.norm()
    if h.shape!=(4096,) or not torch.isfinite(h).all() or norm<=0:raise ValueError('Invalid real h31_in')
    return torch.cat([h/norm,torch.tensor([math.log1p(float(norm)),t/64.],device=h.device)])


def category(annotation,t):
    row=annotation['tokens'][t]
    if row['category'] in ['EOS','Special']:return 'EOS-Special'
    words=[w for w in annotation['words'] if t in w['token_steps']]
    if len(words)!=1:return row['category']
    w=words[0]
    if not w['exact_complete_single_token'] or w['token_length']!=1:return 'Multi-token-Unaligned'
    return w['word_type']


class OnlinePredictor:
    def __init__(self,parameters,kind,threshold,device='cuda:0'):
        self.kind=kind;self.threshold=threshold
        self.mean=torch.as_tensor(parameters[kind+'_mean'],device=device)
        self.scale=torch.as_tensor(parameters[kind+'_scale'],device=device)
        self.weight=torch.as_tensor(parameters[kind+'_weight'],device=device)
        self.bias=torch.as_tensor(parameters[kind+'_bias'],device=device)

    def probability(self,consumed,hidden):
        features=[]
        if self.kind in ['A','C']:features.append(torch.as_tensor(prefix_features(consumed),device=self.weight.device))
        if self.kind in ['B','C']:features.append(hidden_features(hidden,len(consumed)).to(self.weight.device))
        x=torch.cat(features)
        return torch.sigmoid(((x-self.mean)/self.scale)@self.weight+self.bias)

    def decide(self,consumed,hidden):return float(self.probability(consumed,hidden))>=self.threshold
