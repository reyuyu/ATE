"""Reuse frozen candidate execution; add a fully dense, validation-only reference."""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from .common import *
from ..accuracy_autoresearch.engine import generate as inherited_generate, ResearchWriter
from ..physical_handoff_fastpath import engine as base


def prepare(adapter,sample,label):
    adapter.prepare(sample,{})
    m=sample['input_mapping']; ids=adapter.inputs['input_ids'][0].tolist()
    assert ids==m['prompt_ids']
    assert [i for i,v in enumerate(ids) if v==32000]==m['image']
    layout=base.Layout(len(ids),tuple(m['image']),tuple(i for i,v in enumerate(ids) if v!=32000),
                       tuple(m['R']),tuple(m['parts']['Q']),20)
    c=CONFIGS[label]
    if c['writer'] in ('native','all'): writers=list(layout.relay)
    elif c['writer']=='qsa': writers=sorted(set(layout.question)|set(m['parts']['ASSISTANT']))
    else: writers=conditional_writers(sample['question'],layout.question,m['parts']['ASSISTANT'])[0]
    assert set(writers)<=set(layout.relay) and layout.length-1 not in writers
    if c['writer'] in ('qsa','conditional15'):
        assert set(writers).isdisjoint(m['parts']['Hint']+m['parts']['Options'])
    return layout,writers,dict(input_hashes=adapter.input_hashes,prompt_ids=ids,Q=list(layout.question),
        ASSISTANT=m['parts']['ASSISTANT'],Hint=m['parts']['Hint'],Options=m['parts']['Options'],
        writers=writers,R=list(layout.relay),Pminus1=layout.length-1)


class DenseReference(ResearchWriter):
    def __init__(self,a,l,c,w,length,audit):
        super().__init__(a,l,c,w,length,audit,True)
        self.physical=False; self.condition='D1 Dense'
        self.cache=base.InplaceKV(len(a.layers),l,'D1 Dense',length,self.device)
        late=torch.full((l.length,l.length),torch.finfo(self.dtype).min,device=self.device,dtype=self.dtype).triu(1)
        blocked=torch.tensor((*l.relay,l.length-1),device=self.device)
        image=torch.tensor(l.image,device=self.device)
        late[blocked[:,None],image[None,:]]=-torch.inf
        self.prefill_late=late[None,None]


def generate(adapter,layout,label,writers,length=32,capture=False,audit=False,reference=False):
    c=CONFIGS[label]
    if not reference or label=='Native Full':
        return inherited_generate(adapter,layout,c,writers,length,capture,audit,reference)
    original=adapter.model.forward
    def head(*args,**kw):
        kw['logits_to_keep']=1
        return original(*args,**kw)
    def factory(a,l,condition,max_tokens,collect):
        return DenseReference(a,l,c,writers,max_tokens,collect)
    with patch.object(adapter.model,'forward',head),patch.object(base,'FastHandoff',factory):
        return base.generate(adapter,layout,'Fast-P2',length,False,capture,audit)
