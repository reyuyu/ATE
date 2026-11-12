"""Reuse the frozen C3-V2 physical path; vary only its receiver/image mask."""
from unittest.mock import patch
import torch

from .common import CFG, writer_set
from ..accuracy_autoresearch.engine import generate as inherited_generate
from ..closedqa_transfer.engine import DenseReference
from ..physical_handoff_fastpath import engine as base


def prepare(adapter,sample,label):
    adapter.prepare(sample,{})
    m=sample['input_mapping']; ids=adapter.inputs['input_ids'][0].tolist()
    assert ids==m['prompt_ids'] and [i for i,v in enumerate(ids) if v==32000]==m['image']
    layout=base.Layout(len(ids),tuple(m['image']),tuple(i for i,v in enumerate(ids) if v!=32000),tuple(m['R']),tuple(m['parts']['Q']),20)
    writers=writer_set(m,label)
    spec=dict(input_hashes=adapter.input_hashes,prompt_ids=ids,Q=m['parts']['Q'],
        ASSISTANT=m['parts']['ASSISTANT'],Hint=m['parts']['Hint'],Options=m['parts']['Options'],
        writers=writers,R=m['R'],Pminus1=m['Pminus1'])
    return layout,writers,spec


def generate(adapter,layout,writers,length=32,capture=False,audit=False,dense=False):
    if not dense:
        return inherited_generate(adapter,layout,CFG,writers,length,capture,audit)
    original=adapter.model.forward
    def head(*args,**kw):
        kw['logits_to_keep']=1
        return original(*args,**kw)
    def factory(a,l,condition,max_tokens,collect):
        return DenseReference(a,l,dict(CFG,writer='qsa'),writers,max_tokens,collect)
    with patch.object(adapter.model,'forward',head),patch.object(base,'FastHandoff',factory):
        return base.generate(adapter,layout,'Fast-P2',length,False,capture,audit)
