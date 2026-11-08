"""HF 4.49 bridge for the pinned official V2Drop selection block.

No ATE or Transformers file is modified. Original ranking and retention execute
from the official AST, with only its four fixed image-start constants replaced.
The frozen scope is one RGB image, batch 1, eager attention, ordinary DynamicCache.
"""
import ast
import hashlib
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch

OFFICIAL_SHA='e39ed8d99c3e1f23bd156067bfd28179827771df3556631b6daed4427a7b8b61'


@lru_cache(maxsize=1)
def official_selector(path):
    raw=Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==OFFICIAL_SHA
    tree=ast.parse(raw)
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If)
            and 'decoder_layer.self_attn.layer_idx in target_layer_indices' in ast.unparse(n.test)]
    assert len(blocks)==1
    nodes=[]
    for node in blocks[0].body:
        nodes.append(node)
        if isinstance(node,ast.Assign) and ast.unparse(node)=='keep_indexs = keep_indexs.sort().values':break
    assert ast.unparse(nodes[-1])=='keep_indexs = keep_indexs.sort().values'
    original=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    class Start(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value==35:
                self.count+=1
                return ast.copy_location(ast.Name(id='image_start',ctx=ast.Load()),node)
            return node
    change=Start();module=change.visit(ast.parse(original));assert change.count==4
    prefix='''def select(current_hidden_states, previous, image_start, amount, layer_idx):
    self=SimpleNamespace(prev_hidden_states=previous,image_token_amount_after_selection=amount)
    decoder_layer=SimpleNamespace(self_attn=SimpleNamespace(layer_idx=layer_idx))
    hidden_states=current_hidden_states
'''
    def build(body):
        source=prefix+'\n'.join('    '+line for line in body.splitlines())+'\n    return keep_indexs, selected_indices, l2_diff_candidate\n'
        namespace={'torch':torch,'SimpleNamespace':SimpleNamespace};exec(compile(source,str(path),'exec'),namespace)
        return namespace['select']
    return build(ast.unparse(module)),build(original),{'source_sha256':OFFICIAL_SHA,
        'original_selection_AST_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'only_AST_changes':'Four literal image-start constants 35 replaced with image_start',
        'pruning_layers':[3,17,22],'retention_ratios':[0.5,0.5,0.0],'minimum_retained':1}


class V2Drop:
    def __init__(self,adapter,image_positions,source,enabled=True,trace=False,reference=False,selection_timing=False):
        self.a=adapter;self.image=tuple(image_positions);assert len(self.image)==576
        self.start=self.image[0];assert self.image==tuple(range(self.start,self.start+576))
        self.select,self.fixed_selector,self.provenance=official_selector(source)
        self.enabled=enabled;self.trace=trace;self.reference=reference;self.selection_timing=selection_timing
        self.verify=trace is True;self.decode_index=0
        self.previous=None;self.active=None;self.amount=576;self.events=[];self.layers=[];self.selection_events=[]
        self.original_n=None;self.kv_positions={};self.prefill_rows={}

    def before(self,index,module,args,kw):
        h=args[0];prefill=h.shape[1]>1
        if index==0:
            self.previous=None
            if not prefill:self.decode_index+=1
            if prefill:
                self.original_n=h.shape[1];self.active=torch.arange(self.original_n,device=h.device);self.amount=576
        if prefill:
            if self.enabled and index in (3,17,22):
                begin=end=None
                if self.selection_timing:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                before=self.amount
                keep,selected,scores=self.select(h,self.previous,self.start,self.amount,index)
                if self.reference:
                    # Restore the official fixed-start layout solely for checking
                    # its unmodified kernel against the actual adapted selection.
                    pad=h.new_zeros((1,35,h.shape[-1]));suffix=h[:,self.start:]
                    current=torch.cat([pad,suffix],dim=1)
                    previous=torch.cat([pad,self.previous[:,self.start:]],dim=1)
                    rk,rs,rv=self.fixed_selector(current,previous,35,self.amount,index)
                    assert torch.equal(rs-35,selected-self.start) and torch.equal(rv,scores)
                h=h.index_select(1,keep);self.active=self.active.index_select(0,keep);self.amount=selected.numel()
                if self.selection_timing:
                    end.record();self.selection_events.append((index,begin,end))
                event={'layer':index,'visual_before':before,'visual_after':self.amount,'physical_hidden_rows':h.shape[1]}
                if self.trace:
                    event.update(kept_original_positions=self.active.tolist(),selected_original_image_positions=self.active[self.start:self.start+self.amount].tolist(),
                                 official_fixed35_reference_exact=self.reference)
                self.events.append(event)
            if self.amount!=576:
                active=self.active
                kw=dict(kw)
                for key in ('position_ids','cache_position'):
                    assert kw[key].shape[-1]==self.original_n
                    kw[key]=kw[key].index_select(-1,active)
                kw['position_embeddings']=tuple(x.index_select(1,active) for x in kw['position_embeddings'])
                mask=kw.get('attention_mask')
                if mask is not None:
                    assert mask.ndim==4 and mask.shape[-2]==self.original_n
                    kw['attention_mask']=mask.index_select(-2,active).index_select(-1,active)
            self.previous=h
            self.prefill_rows[index]=h.shape[1]
            if self.trace:
                positions=kw['cache_position'].tolist();assert positions==self.active.tolist()
                self.kv_positions[index]=positions
            if self.verify:
                assert [p for p in positions if p not in self.image]==[p for p in range(self.original_n) if p not in self.image]
                assert torch.equal(kw['position_ids'][0],kw['cache_position'])
                expected=self.a.model.language_model.model.rotary_emb(h,kw['position_ids'])
                assert all(torch.equal(x,y) for x,y in zip(expected,kw['position_embeddings']))
                mask=kw['attention_mask'][0,0];expected_mask=torch.triu(torch.ones_like(mask,dtype=torch.bool),diagonal=1)
                assert torch.equal(mask<0,expected_mask),'Selected causal mask differs'
        else:
            self.previous=h
            if self.trace:
                self.kv_positions[index].append(self.original_n+self.decode_index-1)
            if self.verify:
                cache=kw['past_key_value'];size=cache.get_seq_length(index)+1
                mask=kw.get('attention_mask')
                assert mask is None or torch.count_nonzero(mask[...,:size]).item()==0
                assert self.kv_positions[index][-1]==kw['cache_position'].item()
        if self.trace:self.layers.append({'layer':index,'prefill':prefill,'hidden_rows':h.shape[1],'visual_hidden':self.amount if prefill else 0})
        return (h,*args[1:]),kw

    @contextmanager
    def context(self):
        handles=[]
        try:
            for index,layer in enumerate(self.a.layers):
                handles.append(layer.register_forward_pre_hook(lambda m,args,kw,i=index:self.before(i,m,args,kw),with_kwargs=True))
            yield self
        finally:
            for handle in handles:handle.remove()
            self.previous=None

    def timing(self):
        return [{'layer':layer,'selection_and_hidden_gather_gpu_ms':a.elapsed_time(b)} for layer,a,b in self.selection_events]
