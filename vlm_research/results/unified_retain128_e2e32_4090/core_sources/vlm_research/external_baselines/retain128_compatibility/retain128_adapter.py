"""Retain-128 parameters around pinned official kernels; old adapters untouched."""
import ast,copy,hashlib
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace,MethodType
import torch
from e2e32_official_compat import OfficialCompat as BaseCompat,ROOT
from hf_adapter import V2Drop as BaseV2Drop,OFFICIAL_SHA

def sparse128(path,namespace):
 tree=ast.parse(Path(path).read_text())
 owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlamaDynamicvitModel')
 forward=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 branch=next(n for n in ast.walk(forward) if isinstance(n,ast.If) and 'layer_idx in self.pruning_loc' in ast.unparse(n.test))
 inference=next(n for n in branch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.training').orelse
 first=next(i for i,n in enumerate(inference) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('attn_logits = layer_outputs[2]'))
 body=copy.deepcopy(inference[first:])
 wrapper=ast.parse('''def official_sparse128(hidden_states, attn, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx, position_ids):
 B=1
 retained_tokens=128
 pre_prompt_length_list=[v_token_start]
 image_shape=576
 layer_outputs=(hidden_states,None,attn)
''').body[0]
 wrapper.body+=body+ast.parse('return layer_outputs[0],v_token_num,position_ids').body
 module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));exec(compile(module,str(path),'exec'),namespace)
 return namespace['official_sparse128']

@lru_cache(maxsize=1)
def v2_selector128(path):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==OFFICIAL_SHA
 tree=ast.parse(raw);block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'decoder_layer.self_attn.layer_idx in target_layer_indices' in ast.unparse(n.test))
 nodes=[]
 for n in block.body:
  nodes.append(copy.deepcopy(n))
  if isinstance(n,ast.Assign) and ast.unparse(n)=='keep_indexs = keep_indexs.sort().values':break
 changes=[]
 for n in nodes:
  if isinstance(n,ast.If) and ast.unparse(n.test) in ('current_layer == 3','current_layer == 17'):
   assert len(n.body)==1 and ast.unparse(n.body[0])=='keep_ratio = 0.5'
   value=.28 if ast.unparse(n.test)=='current_layer == 3' else .25
   n.body[0].value=ast.copy_location(ast.Constant(value=value),n.body[0].value);changes.append(value)
 assert changes==[.28,.25]
 original=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
 class Start(ast.NodeTransformer):
  count=0
  def visit_Constant(self,node):
   if node.value==35:self.count+=1;return ast.copy_location(ast.Name(id='image_start',ctx=ast.Load()),node)
   return node
 change=Start();module=change.visit(ast.parse(original));assert change.count==4
 prefix='''def select(current_hidden_states, previous, image_start, amount, layer_idx):
 self=SimpleNamespace(prev_hidden_states=previous,image_token_amount_after_selection=amount)
 decoder_layer=SimpleNamespace(self_attn=SimpleNamespace(layer_idx=layer_idx))
 hidden_states=current_hidden_states
'''
 def build(body):
  source=prefix+'\n'.join(' '+line for line in body.splitlines())+'\n return keep_indexs, selected_indices, l2_diff_candidate\n'
  ns={'torch':torch,'SimpleNamespace':SimpleNamespace};exec(compile(source,str(path),'exec'),ns);return ns['select']
 return build(ast.unparse(module)),build(original),dict(source_sha256=OFFICIAL_SHA,pruning_layers=[3,17,22],retention_ratios=[.28,.25,0],minimum_retained=1,
  parameter_source='V2Drop arXiv:2509.01552v2 Appendix A: remove72%,75%,100%; original selector ranking/topk/floor/min1 unchanged',
  only_changes='Official paper128 ratios replace source demo .5/.5; four fixed image-start35 constants use actual image_start')

class V2Drop(BaseV2Drop):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.select,self.fixed_selector,self.provenance=v2_selector128(args[2])

class OfficialCompat(BaseCompat):
 def __init__(self,a,layout,method,trace=False):
  super().__init__(a,layout,method,trace)
  if method=='ApET':
   self.initial_visual=192;self.stage_visual={16:64}
   self.proxy.image_tokens=[192];self.proxy.image_token_list=[192,64]
  elif method=='DUET':
   self.initial_visual=205;self.stage_visual={16:102,24:0};self.proxy.image_tokens=[205]
  elif method=='SparseVLM':
   self.namespace=dict(self.namespace)
   score=ROOT/'sparsevlm_icml2025/official_source/llava/model/language_model/score.py'
   ns={}
   for n in ast.parse(score.read_text()).body:
    if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),str(score),'exec'),ns)
   assert ns['sparse_token_dict'][128]==[303,110,36]
   self.namespace.update(layer_dict=ns['layer_dict'],sparse_token_dict=ns['sparse_token_dict'])
   self.load(score,'attn_postprocess_topk')
   self.prune=sparse128(score.with_name('modelling_sparse_llama.py'),self.namespace)
   self.initial_visual=576;self.stage_visual={3:312,7:117,16:39}
 def vision(self,pixel_values,*args,**kwargs):
  if self.method in ('BridgeNative','SparseVLM'):return super().vision(pixel_values,*args,**kwargs)
  self.vision_calls+=1
  if self.method=='ApET':
   def tower(images):
    out=self.model.vision_tower(images,output_hidden_states=True);return out.hidden_states[-2],None
   proxy_model=SimpleNamespace(basis_token_num=10,mm_projector=self.model.multi_modal_projector,get_vision_tower=lambda:tower)
   proxy=SimpleNamespace(model=proxy_model,get_model=lambda:proxy_model,get_visual_token_num=lambda:192)
   proxy.token_merging=MethodType(self.vision_merge,proxy)
   full,mask=self.vision_encode(proxy,pixel_values);assert full.shape[1]==576 and mask.shape==(1,576)
   return full[mask].view(1,192,full.shape[-1])
  # Base context shares this dictionary with every official CLIP layer.
  self.vision_info.update(dominant=169,contextual=35,cluster_width=4)
  out,indices=self.vision_select(SimpleNamespace(vision_tower=self.model.vision_tower,device=pixel_values.device,dtype=pixel_values.dtype),pixel_values)
  assert out.shape[1]==205 and indices.shape[1]==170
  self.dominant_indices=indices
  return self.model.multi_modal_projector(out)
