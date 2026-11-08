"""Load pinned official prompt/preprocessing functions without a model."""
from pathlib import Path
from types import SimpleNamespace
import ast,importlib.util,sys
from PIL import Image
import torch
O=Path('/home/featurize/work/ATE/vlm_research/results/featurize_4090_gqa_native_protocol_audit')
D=Path('/home/featurize/data/ate-recovery')
M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
S=O/'official_sources'

def official_functions():
 name='ate_gqa_official_conversation';spec=importlib.util.spec_from_file_location(name,S/'release_conversation.py')
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 source=ast.parse((S/'release_mm_utils.py').read_text())
 nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'expand2square','process_images','tokenizer_image_token'}]
 assert len(nodes)==3
 ns={'Image':Image,'torch':torch,'IMAGE_TOKEN_INDEX':-200};exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_release_mm_utils_subset','exec'),ns)
 return module,SimpleNamespace(**{n.name:ns[n.name] for n in nodes})

def official_prompt(sample,conv):
 c=conv.conv_templates['vicuna_v1'].copy()
 c.append_message(c.roles[0],'<image>\n'+sample['prompt_text']);c.append_message(c.roles[1],None)
 return c.get_prompt()

def official_ids(prompt,tokenizer,functions):
 ids=functions.tokenizer_image_token(prompt,tokenizer)
 assert ids[0]==1 and ids.count(-200)==1
 return [v for t in ids for v in ([32000]*576 if t==-200 else [t])]

def configure(processor,config):
 processor.patch_size=config.vision_config.patch_size;processor.num_additional_image_tokens=1
 processor.vision_feature_select_strategy=config.vision_feature_select_strategy;processor.tokenizer.padding_side='left'
 return processor

def historical_prompt(sample,processor):
 return processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':sample['prompt_text']}]}],tokenize=False,add_generation_prompt=True)

def make_inputs(sample,condition,processor,tokenizer,conv,functions):
 prompt=historical_prompt(sample,processor) if condition in {'HISTORICAL','B'} else official_prompt(sample,conv)
 with Image.open(sample['image_path']) as image:
  image=image.convert('RGB')
  pixels=(functions.process_images([image],processor.image_processor,SimpleNamespace(image_aspect_ratio='pad'))
          if condition in {'B','C'} else processor.image_processor(image,return_tensors='pt')['pixel_values'])
 if condition in {'HISTORICAL','B'}:
  compact=processor.tokenizer(prompt)['input_ids'];assert compact.count(32000)==1
  ids=[v for t in compact for v in ([32000]*576 if t==32000 else [t])]
 else:ids=official_ids(prompt,tokenizer,functions)
 return {'input_ids':torch.tensor([ids],dtype=torch.long),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long),'pixel_values':pixels},prompt
