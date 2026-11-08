"""Fetch only small primary-source protocol files; no weights or datasets."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib,subprocess
HERE=Path(__file__).resolve().parent;DEST=HERE/'official_sources';DEST.mkdir(exist_ok=True)
REL='ce1aa08d129bd7df931a7067bcb79cf3cb3a1af5';MAIN='c121f0432da27facab705978f83c4ada465e46fd';REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
sources={
 'release_conversation.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/conversation.py',
 'release_mm_utils.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/mm_utils.py',
 'release_model_vqa_loader.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/eval/model_vqa_loader.py',
 'release_gqa.sh':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/scripts/v1_5/eval/gqa.sh',
 'release_convert_gqa.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/scripts/convert_gqa_for_eval.py',
 'release_builder.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/model/builder.py',
 'release_clip_encoder.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/model/multimodal_encoder/clip_encoder.py',
 'release_llava_arch.py':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/llava/model/llava_arch.py',
 'MODEL_ZOO.md':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{MAIN}/docs/MODEL_ZOO.md',
 'Evaluation.md':f'https://raw.githubusercontent.com/haotian-liu/LLaVA/{REL}/docs/Evaluation.md',
 'HF_model_card.md':f'https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/{REV}/README.md',
 'HF_conversion_reference.py':'https://raw.githubusercontent.com/huggingface/transformers/19d179f2165fc02dbc7c45456d794a3f78d7e310/src/transformers/models/llava/convert_llava_weights_to_hf.py',
 'CLIP_processor.json':'https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/preprocessor_config.json',
 'GQA_eval_llava.py':'https://gist.githubusercontent.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187/raw',
 'original_model_api.json':'https://huggingface.co/api/models/liuhaotian/llava-v1.5-7b?blobs=true',
}
def fetch(item):
 name,url=item;p=DEST/name
 if p.exists():raw=p.read_bytes()
 else:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ATE-GQA-protocol-audit'}),timeout=40) as response:raw=response.read(2*1024**2+1)
  assert len(raw)<=2*1024**2,'Reject unexpected large source';p.write_bytes(raw)
 return name,{'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool:manifest=dict(pool.map(fetch,sources.items()))
meta=json.loads((DEST/'original_model_api.json').read_text(encoding='utf-8'));pin=meta['sha']
for name in ['config.json','pytorch_model.bin.index.json','tokenizer_config.json','special_tokens_map.json']:
 key='original_'+name
 _,info=fetch((key,f'https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/{pin}/{name}'));manifest[key]=info
(DEST/'source_manifest.json').write_text(json.dumps({'llava_release_commit':REL,'llava_current_commit':MAIN,'original_model_revision':pin,'sources':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'sources':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'original_model_revision':pin}),flush=True)
