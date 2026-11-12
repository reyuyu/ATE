from pathlib import Path
import json,hashlib,shutil
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];out=A/'runtime_source_snapshots';out.mkdir(exist_ok=True)
for kind in ['Qwen','InternVL','LLaVA']:
 envpath=A/kind/'screen/environment.json'
 if not envpath.exists():continue
 env=json.loads(envpath.read_text());root=Path(env['transformers_import']).parent
 names=['cache_utils.py','modeling_attn_mask_utils.py','modeling_utils.py','generation/utils.py']
 names+= {'Qwen':['models/qwen3_vl/modeling_qwen3_vl.py'],'InternVL':['models/qwen2/modeling_qwen2.py'],'LLaVA':['models/llava/modeling_llava.py','models/llama/modeling_llama.py']}[kind]
 for name in names:
  p=root/name
  if not p.exists():continue
  dest=out/kind/'transformers'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  rows.append(dict(kind=kind,path=str(p),sha256=sha(p),snapshot=str(dest.relative_to(A)),library='transformers',version=env['transformers']))
model=Path('/home/featurize/ate-local-models/InternVL3-8B');cache=Path('/home/featurize/.cache/huggingface/modules/transformers_modules/InternVL3-8B')
for p in sorted(model.glob('*.py')):
 dest=out/'InternVL'/'model_remote_code'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 row=dict(kind='InternVL',path=str(p),sha256=sha(p),snapshot=str(dest.relative_to(A)),model_revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9')
 cp=cache/p.name
 if cp.exists():row.update(actual_dynamic_cache=str(cp),cache_sha256=sha(cp),cache_equals_model_source=sha(cp)==sha(p));assert row['cache_equals_model_source'],row
 rows.append(row)
(A/'runtime_dependency_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(source_files=len(rows),intern_remote_code=sum('model_revision' in r for r in rows),all_cached_model_sources_match=True)))
