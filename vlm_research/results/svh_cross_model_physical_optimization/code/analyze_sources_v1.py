from pathlib import Path
import json,hashlib,difflib,collections
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
W=Path('/home/featurize/ate-experiments/rtx3090-18conditions-20261113')
R=Path('/home/featurize/work/ATE')
O=W/'vlm_research/results/rtx3090_unified_efficiency_18conditions'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
changes=[];(A/'source_diffs').mkdir(exist_ok=True)
for ns in ['opt_b','opt_c','opt_d','opt_e']:
 for p in sorted((A/ns).rglob('*.py')):
  rel=p.relative_to(A/ns)
  if rel.as_posix()=='prefix_fusion.py':old=R/'vlm_research/results/v2drop_prefix_svh_accuracy_3090/fusion_engine.py'
  elif rel.as_posix()=='apet_fusion.py':old=O/'code/apet_fusion.py'
  else:old=W/'vlm_research'/rel
  if not old.exists():
   alt=R/'vlm_research'/rel
   if alt.exists():old=alt
  if old.exists():
   oldhash=sha(old)
   if oldhash==sha(p):continue
   before=old.read_text();after=p.read_text()
   text=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(old),tofile=str(p)))
   diff=A/'source_diffs'/(ns+'__'+rel.as_posix().replace('/','__')+'.diff');diff.write_text(text)
   snap=A/'frozen_sources'/str(old).lstrip('/')
   if not snap.exists():snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(old.read_bytes())
   assert sha(snap)==oldhash
   changes.append(dict(namespace=ns,optimized_path=str(p),optimized_sha256=sha(p),frozen_path=str(old),frozen_sha256=oldhash,frozen_snapshot=str(snap.relative_to(A)),diff=str(diff.relative_to(A))))
  elif p.read_text().strip():
   changes.append(dict(namespace=ns,optimized_path=str(p),optimized_sha256=sha(p),frozen_path=None,note='New helper or namespace entry; see full independent package'))
save(A/'changed_source_manifest.json',changes)
models=[]
for name in ['llava-1.5-7b-hf','Qwen3-VL-8B-Instruct','InternVL3-8B']:
 p=Path('/home/featurize/ate-local-models')/name/'.ate_revision.json'
 models.append(dict(path=str(p.parent),receipt=str(p),receipt_sha256=sha(p),receipt_content=json.loads(p.read_text())))
save(A/'model_manifest.json',models)
print(json.dumps(dict(changed_source_files=len(changes),by_namespace=dict(collections.Counter(c['namespace'] for c in changes)),models=len(models))))
