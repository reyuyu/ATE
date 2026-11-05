import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
exec(open('/data/lk/ATE-resources/experiments/textvqa_decode_l0_l1_factorial/prepare_data.py').read().split('def main():')[0])
import sys
sys.path.insert(0,str(R))
raw=read(ANN)['data'];byq={str(x['question_id']):x for x in raw};grouped=collections.defaultdict(list)
for x in raw:grouped[x['image_id']].append(x)
inventory=read(O/'historical_image_inventory.json')
write(O/'historical_candidate_inventory.json',inventory)
sources=[x for x in inventory['sources'] if 'selection_plan' not in Path(x['path']).name]
blocked_ids=set().union(*(set(x['TextVQA_image_ids']) for x in sources));blocked_sha=set().union(*(set(x['image_sha256']) for x in sources))
paper=Path('/data/lk/ATE-worktrees/paper-rq1-attention-time/vlm_research/results/paper_rq1_attention_time')
cp_ident=[];pc=collections.Counter()
for stage in ['discovery','confirmation']:
 for p in sorted((paper/stage).glob('shard*/checkpoints/*.json')):
  stem=p.stem;iid=None
  if stem.startswith('textvqa__'):iid=byq[stem.split('__')[1]]['image_id'];pc[stage+'_textvqa']+=1
  if stem.startswith('textcaps__caption_'):iid=stem[len('textcaps__caption_'):];pc[stage+'_textcaps']+=1
  if iid in grouped:
   blocked_ids.add(iid);cp_ident.append(dict(path=str(p),image_id=iid,stage=stage,identity_source='checkpoint filename, no prediction values read'))
cache=Path('/data/lk/ATE-resources/paper_rq1/images/openimages')
for iid in blocked_ids:
 p=cache/(iid+'.jpg')
 if p.exists():blocked_sha.add(sha(p))
write(O/'historical_image_inventory.json',dict(sources=sources,actual_paper_checkpoint_identities=cp_ident,actual_paper_counts=dict(pc),excluded_image_ids=sorted(blocked_ids),excluded_image_sha256=sorted(blocked_sha),scope='All discovered frozen input manifests except unexecuted candidate plans, plus actual paper checkpoint identities. Candidate plans separately preserved; no accuracy used.'))
ids=sorted(set(grouped)-blocked_ids,key=key);chosen=[];missing=[];paths={}
for iid in ids:
 p=cache/(iid+'.jpg')
 if not p.exists():p=P/'images'/(iid+'.jpg')
 paths[iid]=p;chosen.append(iid)
 if not p.exists():missing.append(dict(image_identity='openimages:'+iid,image_path=str(p)))
 if sum(len(grouped[z]) for z in chosen)>=800:break
print('FROZEN_INPUT_CANDIDATES',len(chosen),'questions',sum(len(grouped[z]) for z in chosen),'downloads',len(missing),flush=True)
receipts=[]
if missing:
 from paper_rq1.data import fetch_openimages
 receipts=fetch_openimages(missing)
selected=[];attempts=[];seen={}
from PIL import Image
for iid in chosen:
 p=paths[iid];h=sha(p)
 if h in blocked_sha:attempts.append(dict(image_id=iid,status='SHA_OVERLAP'));continue
 if h in seen:print('EXACT_ALIAS',iid,seen[h],flush=True)
 seen[h]=iid
 with Image.open(p) as im:im.load();size=list(im.size)
 for x in sorted(grouped[iid],key=lambda x:str(x['question_id'])):
  selected.append(dict(question_id=str(x['question_id']),image_id=iid,question=x['question'],answers=x['answers'],image_path=str(p),image_sha256=h,image_size=size,cohort='new_images',set_name='val'))
 attempts.append(dict(image_id=iid,status='WHOLE_IMAGE_GROUP',questions=len(grouped[iid]),sha256=h))
old=read(HO/'sample_manifest.json')['rows']
assert len(selected)>=800 and not {x['image_sha256'] for x in selected}&blocked_sha
write(P/'data_selection.json',dict(new_rows=selected,old_rows=old,attempts=attempts,download_receipts=receipts,annotation_path=str(ANN),annotation_sha256=sha(ANN)))
write(O/'image_overlap_audit.json',dict(passed=True,official_questions=5000,official_images=len(grouped),historical_image_IDs=len(blocked_ids),new_questions=len(selected),new_images=len(seen),ID_overlap=0,SHA_overlap=0,all_questions_of_selected_image_IDs_included=True,selection_rule='SHA256(20261105|textvqa_decode_l0_l1_factorial|image_id), whole image groups until >=800 questions; no predictions accessed',candidate_plan_handling='Only actual checkpoint identities excluded for paper candidate pools; frozen non-candidate manifests conservatively excluded. Full preliminary candidate union separately archived.',annotation_sha256=sha(ANN),limitations='Exact bytes and original image IDs; re-encoded near duplicates and unavailable external histories cannot be ruled out. Same public validation source, not an external dataset.'))
print('DATA_SELECTION_COMPLETE',len(selected),len(seen),flush=True)
