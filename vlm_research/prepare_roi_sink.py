"""CPU-only 240-image true-Processor ROI audit and outcome-independent main selection."""
import csv,json,random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoProcessor
from vlm_common import ROOT,default_model
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from fixed_k_exit import atomic_json,verify_manifest
from summarize_fixed_k import write_csv
from roi_sink_common import OUTPUT,PROTOCOL,SEED,geometry,map_box


def eligible(row):
    a,b=row['objects'];counts=[a['patch_count'],b['patch_count']];reasons=[]
    if min(counts)==0:reasons.append('one_or_both_objects_cropped_out')
    if min(counts)<2:reasons.append('object_fewer_than_two_patches')
    if not 4<=len(row['roi_indices'])<=192:reasons.append('union_outside_4_to_192')
    if min(counts)>0 and max(counts)/min(counts)>16:reasons.append('object_patch_area_ratio_over_16')
    return reasons


def overlay(sample,out):
    image=Image.open(sample['image_path']).convert('RGB');ip=AutoProcessor.from_pretrained(default_model(),local_files_only=True).image_processor
    _,crop=geometry(image,ip);fig,axes=plt.subplots(1,3,figsize=(13,4.7));axes[0].imshow(image);axes[1].imshow(crop);axes[2].imshow(crop)
    for j,obj in enumerate(sample['objects']):
        color=['cyan','magenta'][j];x,y,w,h=obj['original_xywh'];axes[0].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,lw=2))
        x0,y0,x1,y1=obj['crop_xyxy'];axes[1].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=2))
        for index in obj['patch_indices']:
            r,c=divmod(index,24);axes[2].add_patch(Rectangle((c*14,r*14),14,14,color=color,alpha=.22))
    for t in range(0,337,14):axes[2].axhline(t,color='white',alpha=.2,lw=.3);axes[2].axvline(t,color='white',alpha=.2,lw=.3)
    for ax,title in zip(axes,['Original official boxes','Actual CLIP resize + center crop','24x24 row-major ROI patches']):ax.set_title(title);ax.axis('off')
    fig.suptitle(f"{sample['question_id']}: {sample['question']}\nVisible fractions {[round(o['visible_fraction'],3) for o in sample['objects']]}; patches {[o['patch_count'] for o in sample['objects']]}",fontsize=10)
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main():
    if OUTPUT.exists():raise ValueError('Prepared output exists: never resample or overwrite')
    oldpath=ROOT/'results/phase3_2_fixed_k_exit/manifest.json';old=json.loads(oldpath.read_text());verify_manifest(old)
    source=ROOT/'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv'
    if sha256(source)!=old['source_csv_sha256']:raise ValueError('CSV changed')
    lookup={r['question_id']:r for r in csv.DictReader(source.open())}
    OUTPUT.mkdir(parents=True);atomic_json(OUTPUT/'preregistered_protocol.json',PROTOCOL)
    ip=AutoProcessor.from_pretrained(default_model(),local_files_only=True).image_processor
    prior=list(csv.DictReader((ROOT/'results/phase3_2_fixed_k_exit/per_sample_results.csv').open()))
    cases=[r['question_id'] for r in prior if r['cutoff']=='24' and r['full_correct']=='False' and r['candidate_correct']=='True']+['0216546','02310173','05480434']
    if len(set(cases))!=5:raise ValueError('Expected fixed two K24 gains and three K16 losses')
    audited=[];flat=[]
    for s in old['samples']:
        r=lookup[s['question_id']]
        if digest(r)!=s['source_row_sha256']:raise ValueError('Official source row changed')
        im=Image.open(s['image_path']).convert('RGB');g,_=geometry(im,ip)
        if list(im.size)!=s['source_image_size']:raise ValueError('Dimensions mismatch')
        objects=[]
        for j in (1,2):
            o=map_box([r[f'object{j} {key}'] for key in ('x','y','w','h')],g)
            o.update(name=r[f'object{j} name'],object_id=r[f'object{j} id']);objects.append(o)
        roi=sorted(set(objects[0]['patch_indices'])|set(objects[1]['patch_indices']))
        item=dict(s,geometry=g,objects=objects,roi_indices=roi,background_indices=sorted(set(range(576))-set(roi)))
        item['geometry_exclusion_reasons']=eligible(item);item['historical_case']=s['question_id'] in cases
        item['main_eligible']=not item['geometry_exclusion_reasons'] and not item['historical_case'];audited.append(item)
        flat.append(dict(question_id=s['question_id'],img_id=s['img_id'],answer=s['answer'],object1=objects[0]['name'],object2=objects[1]['name'],
                         object1_visible_fraction=objects[0]['visible_fraction'],object2_visible_fraction=objects[1]['visible_fraction'],
                         object1_patches=objects[0]['patch_count'],object2_patches=objects[1]['patch_count'],roi_patches=len(roi),background_patches=576-len(roi),
                         main_eligible=item['main_eligible'],historical_case=item['historical_case'],exclusion_reasons=';'.join(item['geometry_exclusion_reasons'])))
    rng=random.Random(SEED);selected=[]
    for label in ('left','right'):
        pool=sorted([r for r in audited if r['answer']==label and r['main_eligible']],key=lambda s:s['question_id']);rng.shuffle(pool)
        if len(pool)<24:raise ValueError('Not enough geometry-eligible images')
        selected+=pool[:24]
    counts={str(k):sum(sum(o['patch_count']>0 for o in r['objects'])==k for r in audited) for k in [0,1,2]}
    manifest=dict(phase='3.4',parent_manifest_sha256=sha256(oldpath),protocol_sha256=digest(PROTOCOL),processor_config=ip.to_dict(),
                  main_question_ids=[s['question_id'] for s in selected],case_question_ids=cases,samples=audited)
    atomic_json(OUTPUT/'manifest.json',manifest);write_csv(OUTPUT/'roi_mapping_audit.csv',flat)
    atomic_json(OUTPUT/'roi_mapping_audit.json',dict(visible_object_count=counts,all_processor_pixels_exact=True,total=240,
       geometry_exclusions=sum(bool(r['geometry_exclusion_reasons']) for r in audited),
       exclusion_reason_counts={reason:sum(reason in r['geometry_exclusion_reasons'] for r in audited) for reason in sorted({x for r in audited for x in r['geometry_exclusion_reasons']})},
       eligible_by_label={label:sum(r['main_eligible'] and r['answer']==label for r in audited) for label in ['left','right']},main_n=48,cases=cases))
    sorted_main=sorted(selected,key=lambda r:len(r['roi_indices']));examples=[sorted_main[i] for i in np.linspace(0,47,12,dtype=int)]
    for num in (0,1):
        extra=next((r for r in audited if sum(o['patch_count']>0 for o in r['objects'])==num),None)
        if extra:examples.append(extra)
    folder=OUTPUT/'roi_overlays';folder.mkdir()
    for s in examples:overlay(s,folder/f"{s['question_id']}.png")
    atomic_json(folder/'index.json',[dict(question_id=s['question_id'],main=s['question_id'] in manifest['main_question_ids']) for s in examples])
    print(json.dumps(json.loads((OUTPUT/'roi_mapping_audit.json').read_text()),indent=2))


if __name__=='__main__':main()
