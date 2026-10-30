import json
import subprocess
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from attention_position_probe import locate_question_tokens
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.protocols import prompt_text
from gqa6_update_decomposition.run import verify as verify_updates
from .protocol import ROOT,OUT,RAW,OLD,UPDATES,PROTOCOL,SEED,TASKS,ANNOTATION,IMAGES,MODEL_PATH


def split(image_sha):
    return 'train' if int(digest([SEED,'bos-discovery-split-v1',image_sha])[:12],16)%100<70 else 'calibration'


def mapping(sample,processor):
    path=Path(sample['image_path'])
    with Image.open(path) as original:original.load();image=original.convert('RGB');size=list(image.size)
    sample.update(image_sha256=sha256(path),image_size=size)
    sample['prompt_text']=prompt_text(dict(dataset='gqa_balanced',question=sample['question']))
    chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=sample['prompt_text'])])],tokenize=False,add_generation_prompt=True)
    native=processor(images=image,text=chat,return_tensors='pt')['input_ids'][0].tolist()
    gold=processor(images=image,text=chat+' '+sample['answer'],return_tensors='pt')['input_ids'][0].tolist()
    bos=[i for i,t in enumerate(native) if t==processor.tokenizer.bos_token_id]
    visual=[i for i,t in enumerate(native) if t==32000]
    if len(bos)!=1 or len(visual)!=576 or gold[:len(native)]!=native:raise ValueError('Invalid native input mapping')
    question,_=locate_question_tokens(processor,sample['prompt_text'],native,32000)
    sample['input_mapping']=dict(prompt_ids=native,p_minus_one=len(native)-1,bos_positions=bos,bos_token_id=processor.tokenizer.bos_token_id,visual_positions=visual,question_and_instruction_positions=[r['sequence_index'] for r in question],gold_continuation_ids=gold[len(native):],chat_prompt_sha256=digest(chat))
    return sample


def historical_exclusions():
    paths=list(Path('/data/lk/ATE-worktrees').glob('*/vlm_research/results/**/*manifest*.json'))
    paths+=list(Path('/data/lk/深度学习/ATE/vlm_research/results').glob('**/*manifest*.json'))
    ids=set();hashes=set();records={};seen=set()
    def walk(x):
        if isinstance(x,dict):
            if 'image_id' in x:ids.add(str(x['image_id']))
            if 'imageId' in x:ids.add(str(x['imageId']))
            if 'image_sha256' in x:hashes.add(x['image_sha256'])
            for v in x.values():
                if isinstance(v,(dict,list)):walk(v)
        elif isinstance(x,list):
            for v in x:
                if isinstance(v,(dict,list)):walk(v)
    for p in sorted(paths):
        if 'bos_cheap_discovery' in p.parts:continue
        h=sha256(p)
        if h in seen:continue
        seen.add(h);before=len(ids)
        walk(json.loads(p.read_text()));records[str(p)]=dict(sha256=h,additional_ids=len(ids)-before)
    return ids,hashes,records


def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'execution_manifest.json').exists():raise ValueError('Already GPU-frozen')
    verify_updates()
    old=json.loads((OLD/'sample_manifest.json').read_text())['samples']
    for s in old:
        if sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Historical image changed')
    used,used_hashes,sources=historical_exclusions()
    used.update(s['image_id'] for s in old);used_hashes.update(s['image_sha256'] for s in old)
    # PyArrow stays in the data-only AFAC environment, not the frozen model environment.
    command='import json,pyarrow.parquet as pq,sys;json.dump(pq.read_table(sys.argv[1]).to_pylist(),sys.stdout)'
    rows=json.loads(subprocess.check_output(['/home/likai/anaconda3/envs/AFAC/bin/python','-c',command,str(ANNOTATION)]))
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
    processor.patch_size,processor.num_additional_image_tokens=14,1;processor.vision_feature_select_strategy='default'
    chosen=[];pools={};assigned=set();assigned_hashes=set();image_hash={}
    # Smallest available task first avoids consuming scarce independent images.
    pools_rows={t:[r for r in rows if r['types']['detailed'] in detailed and str(r['imageId']) not in used] for t,detailed in TASKS.items()}
    for task in sorted(TASKS,key=lambda t:len({r['imageId'] for r in pools_rows[t]})):
        candidates=sorted(pools_rows[task],key=lambda r:digest([SEED,'bos-new-image-v1',task,r['id']]))
        selected=[]
        for r in candidates:
            im=str(r['imageId'])
            if im in assigned:continue
            path=IMAGES/f'image_{im}.jpg'
            if not path.exists():raise ValueError('Input-only selected candidate missing; no silent replacement')
            if im not in image_hash:image_hash[im]=sha256(path)
            h=image_hash[im]
            if h in used_hashes or h in assigned_hashes:continue
            s=dict(task=task,question_id=r['id'],image_id=im,image_path=str(path),question=r['question'],answer=r['answer'],type_detailed=r['types']['detailed'],source_row_sha256=digest(r),source='Frozen official GQA balanced validation parquet',accuracy_subset=False,rank=int(digest([SEED,'worker',im]),16)%4)
            selected.append(mapping(s,processor));assigned.add(im);assigned_hashes.add(h)
            if len(selected)==100:break
        pools[task]=dict(unused_questions=len(candidates),unused_images=len({r['imageId'] for r in candidates}),selected=len(selected))
        chosen+=selected;print(task,pools[task],flush=True)
    old_split={s['task']+'__'+s['question_id']:split(s['image_sha256']) for s in old}
    if len({s['image_id'] for s in chosen})!=len(chosen) or used_hashes&{s['image_sha256'] for s in chosen}:raise ValueError('Heldout image leakage')
    # Strictly bind the observed update decomposition to old sample/input identities.
    update_manifest=json.loads((UPDATES/'sample_manifest.json').read_text())['samples'];lookup={s['task']+'__'+s['question_id']:s for s in old}
    for s in update_manifest:
        if digest(s)!=digest(lookup[s['task']+'__'+s['question_id']]):raise ValueError('Update/sample binding differs')
    freeze_json(OUT/'sample_manifest.json',dict(old_samples=old,new_samples=chosen,old_splits=old_split))
    for rank in range(4):freeze_json(OUT/f'manifests/shard{rank}.json',dict(old=[s for s in old if s['rank']==rank],new=[s for s in chosen if s['rank']==rank]))
    freeze_json(OUT/'data_manifest.json',dict(protocol=PROTOCOL,historical_questions=3000,new_questions=len(chosen),requested_new=600,independent_confirmation_available=len(chosen)==600,
        status='exploratory_only' if len(chosen)<600 else 'ready',pools=pools,excluded_image_ids=len(used),historical_manifest_sources=sources,
        historical_split_counts={t:{part:sum(s['task']==t and old_split[s['task']+'__'+s['question_id']]==part for s in old) for part in ['train','calibration']} for t in TASKS},
        reused_updates=dict(commit='b91f4ae259ecf4ce1307a1abb2836d035e612077',samples=600,strict_sample_binding=True,fields='Actual FP16 nodes; FP32 ratios share h_in denominator; post-Attention features mechanism-only.'),
        source_sha256={str(p):sha256(p) for p in [ANNOTATION,OLD/'sample_manifest.json',OLD/'completion.json',UPDATES/'completion.json',UPDATES/'per_question_updates.csv',UPDATES/'sample_manifest.json']},
        sample_sha256=sha256(OUT/'sample_manifest.json'),shard_sha256={str(i):sha256(OUT/f'manifests/shard{i}.json') for i in range(4)}))


if __name__=='__main__':main()
