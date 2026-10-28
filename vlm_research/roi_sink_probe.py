"""Phase 3.4: observe -> freeze -> GPU smoke -> small paired causal diagnostic."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from vlm_common import ROOT,load_model,prepare_inputs
from knockout_probe import resource_preflight,tensor_hash
from attention_position_probe import locate_question_tokens
from physical_exit_pilot import candidate_inputs,sequence_score
from physical_exit import cosine_distances
from fixed_k_exit import atomic_json,configuration as historical_config
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from roi_sink_common import OUTPUT,PROTOCOL,NEUTRAL,RANDOM_SEEDS,WINDOWS,forward,sink_candidates,matched_background

PARENT=ROOT/'results/phase3_2_fixed_k_exit'


def config_and_data():
    manifest=json.loads((OUTPUT/'manifest.json').read_text());protocol=json.loads((OUTPUT/'preregistered_protocol.json').read_text())
    if protocol!=PROTOCOL or manifest['protocol_sha256']!=digest(protocol):raise ValueError('Frozen protocol changed')
    if sha256(PARENT/'manifest.json')!=manifest['parent_manifest_sha256']:raise ValueError('Parent data changed')
    previous=historical_config()
    if previous!=json.loads((PARENT/'experiment_config.json').read_text()):raise ValueError('Old model/environment/source changed')
    ids=manifest['main_question_ids']+manifest['case_question_ids'];lookup={s['question_id']:s for s in manifest['samples']}
    if len(ids)!=53 or len(set(ids))!=53:raise ValueError('Wrong or overlapping main/cases')
    samples=[lookup[q] for q in ids]
    for s in samples:
        if sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Image changed')
    config=dict(phase='3.4',manifest_sha256=sha256(OUTPUT/'manifest.json'),protocol_sha256=digest(protocol),
                code_sha256={x:sha256(ROOT/x) for x in ['roi_sink_common.py','roi_sink_probe.py','prepare_roi_sink.py']},
                historical_config=previous,main_n=48,case_n=5,raw_observation='Q mean rows and P-1 per head, float16 storage; neutral maps head mean only; no full attention matrices saved',
                activation='Published ICLR phi: max signed raw pre-layer hidden[1415,2533] / RMS over4096, threshold20 descriptive only; candidates remain attention-persistence defined.')
    path=OUTPUT/'experiment_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Experiment config/code mismatch')
    if not path.exists():atomic_json(path,config)
    return config,manifest,samples


def save_checkpoint(path,binding,sample,data):
    if path.exists():raise ValueError('Never overwrite completed sample')
    body=dict(binding=binding,sample_sha256=digest(sample),data=data)
    atomic_json(path,dict(body=body,sha256=digest(body)))


def read_checkpoint(path,binding,sample):
    envelope=json.loads(path.read_text());body=envelope['body']
    if envelope['sha256']!=digest(body) or body['binding']!=binding or body['sample_sha256']!=digest(sample):raise ValueError('Checkpoint mismatch/tamper')
    return body['data']


def dataset(stage,samples,binding):
    folder=OUTPUT/'checkpoints'/stage;folder.mkdir(parents=True,exist_ok=True)
    allowed={s['question_id']:s for s in samples};result={}
    for f in folder.glob('*.json'):
        if f.stem not in allowed:raise ValueError('Unexpected checkpoint')
        result[f.stem]=read_checkpoint(f,binding,allowed[f.stem])
        for path,expected in result[f.stem].get('artifact_sha256',{}).items():
            if sha256(OUTPUT/path)!=expected:raise ValueError('Observation artifact changed')
    return result


def gate(stage,binding,count):
    d=json.loads((OUTPUT/f'{stage}_complete.json').read_text())
    if d['binding']!=binding or not d['passed'] or d['count']!=count:raise ValueError('Preceding stage incomplete')


def inputs_and_history(model,processor,s):
    prompt,counts=prepare_inputs(model,processor,s['image_path'],s['question']);ids=prompt['input_ids'][0].cpu().numpy();P=len(ids)
    records,audit=locate_question_tokens(processor,s['question'],ids,model.config.image_token_index);qpos=[r['sequence_index'] for r in records]
    history=json.loads((PARENT/'checkpoints/run'/f"{s['question_id']}.json").read_text())['body']['data']
    candidates=candidate_inputs(processor,prompt,s['candidates'])
    if ids.tolist()!=history['prompt_ids'] or records!=history['question_mapping'] or max(qpos)>=P-1:raise AssertionError('Frozen question/prompt alignment')
    for (x,rec),old in zip(candidates,history['candidate_mapping']):
        if len(rec)!=1 or rec[0]['query_position']!=P-1 or rec[0]['target_token_position']!=P:raise AssertionError('Wrong P-1 prediction query')
        if x['input_ids'][0].cpu().tolist()!=old['input_ids'] or rec!=old['answer_records']:raise AssertionError('Candidate mapping changed')
    return prompt,candidates,qpos,P-1,history,records,counts


def score(out,record):return sequence_score(out['prediction_logits'][None],[record[0]['token_id']])


def observe_sample(model,processor,s,binding):
    prompt,candidates,qpos,apos,history,records,counts=inputs_and_history(model,processor,s)
    full=[forward(model,x,qpos,apos,observe=(i==0),norms=(i==0)) for i,(x,rec) in enumerate(candidates)]
    logs=[score(o,rec) for o,(x,rec) in zip(full,candidates)]
    if logs!=[c['logp'] for c in history['full_candidates']]:raise AssertionError('Historical Full scores changed')
    if not torch.equal(full[0]['prediction_logits'],full[1]['prediction_logits']):raise AssertionError('Candidate leaks into P-1')
    plain=forward(model,candidates[0][0],qpos,apos)
    if not torch.equal(full[0]['prediction_logits'],plain['prediction_logits']) or not torch.equal(full[0]['question_hidden'],plain['question_hidden']):raise AssertionError('Stream attention affects predictions')
    folder=OUTPUT/'attention_rows';folder.mkdir(exist_ok=True);path=folder/f"{s['question_id']}_task.npz"
    np.savez_compressed(path,attention=full[0]['rows'],question_hidden=full[0]['question_hidden'].numpy(),
                        key_norm=full[0]['key_norm'].mean(1).astype(np.float16),value_norm=full[0]['value_norm'].mean(1).astype(np.float16),
                        activation_max=full[0]['activation_max'].astype(np.float16),sink_phi=full[0]['sink_phi'].astype(np.float16))
    norm_stats=[]
    for name in ['key_norm','value_norm']:
        # Per-head regional norm summaries will use frozen candidates later; save complete small norm arrays separately.
        norm_stats.append((name,full[0][name].astype(np.float16)))
    normpath=folder/f"{s['question_id']}_norms.npz";np.savez_compressed(normpath,**dict(norm_stats))
    neutral=[];measures=[]
    for text in NEUTRAL:
        x,_=prepare_inputs(model,processor,s['image_path'],text);r,_=locate_question_tokens(processor,text,x['input_ids'][0].cpu().numpy(),model.config.image_token_index)
        o=forward(model,x,[v['sequence_index'] for v in r],x['input_ids'].shape[1]-1,observe=True)
        neutral.append(o['rows'].astype(np.float64).mean(1).astype(np.float16));measures.append(o['measurement'])
    npath=folder/f"{s['question_id']}_neutral.npz";np.savez_compressed(npath,attention=np.stack(neutral))
    gold=s['candidates'].index(s['answer']);margin=logs[gold]-logs[1-gold]
    return dict(question_id=s['question_id'],main=s['question_id'] in json.loads((OUTPUT/'manifest.json').read_text())['main_question_ids'],
                full_logp=logs,full_margin=margin,question_mapping=records,answer_query=apos,counts=counts,
                attention_audit=full[0]['attention_audit'],stream_prediction_exact=True,history_full_exact=True,
                measurements=[o['measurement'] for o in full]+[plain['measurement']]+measures,
                artifact_sha256={str(f.relative_to(OUTPUT)):sha256(f) for f in [path,normpath,npath]})


def freeze(config,manifest,samples,binding):
    gate('observe',binding,53);dataset('observe',samples,binding);entries=[]
    for s in samples:
        q=s['question_id'];f=OUTPUT/'attention_rows'/f'{q}_neutral.npz'
        candidate=sink_candidates(np.load(f)['attention']);roi=set(s['roi_indices']);sink=set(candidate['indices']);pool=sorted(set(range(576))-roi-sink)
        nonroi=sorted(sink-roi);randoms={}
        for label,target in [('roi',sorted(roi)),('sink',nonroi)]:
            for seed in RANDOM_SEEDS:randoms[f'random_{label}_s{seed}']=matched_background(target,pool,seed+int(q))
        entries.append(dict(question_id=q,main=q in manifest['main_question_ids'],roi_indices=sorted(roi),
                            sink_indices=sorted(sink),sink_nonroi_indices=nonroi,roi_sink_overlap_indices=sorted(roi&sink),
                            neutral_npz_sha256=sha256(f),sink_identification=candidate,random_controls=randoms))
    value=dict(binding=binding,rule=PROTOCOL['sink_rule'],samples=entries)
    path=OUTPUT/'sink_manifest.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen sink definitions changed')
    if not path.exists():atomic_json(path,value)
    print('Frozen sink candidates and spatial/count controls before causal scoring:',len(entries))


def plans(entry):
    targets=[('roi',entry['roi_indices'],True),('sink_nonroi',entry['sink_nonroi_indices'],True)]
    targets += [(name,r.get('indices',[]),r['available']) for name,r in entry['random_controls'].items()]
    return [(group,window,name,patches,available) for group in ['question','answer'] for window in WINDOWS for name,patches,available in targets]


def causal_sample(model,processor,s,entry,smoke=False):
    prompt,candidates,qpos,apos,history,records,counts=inputs_and_history(model,processor,s)
    full=[forward(model,x,qpos,apos) for x,rec in candidates];baseline=[score(o,rec) for o,(x,rec) in zip(full,candidates)]
    if baseline!=[c['logp'] for c in history['full_candidates']]:raise AssertionError('Native Full/history mismatch')
    gold=s['candidates'].index(s['answer']);full_margin=baseline[gold]-baseline[1-gold];rows=[];audits=[]
    def record(group,window,name,keys,available,logs=None,values=None):
        margin=logs[gold]-logs[1-gold] if logs is not None else None
        rows.append(dict(question_id=s['question_id'],img_id=s['img_id'],answer=s['answer'],main=entry['main'],query_group=group,window=window,condition=name,
                         available=available,blocked_patches=len(keys),sham=(len(keys)==0),full_margin=full_margin,full_correct=full_margin>0,
                         answer_margin=margin,answer_margin_drop=full_margin-margin if margin is not None else None,
                         candidate_correct=margin>0 if margin is not None else None,
                         correct_logp=logs[gold] if logs is not None else None,wrong_logp=logs[1-gold] if logs is not None else None,
                         correct_probability=float(np.exp(logs[gold])) if logs is not None else None,
                         weak_margin=(abs(full_margin)<=.15 or abs(margin)<=.15) if margin is not None else None,
                         question_drift=float(cosine_distances(full[0]['question_hidden'],values[0]['question_hidden']).mean()) if values else None,
                         forward_seconds_two_candidates=sum(o['measurement']['forward_seconds'] for o in values) if values else None,
                         peak_allocated_mib=max(o['measurement']['gpu_memory']['peak_allocated_mib'] for o in values) if values else None))
    record('full','none','full',[],True,baseline,full)
    for group,window,name,keys,available in plans(entry):
        if not available:record(group,window,name,keys,False);continue
        queries=qpos if group=='question' else [apos]
        values=[forward(model,x,qpos,apos,queries,keys,WINDOWS[window],observe=smoke and i==0) for i,(x,rec) in enumerate(candidates)]
        logs=[score(o,rec) for o,(x,rec) in zip(values,candidates)]
        if group=='answer' and not all(torch.equal(a['question_hidden'],b['question_hidden']) for a,b in zip(full,values)):raise AssertionError('A-only changed earlier Question')
        if not keys and (logs!=baseline or not torch.equal(values[0]['question_hidden'],full[0]['question_hidden'])):raise AssertionError('Empty-key sham changed output')
        for o in values:
            if [r['applied'] for r in o['hook_audit']]!=[i in WINDOWS[window] for i in range(32)]:raise AssertionError('Wrong active layers')
        record(group,window,name,keys,True,logs,values)
        audits.append(dict(query_group=group,window=window,condition=name,attention_audit=values[0]['attention_audit'],
                           hook_audit=values[0]['hook_audit'],measurements=[o['measurement'] for o in values]))
    after=forward(model,candidates[0][0],qpos,apos)
    if not torch.equal(after['prediction_logits'],full[0]['prediction_logits']) or not torch.equal(after['question_hidden'],full[0]['question_hidden']):raise AssertionError('Post-intervention contamination')
    if len(rows)!=33 or len({(r['query_group'],r['window'],r['condition']) for r in rows})!=33:raise AssertionError('Incomplete/duplicate plans')
    return dict(rows=rows,audits=audits,post_cleanup_exact=True,answer_question_hidden_exact=True,history_full_exact=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['observe','freeze','smoke','causal']);args=parser.parse_args()
    config,manifest,samples=config_and_data();binding=digest(config)
    lock=(OUTPUT/'.runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='freeze':freeze(config,manifest,samples,binding);return
    entries=None;stage_binding=binding
    if args.stage in ['smoke','causal']:
        gate('observe',binding,53);sink=json.loads((OUTPUT/'sink_manifest.json').read_text())
        if sink['binding']!=binding:raise ValueError('Sink config mismatch')
        for e in sink['samples']:
            if e['neutral_npz_sha256']!=sha256(OUTPUT/'attention_rows'/f"{e['question_id']}_neutral.npz"):raise ValueError('Neutral observations changed')
        entries={r['question_id']:r for r in sink['samples']};stage_binding=digest(dict(config_binding=binding,sink_sha256=sha256(OUTPUT/'sink_manifest.json')))
        smoke_ids=[manifest['main_question_ids'][0],manifest['main_question_ids'][24],manifest['case_question_ids'][0]]
        if args.stage=='smoke':samples=[s for s in samples if s['question_id'] in smoke_ids]
        else:gate('smoke',stage_binding,3)
    done=dataset(args.stage,samples,stage_binding)
    if len(done)==len(samples):print('Already complete, checkpoints verified',args.stage,len(done));return
    logs=OUTPUT/'logs';logs.mkdir(exist_ok=True);stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());started=time.perf_counter()
    try:
        resource=resource_preflight();atomic_json(logs/f'{stamp}_{args.stage}_resource.json',resource)
        if not resource['ready']:raise RuntimeError('GPU unavailable; no other process stopped')
        model,processor,environment=load_model();atomic_json(logs/f'{stamp}_{args.stage}_environment.json',environment)
        for i,s in enumerate(samples):
            if s['question_id'] in done:continue
            print(args.stage,i+1,'/',len(samples),s['question_id'],flush=True)
            data=observe_sample(model,processor,s,binding) if args.stage=='observe' else causal_sample(model,processor,s,entries[s['question_id']],args.stage=='smoke')
            save_checkpoint(OUTPUT/'checkpoints'/args.stage/f"{s['question_id']}.json",stage_binding,s,data)
        complete=dataset(args.stage,samples,stage_binding)
        atomic_json(OUTPUT/f'{args.stage}_complete.json',dict(binding=stage_binding,count=len(complete),passed=len(complete)==len(samples),wall_seconds=time.perf_counter()-started))
        print('COMPLETE',args.stage,len(complete),time.perf_counter()-started,flush=True)
    except Exception as e:
        atomic_json(logs/f'{stamp}_{args.stage}_failure.json',dict(error=str(e),traceback=traceback.format_exc(),gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
          recovery='Keep frozen data/code, rerun same stage. Completed checkpoint checksums validated, skipped; no overwrite or resampling.'))
        raise


if __name__=='__main__':main()
