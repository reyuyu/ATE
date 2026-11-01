"""Optional frozen LLaVA sparse-visual rescue under relay OFF, no new search."""
import argparse
from datetime import datetime
import gc
import time
import traceback

import torch

from .common import ROOT,OUT,EXPERIMENT_END,read,write,sha,digest,preflight,command
from .relay import RDIR,MODEL,prepare,generate
from .engine import metrics
from ..internvl3_smoke.engine import tensor_hash,hooks_count

DIR=OUT/'llava_rescue'
CONDITIONS={'FullImageAccess':[],'AllImageOFF':list(range(32)),
            'KeepSparse':[i for i in range(32) if i not in (14,19)]}


def freeze():
    assert not (DIR/'protocol.json').exists()
    assert read(RDIR/'gpu3_shard0of1/complete.json')['done']==240
    samples=read(RDIR/'sample_manifest.json')
    write(DIR/'sample_manifest.json',samples)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'vlm_research/visual_pathways/rescue.py',ROOT/'vlm_research/visual_pathways/relay.py',ROOT/'vlm_research/visual_pathways/engine.py']}
    write(DIR/'protocol.json',dict(parent_protocol_sha256=sha(OUT/'protocol.json'),
        source_sha256=sources,manifest_sha256=sha(DIR/'sample_manifest.json'),conditions=CONDITIONS,
        keep_layers=[14,19],relay_off=True,decode='Fully native, independent per-condition cache',
        intervention='Original P-1 Image-source hard mask except frozen keep layers; all nonvisual earlier Prefill queries Image OFF in every layer',
        fullimage_reuse='Exact same-sample Task2 C generation; native A first logits recomputed once only for fixed target verification',
        bootstrap=read(OUT/'protocol.json')['bootstrap'],scope='Optional exploratory 240-question rescue, not extra layer search'))


@torch.no_grad()
def worker(gpu):
    p=read(DIR/'protocol.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    assert sha(DIR/'sample_manifest.json')==p['manifest_sha256']
    folder=DIR/f'gpu{gpu}';preflight(gpu,folder)
    from ..vlm_common import load_model
    model,processor,environment=load_model(MODEL)
    write(folder/'environment.json',environment)
    samples=read(DIR/'sample_manifest.json')['samples']
    # Directly exercise the most restrictive P-1 Image mask and exact no-op/native.
    inputs=prepare(model,processor,samples[0])
    native,first,_=generate(model,processor,inputs,native=True)
    noop,noop_first,_=generate(model,processor,inputs,True,image_off=range(32),noop=True)
    assert native['token_ids']==noop['token_ids'] and torch.equal(first,noop_first)
    validations=[]
    for name in ('AllImageOFF','KeepSparse'):
        result,_,_=generate(model,processor,inputs,True,image_off=CONDITIONS[name])
        assert len(result['prefill_mask_audits'])==32
        assert all(a['masked_probability_max']==0 for a in result['prefill_mask_audits'].values())
        validations.append(dict(condition=name,masked_layers=CONDITIONS[name],native_decode=True,probability_zero=True))
    write(folder/'validation.json',dict(passed=True,noop_logits_max_abs=0,checks=validations,code_commit=command('git','rev-parse','HEAD')))
    versions=[(p._version,p.data_ptr()) for p in model.parameters()];baseline=hooks_count(model)
    start=time.perf_counter();done=0
    for s in samples:
        if datetime.now().astimezone()>=datetime.fromisoformat(EXPERIMENT_END):
            break
        key=f'{s["task"]}__{s["question_id"]}';path=DIR/'checkpoints'/f'{key}.json'
        if path.exists() and read(path).get('complete'):
            r=read(path);assert r['sample_sha256']==digest(s) and r['protocol_sha256']==sha(DIR/'protocol.json')
            done+=1;continue
        oldpath=RDIR/'checkpoints'/f'{key}.json';old=read(oldpath)
        assert old['complete'] and old['sample_sha256']==digest(s) and old['protocol_sha256']==sha(RDIR/'protocol.json')
        inputs=prepare(model,processor,s)
        with torch.inference_mode():
            full=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu().clone()
        assert tensor_hash(full)==old['full_logits_sha256'], 'Recomputed native first logits differ from same-machine Task2 Full'
        conditions={'FullImageAccess':dict(old['conditions']['C'])}
        for name in ('AllImageOFF','KeepSparse'):
            result,changed,_=generate(model,processor,inputs,True,image_off=CONDITIONS[name])
            result.update(metrics(full,changed),correct=int(result['prediction']==s['answer']),physical_gpu=gpu)
            conditions[name]=result
        write(path,dict(complete=True,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
            image_sha256=s['image_sha256'],sample_sha256=digest(s),protocol_sha256=sha(DIR/'protocol.json'),
            full_logits_sha256=old['full_logits_sha256'],native_full_accuracy=old['conditions']['A']['correct'],
            reused_from=str(oldpath),reused_sha256=sha(oldpath),conditions=conditions))
        done+=1
        write(folder/'progress.json',dict(done=done,total=240,seconds=time.perf_counter()-start))
        if done%20==0:print('RESCUE',done,240,round(time.perf_counter()-start,1),flush=True)
    assert versions==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==baseline
    write(folder/'complete.json',dict(passed=True,done=done,total=240,seconds=time.perf_counter()-start,
        weights_unchanged=True,hooks_clean=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker']);p.add_argument('--gpu',type=int,choices=[0,2,3],default=3);args=p.parse_args()
    if args.action=='freeze':freeze()
    else:
        try:worker(args.gpu)
        except BaseException:
            write(DIR/f'gpu{args.gpu}'/'error.json',dict(traceback=traceback.format_exc()));raise
