import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from benchmarks.dataset_registry import digest,freeze_json,sha256
from paper_rq1.analyze import write_csv
from paper_rq1_token_role_decode.labels import align
from paper_rq1_token_role_decode.run import raw_save
from .protocol import ROOT,OUT,MODEL_PATH,PROTOCOL,SEED
from .predictor import prefix_features,hidden_features,category
from .analysis import figure
import matplotlib.pyplot as plt


def training_sources():
    return {str(ROOT/n):sha256(ROOT/n) for n in ['paper_rq1_function_attention_sparsify/predictor.py',
        'paper_rq1_function_attention_sparsify/train.py','paper_rq1_function_attention_sparsify/analysis.py',
        'test_function_predictor.py']}


def precision_interval(predicted,y,images):
    keys=sorted(set(images));counts=np.array([[np.sum(predicted & (images==i) & y),np.sum(predicted & (images==i))] for i in keys])
    rng=np.random.default_rng(SEED);res=[]
    for i in range(0,10000,500):
        sums=counts[rng.integers(0,len(keys),(500,len(keys)))].sum(1)
        res.extend(np.divide(sums[:,0],sums[:,1],out=np.zeros(500),where=sums[:,1]>0))
    return np.quantile(res,[.025,.975]).tolist()


def measures(p,y,threshold):
    selected=p>=threshold;n=int(selected.sum());tp=int((selected & y).sum());fp=int((selected & ~y).sum())
    return dict(threshold=threshold,predicted=n,true_positive=tp,false_positive=fp,
        precision=tp/n if n else None,recall=tp/max(1,int(y.sum())),false_positive_rate=fp/max(1,int((~y).sum())),
        coverage=n/len(y),brier=float(np.mean((p-y.astype(float))**2)))


def main():
    if not (OUT/'stage_a_gate.json').exists():raise ValueError('StageA must finish first')
    if (OUT/'predictor_freeze.json').exists():raise ValueError('Predictor already frozen; refuse retrain')
    freeze_json(OUT/'training_protocol_freeze.json',dict(protocol=PROTOCOL['classifier'],sources=training_sources(),
        sample_sha256=sha256(OUT/'sample_manifest.json'),stage_a_gate_sha256=sha256(OUT/'stage_a_gate.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),holdout_outputs_read=False))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=True)
    samples=json.loads((OUT/'sample_manifest.json').read_text())['development'];rows=[];prefixes=[];hidden=[]
    for s in samples:
        path=OUT/f"stage_a/shard{s['rank']}/checkpoints/{s['image_identity'].replace(':','_')}.json"
        record=json.loads(path.read_text())
        if sha256(record['raw_path'])!=record['raw_sha256']:raise ValueError('Changed feature artifact')
        with np.load(record['raw_path']) as f:h=f['h31_in']
        tokens=record['token_ids'];annotation=align(tokenizer,tokens)
        for t in range(1,len(tokens)):
            c=category(annotation,t)
            rows.append(dict(image_id=s['image_identity'],image_sha256=s['image_sha256'],split=s['split'],t=t,
                token_id=tokens[t],category=c,function=c=='Function',token_piece=annotation['tokens'][t]['token_piece']))
            prefixes.append(prefix_features(tokens[:t]));hidden.append(hidden_features(h[t],t).numpy())
    write_csv(OUT/'all_development_token_labels.csv',rows)
    xs={'A':np.stack(prefixes),'B':np.stack(hidden)};xs['C']=np.concatenate([xs['A'],xs['B']],1)
    y=np.array([r['function'] for r in rows]);train=np.array([r['split']=='train' for r in rows]);cal=~train
    image=np.array([r['image_id'] for r in rows]);parameters={};predictions={};summaries=[];curves=[];timings=[];categories=[];chosen=[]
    torch.manual_seed(SEED);torch.set_num_threads(4)
    for kind,x in xs.items():
        mean=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-6]=1.
        features=torch.from_numpy((x[train]-mean)/scale);labels=torch.from_numpy(y[train].astype(np.float32))
        weight=torch.zeros(x.shape[1],requires_grad=True);bias=torch.zeros((),requires_grad=True)
        optimizer=torch.optim.Adam([weight,bias],lr=.03,weight_decay=.01);start=time.perf_counter()
        for epoch in range(300):
            optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(features@weight+bias,labels);loss.backward();optimizer.step()
        train_seconds=time.perf_counter()-start
        with torch.no_grad():p=torch.sigmoid(torch.from_numpy((x-mean)/scale)@weight+bias).numpy()
        predictions[kind]=p
        for name,value in [('mean',mean),('scale',scale),('weight',weight.detach().numpy()),('bias',bias.detach().numpy())]:parameters[kind+'_'+name]=value
        candidates=[]
        for threshold in PROTOCOL['classifier']['threshold_grid']:
            r=measures(p[cal],y[cal],threshold);selected=(p[cal]>=threshold);span=len(set(image[cal][selected]))
            lo,hi=precision_interval(selected,y[cal],image[cal]) if selected.any() else [None,None]
            r.update(predictor=kind,split='calibration',precision_ci95_low=lo,precision_ci95_high=hi,predicted_images=span)
            r['eligible']=lo is not None and lo>=.95 and r['predicted']>=50 and span>=20
            curves.append(r)
            if r['eligible']:candidates.append(r)
        best=max(candidates,key=lambda r:r['coverage']) if candidates else dict(measures(p[cal],y[cal],1.000001),predictor=kind,eligible=False,precision_ci95_low=None,precision_ci95_high=None,predicted_images=0,split='calibration')
        chosen.append(best);summaries.append(best)
        selected=p[cal]>=best['threshold']
        for c in sorted({r['category'] for r in rows}):
            mask=np.array([r['category']==c for r in rows])[cal]
            categories.append(dict(predictor=kind,category=c,n=int(mask.sum()),predicted_function=int((mask&selected).sum()),
                rate=float((mask&selected).sum()/mask.sum()) if mask.any() else None))
        timings.append(dict(predictor=kind,features=x.shape[1],train_tokens=int(train.sum()),calibration_tokens=int(cal.sum()),train_seconds=train_seconds,epochs=300,final_train_loss=float(loss)))
    eligible=[r for r in chosen if r['eligible']];selected=None
    for r in eligible:
        if selected is None or r['recall']>=selected['recall']+.05:selected=r
    artifact=raw_save(OUT/'predictor_parameters.npz',**parameters)
    write_csv(OUT/'classifier_calibration.csv',summaries);write_csv(OUT/'classifier_precision_coverage.csv',curves)
    write_csv(OUT/'classifier_category_errors.csv',categories);write_csv(OUT/'classifier_training_cost.csv',timings)
    write_csv(OUT/'development_predictions.csv',[dict(r,**{kind+'_probability':float(p[i]) for kind,p in predictions.items()}) for i,r in enumerate(rows)])
    gate=json.loads((OUT/'stage_a_gate.json').read_text())
    freeze_json(OUT/'predictor_freeze.json',dict(selected=selected,classifier_gate_passed=selected is not None,
        dynamic_generation_allowed=selected is not None and gate['dynamic_source_gate_passed'],thresholds={r['predictor']:r['threshold'] for r in chosen},
        parameters=artifact,sources=training_sources(),training_protocol_sha256=sha256(OUT/'training_protocol_freeze.json'),
        holdout_outputs_read=False,selection_rule=PROTOCOL['classifier']['calibration_gate'],
        failure='No threshold satisfies image-cluster precision lower bound and coverage minimum' if selected is None else None))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for kind,color in zip(xs,['#176b91','#b24039','#4b7951']):
        rs=[r for r in curves if r['predictor']==kind and r['precision'] is not None]
        axes[0].plot([r['coverage'] for r in rs],[r['precision'] for r in rs],marker='o',label=kind,color=color)
        ps=predictions[kind][cal]
        bins=[]
        for lo in np.linspace(0,.9,10):
            m=(ps>=lo)&(ps<lo+.1)
            if m.any():bins.append((ps[m].mean(),y[cal][m].mean()))
        axes[1].plot([a for a,b in bins],[b for a,b in bins],marker='o',label=kind,color=color)
        rs=[r for r in categories if r['predictor']==kind and r['category']!='Function']
        axes[2].plot(range(len(rs)),[r['rate'] or 0 for r in rs],marker='o',label=kind,color=color)
    axes[0].set(xlabel='Calibration coverage',ylabel='Function precision',ylim=(0,1.02));axes[0].axhline(.95,color='grey',ls=':')
    axes[1].plot([0,1],[0,1],color='grey',ls=':');axes[1].set(xlabel='Predicted probability',ylabel='Empirical Function rate')
    axes[2].set(xticks=range(len(rs)),xticklabels=[r['category'] for r in rs],ylabel='Predicted-Function rate');axes[2].tick_params(axis='x',rotation=65)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    figure(fig,'figure3_function_identification_calibration')
    print(json.dumps(json.loads((OUT/'predictor_freeze.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
