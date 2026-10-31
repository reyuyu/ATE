"""CPU-only paired statistics and small source-specific paper figures."""
import csv
import itertools
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LogNorm
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import digest,sha256
from paper_rq1.scorers import gqa_prediction
from l0l1_bos.analyze import write_csv,save
from .protocol import OUT,CONDITIONS,KEYS,old,read,verify,entries,execution,identity,check_reuse,atomic_json
from .run import valid_checkpoint
from .statistics import summarize,intersections


def plots(summary,taskwise,interactions,masses,overlap):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    selected=summary[1:];x=np.arange(11)
    fig,ax=plt.subplots(figsize=(10.5,5.5),constrained_layout=True)
    estimates=np.array([r['delta_full_pp'] for r in selected])
    ax.errorbar(x,estimates,yerr=[estimates-np.array([r['delta_ci95_low'] for r in selected]),
        np.array([r['delta_ci95_high'] for r in selected])-estimates],fmt='o',color='#247697',capsize=3,label='Paired95 CI')
    ax.axhline(0,color='#777777',lw=.8);ax.set_xticks(x,[r['condition'].replace('_','\n') for r in selected],rotation=30,ha='right')
    ax.set_ylabel('Complete-answer Accuracy - Full (percentage points)')
    ax.set_title('240 frozen questions | Prefill only | A1-A6 exploratory')
    ax.legend();save(fig,folder/'accuracy_effects')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for j,r in enumerate(interactions):
        axes[0].bar([j*3,j*3+1],[r['normal_source_harm_pp'],r['earlyoff_source_harm_pp']],
            color=['#267a99','#c85846'],label=None)
    axes[0].set_xticks([.5,3.5],['L2 Image','L2 Question'])
    axes[0].set_ylabel('Accuracy harm from L2 source mask (pp)')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='#267a99',label='Normal early attention'),Patch(color='#c85846',label='Early P-1 OFF')])
    for j,r in enumerate(interactions):
        value=r['D_pp']
        axes[1].errorbar(j,value,yerr=[[value-r['D_family95_low']],[r['D_family95_high']-value]],fmt='none',ecolor='#aab2b5',capsize=6,lw=4)
        axes[1].errorbar(j,value,yerr=[[value-r['D_ci95_low']],[r['D_ci95_high']-value]],fmt='o',color='#273b43',capsize=3)
    axes[1].axhline(0,color='#777777',lw=.8);axes[1].set_xticks([0,1],['D_visual','D_question'])
    axes[1].set_ylabel('Extra conditional Accuracy harm (pp)')
    axes[1].set_title('Dark: paired95 | Light: family95 (two tests)')
    save(fig,folder/'conditional_source_dependence')
    lookup={(r['group'],r['condition']):r for r in taskwise}
    a=np.array([[lookup[t,c]['accuracy_pct'] for c in KEYS] for t in old.TASKS])
    diff=a-a[:,:1];limit=max(abs(diff).max(),.01)
    fig,ax=plt.subplots(figsize=(13,4.2),constrained_layout=True)
    image=ax.imshow(diff,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for y,x in itertools.product(range(6),range(12)):ax.text(x,y,f'{a[y,x]:.1f}',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(12),[k.replace('_','\n') for k in KEYS],rotation=35,ha='right');ax.set_yticks(range(6),old.TASKS)
    ax.set_title('Complete-answer Accuracy (%) | 40/task | Color: change from Full (pp)')
    fig.colorbar(image,ax=ax,label='Accuracy change (pp)');save(fig,folder/'task_accuracy')
    fig,axes=plt.subplots(1,4,figsize=(14,5.5),constrained_layout=True)
    for ax,source in zip(axes,['bos','image','question','template']):
        matrix=np.array([[np.mean([r['delta_'+source+'_pp'] for r in masses if r['condition']==c and r['layer']==l])
            for l in range(2,6)] for c in CONDITIONS])
        limit=max(abs(matrix).max(),.01)
        image=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(4),['L2','L3','L4','L5']);ax.set_yticks(range(11),list(CONDITIONS),fontsize=8)
        ax.set_title(source.title()+' Mass change')
        fig.colorbar(image,ax=ax,label='OFF - Full (pp)',shrink=.75)
    save(fig,folder/'layer_source_mass_changes')
    matrix=np.zeros((11,11),dtype=int);index={k:i for i,k in enumerate(CONDITIONS)}
    for r in overlap:
        a,b=index[r['condition_a']],index[r['condition_b']]
        matrix[a,b]=matrix[b,a]=r['full_correct_joint_harm']
    for i,r in enumerate(summary[1:]):matrix[i,i]=r['correct_to_wrong']
    fig,ax=plt.subplots(figsize=(8,6.7),constrained_layout=True);image=ax.imshow(matrix,cmap='YlGnBu')
    for y,x in itertools.product(range(11),repeat=2):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=8)
    ax.set_xticks(range(11),list(CONDITIONS),rotation=60,ha='right',fontsize=8);ax.set_yticks(range(11),list(CONDITIONS),fontsize=8)
    ax.set_title('Joint harm among Full-correct questions | diagonal: individual harm')
    fig.colorbar(image,ax=ax,label='Questions');save(fig,folder/'error_intersections')


def case_figures(entry,bodies):
    s=entry['sample'];folder=OUT/'cases'/identity(s);mapping=entry['mapping']
    arrays={}
    for c,b in bodies.items():
        assert b['arrays']['sha256']==sha256(OUT/b['arrays']['path'])
        with np.load(OUT/b['arrays']['path']) as z:arrays[c]={k:z[k] for k in z.files}
    maps=[]
    for layer in [0,1]:
        raw=arrays['Full'][f'attention_row_{layer}'][:,mapping['image']].astype(float).mean(0)
        maps.append((raw,f'Full L{layer} P-1'))
    for layer in [0,1]:
        raw=arrays['Full'][f'question_visual_row_{layer}'].astype(float).mean(0)
        maps.append((raw,f'Full L{layer} Question mean'))
    for c in ['Full','EarlyOFF']:
        raw=arrays[c]['attention_row_2'][:,mapping['image']].astype(float).mean(0)
        maps.append((raw,f'{c} L2 P-1'))
    high=max((raw/raw.sum()).max() for raw,title in maps if raw.sum()>0)
    fig,axes=plt.subplots(2,4,figsize=(13.5,7.4),constrained_layout=True)
    axes=axes.ravel();axes[0].imshow(Image.open(s['image_path']));axes[0].set_title('Original image');axes[0].axis('off')
    axes[1].imshow(Image.open(folder/'clip_input.png'));axes[1].set_title('Actual CLIP336 crop');axes[1].axis('off')
    for ax,(raw,title) in zip(axes[2:],maps):
        mass=raw.sum()
        assert mass>0
        image=ax.imshow((raw/mass).reshape(24,24),cmap='magma',norm=LogNorm(1e-5,max(high,1.01e-5)),interpolation='nearest')
        ax.set_title(f'{title}\nRaw Image Mass={100*mass:.2f}%',fontsize=9)
        ax.set_xticks([0,8,16,23]);ax.set_yticks([0,8,16,23])
    fig.colorbar(image,ax=list(axes[2:]),label='Visual-conditional distribution after query/head mean (log scale)',shrink=.75)
    title=f'{s["task"]} {s["question_id"]} | outcome-blind representative\n'+textwrap.fill(s['question'],100)
    fig.suptitle(title,fontsize=11);save(fig,folder/'visual_patch_maps')
    fig,axes=plt.subplots(1,3,figsize=(13.5,6),constrained_layout=True)
    for ax,source in zip(axes,['bos','image','question']):
        mat=np.array([arrays[c]['attention_row_2'][:,mapping[source]].astype(float).sum(-1) for c in KEYS])*100
        image=ax.imshow(mat,cmap='viridis',vmin=0,vmax=100,aspect='auto',interpolation='nearest')
        ax.set_yticks(range(12),KEYS,fontsize=8);ax.set_xticks([0,8,16,24,31]);ax.set_xlabel('Head');ax.set_title('L2 P-1 -> '+source)
    fig.colorbar(image,ax=list(axes),label='Raw Attention Mass (%)')
    fg=bodies['Full']['generation'];eg=bodies['EarlyOFF']['generation']
    fig.suptitle(title+f'\nGold={s["answer"]} | Full={fg["text"]!r} | EarlyOFF={eg["text"]!r}',fontsize=10)
    save(fig,folder/'head_source_mass')


def main():
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    binding=execution();p=verify();completion=read(OUT/'completion.json')
    assert completion['status']=='complete' and completion['binding']==digest(binding)
    assert completion['hooks_clean'] and completion['parameters_unchanged']
    data=entries();generations=[];records=[];masses=[];vectors=[];patterns=[];cases=[]
    for entry in data:
        s=entry['sample'];check_reuse(entry);name=identity(s);bodies={};row={}
        for c in KEYS:
            b=valid_checkpoint(OUT/'checkpoints'/name/f'{c}.json',entry,c,binding);bodies[c]=b;g=b['generation'];row[c]=g
            assert g['prediction']==gqa_prediction(g['text']) and g['score']==int(g['prediction']==s['answer'])
            assert g['own_prefill'] and g['own_prefix'] and g['length']<=32
            assert np.isfinite(g['first_logp_loss']) and b['measurements']['bos_kv_exact']
            if c!='Full':assert g['edits']==(2 if CONDITIONS[c].get('early_off') else 0)
            assert b['audit']['decode_mask_edits']==0
            tag=dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],condition=c)
            records.append(dict(**tag,gold=s['answer'],answer=g['text'],prediction=g['prediction'],score=g['score'],
                first_token=g['token_ids'][0],first_token_flip=g['first_top1_flip'],first_logp_loss=g['first_logp_loss'],
                full_first_token_logp=g['full_first_token_logp'],first_logits_sha256=g['first_logits_sha256'],
                tokens=json.dumps(g['token_ids']),truncated=g['truncated'],reused_complete_answer=b['reused_generation'],
                temporal_scope='Full' if c=='Full' else 'original Prefill only; Decode native',
                p_minus_one=entry['mapping']['query'],question_token_count=len(entry['mapping']['question'])))
            for r in b['measurements']['layer_mass']:
                baseline=bodies['Full']['measurements']['layer_mass'][r['layer']]
                masses.append(dict(**tag,**r,**{'delta_'+source+'_pp':100*(r[source+'_mass']-baseline[source+'_mass'])
                    for source in ['bos','image','question','template']}))
            vectors += [dict(**tag,**r) for r in b['measurements']['vector_changes']]
        generations.append(row)
        patterns.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
            full_correct=row['Full']['score'],harm_conditions='|'.join(c for c in CONDITIONS if row['Full']['score'] and not row[c]['score']),
            rescued_conditions='|'.join(c for c in CONDITIONS if not row['Full']['score'] and row[c]['score']),
            wrong_conditions='|'.join(c for c in KEYS if not row[c]['score'])))
        if name in p['cases']:case_figures(entry,bodies);cases.append(name)
    assert len(records)==2880 and len(list((OUT/'checkpoints').glob('*/*.json')))==2880
    ids=[e['sample']['image_sha256'] for e in data]
    overall,interaction=summarize(generations,ids)
    taskwise=[];task_interactions=[]
    for t in old.TASKS:
        ix=[i for i,e in enumerate(data) if e['sample']['task']==t]
        a,b=summarize([generations[i] for i in ix],[ids[i] for i in ix],t)
        taskwise+=a;task_interactions+=b;print('Summarized',t,flush=True)
    overlap=intersections(generations)
    for filename,rows in [('per_question.csv',records),('accuracy_summary.csv',overall),('task_accuracy.csv',taskwise),
        ('source_interactions.csv',interaction),('task_source_interactions.csv',task_interactions),('layer_source_mass.csv',masses),
        ('hidden_query_changes.csv',vectors),('error_intersections.csv',overlap),('per_question_error_patterns.csv',patterns)]:
        write_csv(OUT/filename,rows)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    plots(overall,taskwise,interaction,masses,overlap)
    result=dict(status='complete',code_commit=binding['code_commit'],questions=240,image_clusters=len(set(ids)),
        new_complete_generations=2640,reused_full_complete_answers=240,summary=overall,source_interactions=interaction,
        cases=cases,completion=completion,source_sha256=binding['source_sha256'],
        interpretation='Mass, causal source contribution, compensation and semantic integration require separate claims')
    from .report import write_report
    write_report(result,taskwise,masses)
    result['artifact_sha256']={str(path.relative_to(OUT)):sha256(path) for path in sorted(OUT.rglob('*'))
        if path.suffix in ['.csv','.png','.pdf','.md','.npz']}
    atomic_json(OUT/'result.json',result)
    print(json.dumps(dict(status='complete',questions=240,summary=[{k:r[k] for k in ['condition','accuracy_pct','correct_to_wrong','wrong_to_correct','first_token_flips']} for r in overall],
        source_interactions=interaction),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
