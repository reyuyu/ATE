"""CPU-only, question-weighted paired image-cluster statistics and figures."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import OUT, MODELS, BUDGETS, SEEDS, TASKS, verify, read, write, sha, digest, compact_digest

LABELS={'qwen':'Qwen3-VL-8B','internvl':'InternVL3-8B','llava':'LLaVA-1.5-7B'}
COLORS={'qwen':'#b54a69','internvl':'#2c807d','llava':'#426fa4'}


def csv_write(name, rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def condition_names(p):
    if p==0:return ['Only Text']*3
    if p==1:return ['Full']*3
    return [f'p{p:.1f}_s{seed}' for seed in SEEDS]


@lru_cache(maxsize=8)
def bootstrap_weights(images):
    clusters=sorted(set(images));mapping={h:i for i,h in enumerate(clusters)}
    assignment=np.array([mapping[h] for h in images]);counts=np.bincount(assignment)
    rng=np.random.default_rng(20261102)
    weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=20000).astype(np.float32)
    denominators=weights@counts.astype(np.float32)
    return assignment,weights,denominators


def cluster_intervals(values, images):
    """Rows are questions, columns are paired quantities, already seed-averaged."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim==1:values=values[:,None]
    assert len(values)==len(images) and np.isfinite(values).all()
    assignment,weights,denom=bootstrap_weights(tuple(images))
    sums=np.zeros((weights.shape[1],values.shape[1]),dtype=np.float64)
    np.add.at(sums,assignment,values)
    draws=(weights@sums)/denom[:,None]
    return np.quantile(draws,[.025,.975],axis=0)


def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else -1)


def flags(text, tokens):
    trigrams=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return dict(contains_newline='\n' in text,contains_role_markup=any(v in text for v in ['USER:','ASSISTANT:','<|im_','<img>']),
        answer_prefix=text.lower().startswith(('answer:', 'the answer is')),
        repeated_token_trigrams=len(trigrams)-len(set(trigrams)))


def load():
    p=verify();samples=read(OUT/'sample_manifest.json')['samples'];configs=read(OUT/'layer_sets.json')['conditions']
    rows=[];records={};completion={}
    for model in MODELS:
        done=read(OUT/model/'complete.json');completion[model]=done
        assert done['passed'] and done['questions']==3000 and done['conditions']==14
        assert done['protocol_sha256']==sha(OUT/'protocol.json')
        assert done['model_weights_unchanged'] and done['hooks_clean'] and done['no_nan_inf_oom']
        assert read(OUT/model/'validation.json')['passed'] and not (OUT/model/'error.json').exists()
        histories=read(OUT/f'endpoint_bindings/{model}.json');byq={}
        for s in samples:
            key=s['task']+'__'+s['question_id'];r=read(OUT/model/'checkpoints'/(key+'.json'))
            assert r['complete'] and set(r['conditions'])==set(configs[model])
            assert r['sample_sha256']==compact_digest(s) and r['protocol_sha256']==sha(OUT/'protocol.json')
            assert r['code_commit']==done['code_commit'] and r['physical_gpu']==MODELS[model]['gpu']
            assert r['image_sha256']==s['image_sha256'] and r['history_binding_sha256']==digest(histories[s['question_id']])
            byq[s['question_id']]=r['conditions'];base=r['conditions']['Full'];textonly=r['conditions']['Only Text']
            for name,g in r['conditions'].items():
                c=configs[model][name]
                assert g['source']==('strict_historical_reuse' if c['seed'] is None else 'new_random_condition')
                assert g['correct']==int(g['text'].strip().rstrip('.').lower()==s['answer'])
                assert 1<=len(g['token_ids'])<=32
                assert np.isfinite([g['signed_logp_loss'],g['absolute_logp_change'],g['full_top1_logp'],g['off_full_top1_logp']]).all()
                assert g['absolute_logp_change']==abs(g['signed_logp_loss'])
                if g['source']=='new_random_condition':
                    a=g['audit'];assert a['rows']==MODELS[model]['layers']*len(g['token_ids'])
                    assert a['masked_rows']==len(c['off'])*len(g['token_ids']) and a['blocked_image_probability_max']==0
                    assert a['retained_image_probability_min']>0
                rows.append(dict(model=model,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                    image_sha256=s['image_sha256'],gold=s['answer'],condition=name,budget=c['p'],seed=c['seed'],
                    retained_layers=len(c['keep']),actual_layer_fraction=len(c['keep'])/MODELS[model]['layers'],
                    physical_gpu=r['physical_gpu'],input_binding=r['input_binding'],source=g['source'],
                    first_token_metrics_source=g.get('first_token_source',
                        'historical bound logits/metrics' if g['source']=='strict_historical_reuse' else 'new complete generation'),
                    prediction=g['prediction'],text=g['text'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],
                    full_correct=base['correct'],only_text_correct=textonly['correct'],
                    full_correct_to_wrong=int(base['correct']==1 and g['correct']==0),
                    full_wrong_to_correct=int(base['correct']==0 and g['correct']==1),
                    only_text_correct_to_wrong=int(textonly['correct']==1 and g['correct']==0),
                    only_text_wrong_to_correct=int(textonly['correct']==0 and g['correct']==1),
                    **{k:g[k] for k in ['full_first_token','changed_first_token','full_top1_logp','off_full_top1_logp',
                        'signed_logp_loss','absolute_logp_change','top1_flip']},
                    generated_text_changed=g['text']!=base['text'],sequence_changed=g['token_ids']!=base['token_ids'],
                    first_divergence_zero_based=first_difference(base['token_ids'],g['token_ids']),
                    generated_tokens=len(g['token_ids']),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],
                    **flags(g['text'],g['token_ids'])))
        assert len(byq)==3000;records[model]=byq
        generated=[g for rr in byq.values() for g in rr.values() if g['source']=='new_random_condition']
        assert len(generated)==36000
        completion[model]=dict(done,new_generation_seconds=sum(g['seconds'] for g in generated),
            verified_new_generations=len(generated),resumed=False)
        ledger_path=OUT/model/'resume_ledger.json'
        if ledger_path.exists():
            ledger=read(ledger_path)
            assert ledger['prior_new_generations']+done['new_generations']==36000
            ordered=sorted(samples,key=lambda s:(s['image_sha256'],s['task'],s['question_id']))
            unfinished=ordered[ledger['prior_complete_questions']]
            unfinished_name=unfinished['task']+'__'+unfinished['question_id']+'.json'
            assert unfinished_name in ledger['checkpoint_hashes_at_resume']
            for name,h in ledger['checkpoint_hashes_at_resume'].items():
                path=OUT/model/'checkpoints'/name
                # Only the single unfinished checkpoint may gain more conditions.
                if name==unfinished_name:continue
                assert sha(path)==h, 'A previously complete checkpoint changed during resume'
            completion[model].update(resumed=True,prior_new_generations=ledger['prior_new_generations'],
                resumed_new_generations=done['new_generations'],prior_formal_seconds_lower_bound=ledger['prior_progress']['seconds'])
        else:
            assert done['new_generations']==36000
    assert len(rows)==126000
    assert sum(r['source']=='new_random_condition' for r in rows)==108000
    csv_write('per_question.csv',rows)
    return samples,records,completion


def statistics(samples, records):
    summary=[];seedrows=[];termination=[]
    for model in MODELS:
        allrows=records[model]
        for task in ['all']+list(TASKS):
            ss=[s for s in samples if task=='all' or s['task']==task]
            ids=[s['question_id'] for s in ss];images=[s['image_sha256'] for s in ss]
            full=np.array([allrows[q]['Full']['correct'] for q in ids],dtype=float)
            only=np.array([allrows[q]['Only Text']['correct'] for q in ids],dtype=float)
            entries=[];arrays=[]
            for budget in BUDGETS:
                names=condition_names(budget)
                correct=np.array([[allrows[q][k]['correct'] for k in names] for q in ids],dtype=float)
                signed=np.array([[allrows[q][k]['signed_logp_loss'] for k in names] for q in ids])
                absolute=np.array([[allrows[q][k]['absolute_logp_change'] for k in names] for q in ids])
                flip=np.array([[allrows[q][k]['top1_flip'] for k in names] for q in ids])
                score=correct.mean(1);seedscores=100*correct.mean(0)
                metrics=np.column_stack([100*score,100*(score-full),100*(score-only),signed.mean(1),absolute.mean(1),100*flip.mean(1)])
                arrays.append(metrics)
                entries.append(dict(model=model,task=task,budget=budget,n=len(ids),image_clusters=len(set(images)),
                    retained_layers=int(np.floor(budget*MODELS[model]['layers']+.5)),
                    actual_layer_fraction=int(np.floor(budget*MODELS[model]['layers']+.5))/MODELS[model]['layers'],
                    accuracy_percent=float(metrics[:,0].mean()),seed_min_accuracy=float(seedscores.min()),
                    seed_max_accuracy=float(seedscores.max()),delta_full_pp=float(metrics[:,1].mean()),
                    gain_only_text_pp=float(metrics[:,2].mean()),
                    mean_signed_logp_loss=float(metrics[:,3].mean()),mean_absolute_logp_change=float(metrics[:,4].mean()),
                    top1_flip_percent=float(metrics[:,5].mean()),
                    mean_seed_correct_to_wrong=float(((full[:,None]==1)&(correct==0)).sum(0).mean()),
                    mean_seed_wrong_to_correct=float(((full[:,None]==0)&(correct==1)).sum(0).mean()),
                    mean_seed_only_text_harmed=float(((only[:,None]==1)&(correct==0)).sum(0).mean()),
                    mean_seed_only_text_corrected=float(((only[:,None]==0)&(correct==1)).sum(0).mean())))
                for column,(seed,name) in enumerate(zip(SEEDS,names)):
                    gs=[allrows[q][name] for q in ids]
                    seedrows.append(dict(model=model,task=task,budget=budget,seed=seed,shared_endpoint=budget in (0.,1.),
                        n=len(ids),accuracy_percent=float(seedscores[column]),delta_full_pp=float(100*(correct[:,column]-full).mean()),
                        gain_only_text_pp=float(100*(correct[:,column]-only).mean()),
                        correct_to_wrong=int(((full==1)&(correct[:,column]==0)).sum()),
                        wrong_to_correct=int(((full==0)&(correct[:,column]==1)).sum()),
                        only_text_harmed=int(((only==1)&(correct[:,column]==0)).sum()),
                        only_text_corrected=int(((only==0)&(correct[:,column]==1)).sum()),
                        mean_signed_logp_loss=float(signed[:,column].mean()),mean_absolute_logp_change=float(absolute[:,column].mean()),
                        top1_flip_percent=float(100*flip[:,column].mean()),
                        text_changed=sum(g['text']!=allrows[q]['Full']['text'] for q,g in zip(ids,gs)),
                        sequence_changed=sum(g['token_ids']!=allrows[q]['Full']['token_ids'] for q,g in zip(ids,gs)),
                        mean_generated_tokens=float(np.mean([len(g['token_ids']) for g in gs])),
                        eos=sum(g['eos'] for g in gs),truncated=sum(g['truncated'] for g in gs),empty=sum(g['empty'] for g in gs),
                        role_markup=sum(flags(g['text'],g['token_ids'])['contains_role_markup'] for g in gs),
                        repeated_trigram_outputs=sum(flags(g['text'],g['token_ids'])['repeated_token_trigrams']>0 for g in gs)))
            intervals=cluster_intervals(np.column_stack(arrays),images)
            for j,entry in enumerate(entries):
                for col,name in enumerate(['accuracy','delta_full','gain_only_text','signed_loss','absolute_change','top1_flip']):
                    entry[name+'_ci95_low']=float(intervals[0,j*6+col]);entry[name+'_ci95_high']=float(intervals[1,j*6+col])
                summary.append(entry)
        print('STATISTICS',model,'complete',flush=True)
    csv_write('budget_summary.csv',[r for r in summary if r['task']=='all'])
    csv_write('task_budget_summary.csv',[r for r in summary if r['task']!='all'])
    csv_write('per_seed_summary.csv',seedrows)
    csv_write('termination_summary.csv',[{k:r[k] for k in ['model','task','budget','seed','shared_endpoint','n','eos','truncated','empty',
        'role_markup','repeated_trigram_outputs','text_changed','sequence_changed','mean_generated_tokens']} for r in seedrows])
    return summary,seedrows


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=260,bbox_inches='tight')
    plt.close(fig)


def figures(summary, seeds):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    rows=lambda model,task='all':sorted([r for r in summary if r['model']==model and r['task']==task],key=lambda r:r['budget'])
    def curve(ax,model,metric,low=None,high=None,task='all'):
        r=rows(model,task);x=[t['budget'] for t in r];y=[t[metric] for t in r]
        ax.plot(x,y,'o-',color=COLORS[model],label=LABELS[model],ms=4)
        if low:ax.fill_between(x,[t[low] for t in r],[t[high] for t in r],color=COLORS[model],alpha=.12)
        ax.set_xticks(BUDGETS,['0%','20%','40%','60%','80%','100%']);ax.grid(axis='y',alpha=.15)
        ax.set_xlabel('Allowed direct Image-access layer budget')
    fig,(ax,seedax)=plt.subplots(1,2,figsize=(12.8,4.3),layout='constrained',sharey=True)
    for model in MODELS:
        curve(ax,model,'accuracy_percent','accuracy_ci95_low','accuracy_ci95_high')
        r=rows(model);curve(seedax,model,'accuracy_percent')
        seedax.vlines(BUDGETS,[t['seed_min_accuracy'] for t in r],[t['seed_max_accuracy'] for t in r],color=COLORS[model],alpha=.7,lw=2)
        for seed,marker in zip(SEEDS,['x','+','1']):
            sr=sorted([t for t in seeds if t['model']==model and t['task']=='all' and t['seed']==seed],key=lambda t:t['budget'])
            seedax.scatter([t['budget'] for t in sr],[t['accuracy_percent'] for t in sr],color=COLORS[model],marker=marker,s=35,alpha=.8)
    ax.set_ylabel('GQA complete-answer Accuracy (%)');ax.legend(frameon=False)
    ax.set_title('Seed means + image-cluster bootstrap 95% CI',fontsize=10)
    seedax.set_title('Seed points + min-max range (not a 95% CI)',fontsize=10)
    for seed,marker in zip(SEEDS,['x','+','1']):seedax.scatter([],[],color='#444444',marker=marker,label=str(seed),s=35)
    handles,labels=seedax.get_legend_handles_labels()
    seedax.legend(handles[-3:],labels[-3:],frameon=False,fontsize=8,loc='lower right')
    fig.suptitle('P-1 + Decode Image access; other Prefill queries remain native',fontsize=11)
    save(fig,'figure1_accuracy_budget')
    fig,axes=plt.subplots(1,2,figsize=(11.6,4),layout='constrained')
    for ax,metric,stem,title in [(axes[0],'delta_full_pp','delta_full','Relative to Full'),(axes[1],'gain_only_text_pp','gain_only_text','Relative to Only Text')]:
        for model in MODELS:curve(ax,model,metric,stem+'_ci95_low',stem+'_ci95_high')
        ax.axhline(0,color='#555555',ls='--',lw=1);ax.set_ylabel('Accuracy difference (pp)');ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.suptitle('P-1 + Decode source masks; paired image-cluster 95% intervals',fontsize=11)
    save(fig,'figure2_paired_accuracy_differences')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,task in zip(axes.flat,TASKS):
        for model in MODELS:curve(ax,model,'accuracy_percent','accuracy_ci95_low','accuracy_ci95_high',task)
        ax.set_title(task,fontweight='bold' if task=='CompareAttr' else 'normal');ax.set_ylabel('Accuracy (%)')
    axes[0,0].legend(frameon=False,fontsize=8);save(fig,'figure3_six_task_budgets')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric,stem,title in [(axes[0],'mean_signed_logp_loss','signed_loss','Full Top1 signed LogP loss (nat)'),
        (axes[1],'mean_absolute_logp_change','absolute_change','Full Top1 absolute LogP change (nat)'),
        (axes[2],'top1_flip_percent','top1_flip','First-answer Top1 flip (%)')]:
        for model in MODELS:curve(ax,model,metric,stem+'_ci95_low',stem+'_ci95_high')
        ax.axhline(0,color='#555555',lw=.8,ls='--');ax.set_ylabel(title)
    axes[0].legend(frameon=False,fontsize=8);save(fig,'figure4_first_token_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.5),layout='constrained')
    for ax,metric,title,cm in [(axes[0],'delta_full_pp','Accuracy change vs Full (pp)','RdBu'),
        (axes[1],'top1_flip_percent','First-answer Top1 flip (%)','magma')]:
        values=np.array([[r[metric] for r in rows(model)] for model in MODELS])
        options=dict(cmap=cm,aspect='auto')
        if metric=='delta_full_pp':
            bound=max(float(np.abs(values).max()),.01);options.update(vmin=-bound,vmax=bound)
        im=ax.imshow(values,**options);fig.colorbar(im,ax=ax,shrink=.8)
        ax.set_xticks(range(6),['0%','20%','40%','60%','80%','100%']);ax.set_yticks(range(3),[LABELS[m] for m in MODELS]);ax.set_title(title)
        for i in range(3):
            for j in range(6):
                rgba=im.cmap(im.norm(values[i,j]));luminance=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j,i,f'{values[i,j]:+.2f}' if metric=='delta_full_pp' else f'{values[i,j]:.1f}',ha='center',va='center',color='black' if luminance>.55 else 'white',fontsize=9)
    save(fig,'figure5_budget_quality_summary')


def report(summary,seeds,completed):
    overall=[r for r in summary if r['task']=='all']
    lines=['# GQA Random Visual Access Budget Scaling','',
        '三模型各复用原六任务×500题，共3000题、2409个图片SHA簇。固定历史revision、官方Prompt/Processor和评分；LLaVA FP16/eager，Qwen/InternVL BF16/eager，Qwen DeepStack原生，InternVL max_num=1、256视觉Token、logits_to_keep=1。',
        '仅原始P−1与真实增量Decode Query限制原始Image Key；其他Prefill Query完整原生。Softmax前−inf Image Mask不是Text Mask、o_proj OFF、Relay OFF、Block跳过或KV删除，不代表FLOPs、显存或延迟节省。',
        '0% Only Text仍然有原始图像和其他Prefill Query形成的视觉中继，不是无图像输入或纯文本模型。预算效应同时包含直接视觉路径变化与Softmax概率重新分配，不能把分数差当作独立视觉信息贡献。',
        '三个种子20261102/03/04各产生一个全层排列；每预算截取前floor(p×N+0.5)层，严格嵌套。L0不强制保留。0%与100%为共享端点；随机排列及逐种子层集合见layer_sets.json。',
        '历史端点逐题核验Question/Image SHA、输入Prompt/动态视觉映射、固定模型、生成协议和Logits绑定；Full/Only Text完整答案复用。缺失LLaVA Only Text首Token指标仅补一次Prefill并要求历史Logits哈希精确相同，不重复完整生成。',
        '', '## 总体结果', '|模型|预算|保留层数|Accuracy %（三种子均值）|种子范围|ΔFull pp [95%CI]|增益vs0% pp [95%CI]|',
        '|---|---:|---:|---:|---|---|---|']
    for r in overall:
        lines.append(f"|{LABELS[r['model']]}|{r['budget']:.0%}|{r['retained_layers']}|{r['accuracy_percent']:.3f}|[{r['seed_min_accuracy']:.3f},{r['seed_max_accuracy']:.3f}]|{r['delta_full_pp']:+.3f} [{r['delta_full_ci95_low']:+.3f},{r['delta_full_ci95_high']:+.3f}]|{r['gain_only_text_pp']:+.3f} [{r['gain_only_text_ci95_low']:+.3f},{r['gain_only_text_ci95_high']:+.3f}]|")
    lines+=['','## 核心结论',
        '三模型总体Full点估计均高于0% Only Text，但绝对差异只有0.50–0.70pp；这批短答案任务在保留完整Prefill视觉中继时，直接读图层预算的总体影响较小，并非所有题目无损。InternVL的六个总体均值递增，Qwen与LLaVA出现局部反向变化，不能宣称通用单调预算定律。',
        'CompareAttr从0%到Full：Qwen 82.0%→85.4%（+3.4pp），InternVL 88.8%→90.4%（+1.6pp），LLaVA 87.6%→89.2%（+1.6pp）。但Qwen ChooseRel为87.2%→86.2%（−1.0pp），是更多视觉访问不必提高所有任务Accuracy的反例。以上均为探索性点估计，区间见分任务CSV。',
        '首Token绝对LogP变化通常随预算上升而下降，但InternVL在60%→80%发生反弹（0.07335→0.07888 nat）；此时Top1翻转也由4.589%升至4.689%。不能把总体Accuracy微升等同于每个预测分布都更接近Full。',
        '0%相对Full的对→错/错→对分别为Qwen 43/27、InternVL 38/17、LLaVA 48/33。80%三种子均值仍分别为15.00/8.33、13.00/7.33、15.00/13.33；净分数较小隐藏了双向翻转，均值非整数并不代表新的独立题数。',
        '', '## 曲线与任务差异']
    for model in MODELS:
        rr=sorted([r for r in overall if r['model']==model],key=lambda r:r['budget'])
        d=np.diff([r['accuracy_percent'] for r in rr]);loss=np.diff([r['mean_absolute_logp_change'] for r in rr])
        lines.append(f"{LABELS[model]}：预算递增5段中Accuracy上升/不变/下降为{sum(d>1e-10)}/{sum(abs(d)<=1e-10)}/{sum(d< -1e-10)}；首Token绝对LogP变化下降/不变/上升为{sum(loss< -1e-10)}/{sum(abs(loss)<=1e-10)}/{sum(loss>1e-10)}。不拟合或强制单调性，有限3个排列不能代表所有随机层集合。")
        compare=sorted([r for r in summary if r['model']==model and r['task']=='CompareAttr'],key=lambda r:r['budget'])
        lines.append('CompareAttr各预算Accuracy：'+', '.join(f"{r['budget']:.0%}={r['accuracy_percent']:.2f}%" for r in compare)+'；其余五任务见task_budget_summary.csv。')
        termination=[r for r in seeds if r['model']==model and r['task']=='all' and not r['shared_endpoint']]
        lines.append(f"随机条件累计截断{sum(r['truncated'] for r in termination)}、空答案{sum(r['empty'] for r in termination)}、角色标记异常{sum(r['role_markup'] for r in termination)}（分母36000生成，不是独立图片数）；逐条件EOS/截断/复读/长度和文本变化见termination_summary.csv。")
    lines+=['','## 统计口径与边界',
        '20,000次图片SHA整簇配对Bootstrap，保留原题目加权。先在同题内对三个种子求均值，再整簇重采样；同图跨任务问题一起重采样。种子最小—最大范围及各自点估计另列，不把三个种子当作独立图片。端点共享一条生成，绘制三个重合端点不增加样本量。',
        '置信区间条件于本次固定的三个层排列，反映图片抽样不确定性；种子范围只描述三套层集合的有限波动，不是层随机性总体95%区间。普通区间未经多重比较校正；所有层段/预算/分任务结果均探索性，不作独立非劣效或严格无损认定。',
        '三个模型各用其原生Prompt、Processor、视觉Token表示与精度；跨模型同时存在这些差异。相似曲线只能描述这三个原生系统，不能单独归因于架构，也不能推广为所有VLM的规律。历史冻结题目的采样限制仍保留，复用不构成新的独立确认集。',
        'Full原生Top1固定为每题目标，用全词表FP32 log_softmax计算signed LogP loss与absolute change。不是Gold答案LogP，也不把首Token翻转直接当作完整答案Accuracy变化。完整答案按历史GQA相同strip/rstrip句点/lower精确评分。',
        '对→错和错→对均保留：per_seed_summary.csv为各排列的整数题数；budget_summary.csv中的mean_seed_*为三种子计数均值，可能非整数，不能误当为独立题数。逐题完整序列、分叉、EOS、截断、空答案及格式筛查在per_question.csv。',
        '格式/重复标志为描述性筛查，不自动判定模型答案错误；不删除异常输出、不改评分。六任务曲线和负向/非单调变化全部保留。横轴为名义预算，实际层比例由整数层数决定并单列actual_layer_fraction。',
        '', '## 验证与执行',
        '9项CPU回归测试；各模型真实GPU Full/No-op所有有效Logits及Token精确一致，历史Full/Only Text完整生成精确复现。所有13个干预条件均校验非P−1 Prefill不变、Image概率严格为0、保留层访问正值、独立Cache/真实Decode位置/Hook异常清理与输入哈希。正式每次生成继续检查Image行、Cache和非有限数值。',
        '首次Smoke发现LLaVA历史全位置LM-head与HF默认最后位置LM-head的FP16形状差异；在正式运行前显式logits_to_keep=0恢复历史精确绑定，无容差修改。另将模型加载从inference_mode移至no_grad以保留参数版本审计，前向仍为inference_mode。两次启动前失败及更正留在validation_attempt_1/2，均未产生正式生成。',
        '物理GPU0=Qwen、GPU2=InternVL、GPU3=LLaVA；GPU1未使用。各模型36000次新增完整生成的逐条件计时之和（不含加载、Smoke、首Token补算、断点读写及停机）：'+', '.join(f"{LABELS[m]} {c['new_generation_seconds']/3600:.2f}h" for m,c in completed.items())+'。',
        'LLaVA首次进程在2929题完整保存、下一题8个随机条件保存后意外退出，未留下Python异常，退出原因未证实；完整审计已归档execution_attempt_1。用同一冻结Commit、原GPU3重新通过Smoke后，仅补844次生成；35156+844=36000。已完成检查点逐文件哈希不变，未重复计数。complete.json的new_generations/formal_seconds为最后一次进程的计数与时长，累计计数和逐条件计时见execution_summary.json。',
        '执行代码Commit：'+next(iter(completed.values()))['code_commit']+'；最终图片/模型/源码哈希与GPU释放状态见delivery_audit.json。',
        '', '## 文件',
        'figure1_accuracy_budget、figure2_paired_accuracy_differences、figure3_six_task_budgets、figure4_first_token_sensitivity、figure5_budget_quality_summary均有PNG/PDF。原始绘图输入为budget_summary.csv、task_budget_summary.csv、per_seed_summary.csv。',
        '本轮只含原3000题GQA和固定三种子，不自动扩展COCO、层组合、训练或算法开发。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def main():
    assert read(OUT/'cpu_validation.json')['passed'] and read(OUT/'analysis_cpu_validation.json')['passed']
    start=time.time();samples,records,completed=load();summary,seeds=statistics(samples,records)
    write(OUT/'execution_summary.json',completed)
    figures(summary,seeds);report(summary,seeds,completed)
    write(OUT/'analysis_validation.json',dict(passed=True,models=3,questions_per_model=3000,image_clusters=2409,
        conditions=14,rows=126000,new_complete_generations=108000,historical_complete_generations=18000,
        repetitions=20000,seed_average_before_cluster_resampling=True,
        source_sha256=sha(Path(__file__)),seconds=time.time()-start))
    print('All three complete models analyzed; 126000 condition records, five PNG/PDF figures.',flush=True)


if __name__=='__main__':main()
