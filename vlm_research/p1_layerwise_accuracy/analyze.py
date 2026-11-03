"""Independent token decoding/scoring, paired image clusters, and unsmoothed figures."""
import csv
import ast
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

from .common import *


def historical_correlations(x,y):
    tree=ast.parse((OLD/'paper_rq1_single_layer_gqa6/analyze.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='descriptive_correlations')
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_descriptive_correlations','exec'),scope)
    return scope['descriptive_correlations'](x,y)


def table(path,rows):
    assert rows
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def bootstrap(values,images):
    _,inverse=np.unique(images,return_inverse=True)
    n=int(inverse.max())+1
    sums=np.zeros((n,values.shape[1]));np.add.at(sums,inverse,values)
    counts=np.bincount(inverse,minlength=n)
    rng=np.random.default_rng(SEED);draws=np.empty((20000,values.shape[1]))
    for start in range(0,20000,128):
        stop=min(20000,start+128)
        w=rng.multinomial(n,np.full(n,1/n),size=stop-start)
        draws[start:stop]=(w@sums)/(w@counts)[:,None]
    assert np.isfinite(draws).all()
    return draws


def subset_rows(kind,samples,generations,label):
    cs=conditions(kind);layers=MODELS[kind]['layers']
    values=np.array([[generations[(key(s),c)]['correct'] for c in cs] for s in samples],dtype=float)
    draws=bootstrap(values*100,[s['image_sha256'] for s in samples])
    delta=draws-draws[:,0,None]
    ordinary=np.quantile(delta,[.025,.975],axis=0)
    corrected=np.quantile(delta,[.05/(2*layers),1-.05/(2*layers)],axis=0)
    scores=np.quantile(draws,[.025,.975],axis=0)
    results=[]
    for j,c in enumerate(cs):
        gs=[generations[(key(s),c)] for s in samples]
        base=[generations[(key(s),cs[0])] for s in samples]
        result=dict(task=label,condition=c,layer=j-1 if j else -1,questions=len(samples),
            images=len({s['image_sha256'] for s in samples}),correct=int(values[:,j].sum()),
            accuracy_pct=float(values[:,j].mean()*100),delta_accuracy_pp=float((values[:,j]-values[:,0]).mean()*100),
            ci95_low=float(ordinary[0,j]),ci95_high=float(ordinary[1,j]),
            bonferroni95_low=float(corrected[0,j]),bonferroni95_high=float(corrected[1,j]),
            accuracy_ci95_low=float(scores[0,j]),accuracy_ci95_high=float(scores[1,j]),
            correct_to_wrong=int(((values[:,0]==1)&(values[:,j]==0)).sum()),
            wrong_to_correct=int(((values[:,0]==0)&(values[:,j]==1)).sum()),
            text_changes=sum(g['text']!=f['text'] for g,f in zip(gs,base)),
            first_token_flips=sum(g['token_ids'][0]!=f['token_ids'][0] for g,f in zip(gs,base)),
            eos=sum(g['eos'] for g in gs),truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs),repeated=sum(g['repeated'] for g in gs))
        results.append(result)
    return results,draws,delta


def figures(folder,rows,tasks,joined):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':230})
    model=folder.name.upper();x=np.arange(len(rows)-1);rs=rows[1:]
    fig,ax=plt.subplots(figsize=(7.1,3.5),layout='constrained')
    y=np.array([r['delta_accuracy_pp'] for r in rs]);lo=np.array([r['bonferroni95_low'] for r in rs]);hi=np.array([r['bonferroni95_high'] for r in rs])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',markersize=3,color='#087e8b',elinewidth=.7,capsize=2,lw=1)
    ax.axhline(0,color='.3',lw=.8);ax.set(xlabel='Language decoder layer (0-based)',ylabel='Accuracy change vs Native Full (pp)',
        title=f'{model}: Original P-1 Attention-update Ablation')
    ax.text(.99,.03,f'3,000 questions; Bonferroni-{len(rs)} paired 95% CI',transform=ax.transAxes,ha='right',fontsize=8)
    ax.grid(alpha=.15)
    for ext in ('png','pdf'):fig.savefig(folder/f'figure_A_accuracy_profile.{ext}')
    plt.close(fig)
    task_names=sorted({r['task'] for r in tasks});lookup={(r['task'],r['layer']):r for r in tasks}
    heat=np.array([[lookup[(t,i)]['delta_accuracy_pp'] for i in x] for t in task_names])
    vmax=max(1.,float(np.abs(heat).max()))
    fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained')
    im=ax.imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
    ax.set(yticks=np.arange(6),yticklabels=task_names,xticks=np.arange(0,len(x),2),
        xlabel='Language decoder layer (0-based)',title=f'{model}: Six-task Accuracy Impact (500 questions/task)')
    fig.colorbar(im,ax=ax,label='Raw accuracy change (pp)',shrink=.8)
    for ext in ('png','pdf'):fig.savefig(folder/f'figure_B_task_heatmap.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(7,7),sharex=True,layout='constrained')
    for ax,column,label,color in zip(axes,['historical_signed_loss','historical_absolute_change','historical_top1_flip_pct','delta_accuracy_pp'],
            ['Signed LogP loss (nat)','Absolute LogP change (nat)','First-token flips (%)','Answer Accuracy change (pp)'],
            ['#b24745','#386cb0','#b28a16','#087e8b']):
        ax.plot(x,[r[column] for r in joined],'o-',color=color,markersize=2.5,lw=1)
        ax.set_ylabel(label,fontsize=8);ax.axhline(0,color='.5',lw=.5);ax.grid(alpha=.12)
    axes[0].set_title(f'{model}: Historical First-token Sensitivity vs Complete-answer Accuracy')
    axes[-1].set_xlabel('Language decoder layer (0-based)')
    for ext in ('png','pdf'):fig.savefig(folder/f'figure_C_first_token_vs_accuracy.{ext}')
    plt.close(fig)


def analyze(kind):
    from tokenizers import Tokenizer
    started=time.time();p=verify(kind);folder=OUT/kind
    samples=read(folder/'sample_manifest.json')['samples'];shards=read(folder/'shards.json')['keys']
    index={key(s):s for s in samples};tokenizer=Tokenizer.from_file(str(model_path(kind)/'tokenizer.json'))
    eos=read(model_path(kind)/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    generations={};flat=[];hashes=[];origins={};devices={}
    for shard,keys in enumerate(shards):
        smoke=read(folder/f'shard{shard}/smoke.json');done=read(folder/f'shard{shard}/complete.json')
        assert smoke['passed'] and done['passed'] and smoke['binding']==done['binding']==digest(p)
        paths=sorted((folder/f'shard{shard}/checkpoints').glob('*/*.json'))
        assert len(paths)==len(keys)*len(conditions(kind))==done['conditions']
        for path in paths:
            r=read(path);s=index[key(r)];c=r['condition'];g=r['generation']
            validate_checkpoint(r,p,s,c)
            assert key(r) in keys and (key(r),c) not in generations
            assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
            assert g['prediction']==g['text'].rstrip('.').lower()
            assert g['correct']==int(g['prediction']==s['answer'].lower())
            assert len(g['token_ids'])<=32 and g['eos']==(g['token_ids'][-1] in eos)
            assert g['truncated']==(len(g['token_ids'])==32 and not g['eos'])
            devices.setdefault(s['image_sha256'],r['gpu_uuid']);assert devices[s['image_sha256']]==r['gpu_uuid']
            if r['origin']=='NEW_GENERATION':
                assert g['decode_all_native'] and g['all_logits_finite'] and g['cache_initial_empty']
                assert sum(e['active'] for e in g['intervention_events'])==1
                assert g['intervention_events'][0]['step']==0
                assert all(not e['active'] for e in g['intervention_events'][1:])
                assert all(e['layer']==layer_of(kind,c) and e['non_target_exact'] for e in g['intervention_events'])
                assert len(g['all_step_logits_sha256'])==len(g['token_ids'])
                assert len(g['cache_lengths'])==MODELS[kind]['layers'] and len(set(g['cache_lengths']))==1
            else:
                assert g['full_first_logits_replayed_exact']
            origins[r['origin']]=origins.get(r['origin'],0)+1
            generations[(key(r),c)]=g
            hashes.append(dict(path=str(path.relative_to(folder)),sha256=sha(path)))
    assert len(generations)==3000*len(conditions(kind)) and len(devices)==2409
    for s in samples:
        f=generations[(key(s),'Native Full')]
        for c in conditions(kind):
            g=generations[(key(s),c)]
            flat.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                condition=c,layer=layer_of(kind,c),gold=s['answer'],full_text=f['text'],text=g['text'],
                full_token_ids=str(f['token_ids']),token_ids=str(g['token_ids']),full_correct=f['correct'],correct=g['correct'],
                correct_to_wrong=int(f['correct']==1 and g['correct']==0),wrong_to_correct=int(f['correct']==0 and g['correct']==1),
                first_token_flip=int(f['token_ids'][0]!=g['token_ids'][0]),text_changed=int(f['text']!=g['text']),
                eos=g['eos'],truncated=g['truncated'],empty=g['empty'],repeated=g['repeated'],generated_tokens=len(g['token_ids'])))
    summary,draws,delta=subset_rows(kind,samples,generations,'overall')
    task_rows=[]
    for task in sorted({s['task'] for s in samples}):
        tr,_,_=subset_rows(kind,[s for s in samples if s['task']==task],generations,task);task_rows.extend(tr)
    table(folder/'accuracy_summary.csv',summary);table(folder/'layer_paired_comparisons.csv',summary[1:])
    table(folder/'task_layer_accuracy.csv',task_rows);table(folder/'per_question.csv',flat)
    table(folder/'harm_cases.csv',[r for r in flat if r['correct_to_wrong']])
    table(folder/'recovered_cases.csv',[r for r in flat if r['wrong_to_correct']])
    np.savez_compressed(folder/'bootstrap_draws.npz',accuracy_pct=draws,delta_accuracy_pp=delta)
    first={r['layer']:r for r in read(folder/'historical_first_token_summary.json') if r['task']=='overall'}
    joined=[dict(layer=r['layer'],relative_depth=r['layer']/(MODELS[kind]['layers']-1),
        delta_accuracy_pp=r['delta_accuracy_pp'],accuracy_harm_pp=-r['delta_accuracy_pp'],
        historical_signed_loss=first[r['layer']]['signed_loss'],historical_absolute_change=first[r['layer']]['absolute_change'],
        historical_top1_flip_pct=first[r['layer']]['top1_flip_rate']*100) for r in summary[1:]]
    table(folder/'first_token_vs_accuracy.csv',joined)
    correlations=[]
    for metric in ('historical_signed_loss','historical_absolute_change','historical_top1_flip_pct'):
        x=[r[metric] for r in joined];y=[r['accuracy_harm_pp'] for r in joined]
        pearson,spearman=historical_correlations(x,y)
        correlations.append(dict(metric=metric,pearson=pearson,spearman=spearman,
            unit_of_analysis='layer; overlapping representations; exploratory, not independent images'))
    table(folder/'first_token_accuracy_correlations.csv',correlations)
    figures(folder,summary,task_rows,joined)
    most=min(summary[1:],key=lambda r:r['delta_accuracy_pp'])
    supported=[r['layer'] for r in summary[1:] if r['bonferroni95_high']<0]
    cancelled=[r for r in summary[1:] if r['delta_accuracy_pp']==0 and r['correct_to_wrong']>0]
    peak_abs=max(joined,key=lambda r:r['historical_absolute_change'])['layer']
    peak_signed=max(joined,key=lambda r:r['historical_signed_loss'])['layer']
    # Compare only separately labeled, already measured writing experiments.
    writing_commit='afddd2a9972e6decddbb535f4b524225e52eba0e' if kind=='llava' else '96a66a30bc81db9ffe5c3ed77716673a81aaaa01'
    writing_name='vlm_research/results/'+('gqa3000_visual_writing_necessity/condition_summary.csv' if kind=='llava' else 'qwen3vl_visual_writing_necessity_3000/layer_paired_comparisons.csv')
    try:
        writing=__import__('subprocess').check_output(['git','show',writing_commit+':'+writing_name],cwd=ROOT,text=True)
        (folder/'historical_writing_reference.csv').write_text(writing)
        wr=list(csv.DictReader(io.StringIO(writing)))
        drops=[r for r in wr if str(r.get('condition','')).startswith('Drop-L')]
        delta_name=next(k for k in ('delta_accuracy_pp','delta_acc_pp','delta_pp') if k in drops[0])
        wp=min(drops,key=lambda r:float(r[delta_name]))
        separation='不同' if wp['condition']!='Drop-L'+str(most['layer']) else '在最大点估计处重合'
        writing_note=f"历史R→Image写入Accuracy最大点估计损害为{wp['condition']}（{float(wp[delta_name]):+.4f} pp），本轮P−1整个Attention更新最差为L{most['layer']}。两种干预的关键层{separation}，但不能据此认定因果模块分离。历史原始曲线单独保存；其All-Writer不是本轮Native Full，不混作配对基线。"
    except __import__('subprocess').CalledProcessError:
        writing_note='历史R→Image曲线的结果文件名与预期不一致；不能把它冒充本轮P−1整个Attention消融。'
    text=[f'# {kind.upper()}：GQA3000 P−1单层Attention更新的完整答案Accuracy', '',
        f'**COMPLETE：{len(generations):,}个条件标签，3000题，2409图片SHA簇；六任务各500题。**', '',
        f"Native Full：{summary[0]['correct']}/3000，{summary[0]['accuracy_pct']:.4f}%。仅原始Prefill P−1在一个层的o_proj输出置零；后续Decode完全原生。",'',
        f"最大点估计损害：L{most['layer']}，Accuracy {most['accuracy_pct']:.4f}%，ΔAcc {most['delta_accuracy_pp']:+.4f} pp；普通95%CI [{most['ci95_low']:.4f},{most['ci95_high']:.4f}]，Bonferroni校正95%CI [{most['bonferroni95_low']:.4f},{most['bonferroni95_high']:.4f}] pp；原对→错{most['correct_to_wrong']}，原错→对{most['wrong_to_correct']}。",'',
        f'校正后区间整体低于0的损害层：{supported}。未校正六任务结果仅作描述性分析。',
        f'历史首Token绝对LogP变化峰L{peak_abs}，Signed Loss峰L{peak_signed}；本轮Accuracy最差层L{most["layer"]}。不能把历史首Token指标重标成完整答案Accuracy。',
        f'净Accuracy零变化但存在相反对错翻转的层：{[r["layer"] for r in cancelled]}。全部损害和纠正案例均保留。','',
        '## 完整逐层结果','', '| 层 | Accuracy % | ΔAcc pp | 普通95%CI | 校正95%CI | 损害/纠正 |', '| --- | ---: | ---: | --- | --- | --- |']
    for r in summary[1:]:
        text.append(f"| L{r['layer']} | {r['accuracy_pct']:.3f} | {r['delta_accuracy_pp']:+.3f} | [{r['ci95_low']:.3f},{r['ci95_high']:.3f}] | [{r['bonferroni95_low']:.3f},{r['bonferroni95_high']:.3f}] | {r['correct_to_wrong']}/{r['wrong_to_correct']} |")
    text+=['','## 审计与统计边界','',f'历史标签复用/新增数量：{origins}。所有复用题的输入和来源SHA在GPU前冻结；每题Native Full首Token完整词表Logits在本轮同卡严格复现。',
        '两张GPU各三题真实Smoke通过：同形状Native/No-op全部有效Logits torch.equal；单层单行置零、其他行不变；Decode不干预；全部MLP、真实KV、视觉特征与正常/异常Hook清理检查通过。LLaVA旧480题Smoke复现相同首Token向量与生成Token。',
        'LLaVA沿用历史完整Prefill LM Head（logits_to_keep=0）；Qwen沿用原生logits_to_keep=1。两个模型各自原始Prompt、精度及动态视觉布局不同。',
        f'20,000次Image SHA整簇配对Bootstrap，seed={SEED}，题目等权。普通双侧95%CI；{MODELS[kind]["layers"]}项Bonferroni校正。极端分位数只有有限尾部重复数，存在Monte Carlo误差。未预设非劣效结论；区间跨0或净ACC为0不等于安全省略。',
        writing_note,
        '本轮为P−1整个Attention更新消融，不是R→Image写入、Image Key屏蔽、Decode消融、整层跳过或加速。不能据此宣称独立视觉读取模块或实际计算可安全省略。',
        '## 首Token关联（探索性）','']
    text += [f"- {r['metric']}与完整Accuracy损害：Pearson {r['pearson']:.4f}，Spearman {r['spearman']:.4f}。" for r in correlations]
    text += ['',f'执行源码Commit：`{p["code_commit"]}`。两卡正式执行分钟：'+', '.join(f"{read(folder/f'shard{i}/complete.json')['elapsed_seconds']/60:.2f}" for i in (0,1))+'. GPU进程已退出，见gpu_release.json。']
    (folder/'report_zh.md').write_text('\n'.join(text)+'\n')
    for shard in (0,1):
        archive=folder/f'raw_checkpoints_shard{shard}.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in sorted((folder/f'shard{shard}/checkpoints').glob('*/*.json')):
                tar.add(path,arcname=str(path.relative_to(folder)),recursive=False)
        with tarfile.open(archive) as tar:
            members=tar.getmembers();assert len(members)==len(shards[shard])*len(conditions(kind))
            lookup={r['path']:r['sha256'] for r in hashes}
            for member in members:assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==lookup[member.name]
    seal(folder/'checkpoint_hashes.json',hashes)
    seal(folder/'validation.json',dict(passed=True,labels=len(generations),questions=3000,image_clusters=2409,
        independently_decoded_rescored=len(generations),no_missing_or_duplicates=True,
        all_new_generations_decode_native=True,all_reused_native_full_logits_replayed=True,
        origins=origins,source_binding=digest(p),no_nan_inf_oom=True))
    seal(folder/'completion.json',dict(status='COMPLETE',questions=3000,conditions=len(conditions(kind)),labels=len(generations),
        most_harmful_layer=most['layer'],max_harm_delta_pp=most['delta_accuracy_pp'],
        bonferroni_supported_harm_layers=supported,analysis_seconds=time.time()-started,origins=origins))
    print('COMPLETE',kind,most,flush=True)


def cross():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'cross_model';folder.mkdir(parents=True,exist_ok=True)
    all_rows=[];fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    for kind,color in [('llava','#386cb0'),('qwen','#b24745')]:
        assert read(OUT/kind/'completion.json')['status']=='COMPLETE' and read(OUT/kind/'validation.json')['passed']
        rows=list(csv.DictReader((OUT/kind/'layer_paired_comparisons.csv').open()))
        x=np.array([int(r['layer'])/(MODELS[kind]['layers']-1) for r in rows]);y=np.array([float(r['delta_accuracy_pp']) for r in rows])
        ax.plot(x,y,'o-',label=f'{kind.upper()} ({MODELS[kind]["layers"]} layers)',color=color,markersize=3,lw=1)
        ax.fill_between(x,[float(r['bonferroni95_low']) for r in rows],[float(r['bonferroni95_high']) for r in rows],color=color,alpha=.13)
        all_rows.extend(dict(model=kind,relative_depth=float(depth),**r) for depth,r in zip(x,rows))
    ax.axhline(0,color='.3',lw=.8);ax.set(xlabel='Relative depth: layer / (number of layers - 1)',
        ylabel='Accuracy change vs each Native Full (pp)',title='Answer-position Attention-update Sensitivity')
    ax.legend(frameon=False);ax.grid(alpha=.15)
    for ext in ('png','pdf'):fig.savefig(folder/f'figure_D_relative_depth.{ext}',dpi=230)
    plt.close(fig);table(folder/'cross_model_accuracy.csv',all_rows)
    text=['# 跨模型完整答案Accuracy敏感性','',
        '两个模型按顺序、各自双卡完成，同一图片全部条件同卡配对。Native Full均为原生，不是All-Writer。',
        '仅关闭一个语言层原始P−1的整个o_proj更新；后续Decode原生。图中阴影为各模型32/36项Bonferroni校正95%CI。','']
    for kind in ('llava','qwen'):
        p=read(OUT/kind/'completion.json');text.append(f"- {kind.upper()}最大点估计损害L{p['most_harmful_layer']}：{p['max_harm_delta_pp']:+.4f} pp；校正后可检测损害层{p['bonferroni_supported_harm_layers']}。")
    text+=['','原始Prompt、FP16/BF16、视觉布局及模型版本不同，不能把峰位差异单独归因于架构。',
        '首Token与最终ACC峰位及相关性详见各模型report_zh.md和first_token_vs_accuracy.csv。',
        '本轮不证明独立视觉读取模块、不证明单层可安全省略，也未测量物理跳层或加速。到此停止，不运行InternVL或联合条件。']
    (folder/'report_zh.md').write_text('\n'.join(text)+'\n')
    seal(folder/'completion.json',dict(status='COMPLETE',models=['llava','qwen'],total_labels=210000))


if __name__=='__main__':
    cross() if sys.argv[1]=='cross' else analyze(sys.argv[1])
