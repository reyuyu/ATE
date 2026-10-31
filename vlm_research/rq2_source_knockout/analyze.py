"""One complete-study merge, image-cluster intervals and unsmoothed true data."""
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from benchmarks.dataset_registry import digest, sha256, freeze_json
from paper_rq1_early_joint_attention.statistics import bootstrap
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations
from .engine import SOURCES
from .run import OUT, ROOT, TASKS, SEED, read, verify

COLORS={'Image':'#158c89','Prompt':'#cd6142','BOS':'#7660a5'}


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def collect():
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json')
    complete=[read(OUT/f'shard{k}/completion.json') for k in range(2)]
    assert all(r['binding']==binding and r['status']=='complete' and r['hooks_clean'] and r['parameters_unchanged'] for r in complete)
    actual=list(OUT.glob('shard*/checkpoints/*.json'))
    assert len(actual)==240
    collected=[];seen=set()
    for sample in manifest['samples']:
        rank=manifest['assignments'][sample['question_id']]
        r=read(OUT/f"shard{rank}/checkpoints/{sample['task']}__{sample['question_id']}.json")
        assert r['binding']==binding and r['sample_sha256']==digest(sample) and r['result_sha256']==digest(r['result'])
        assert r['image_sha256']==sample['image_sha256'] and r['physical_gpu']==rank+2
        assert sample['question_id'] not in seen
        seen.add(sample['question_id']);result=r['result']
        assert [(v['layer'],v['source']) for v in result['metrics']]==[(l,s) for l in range(32) for s in SOURCES]
        assert sha256(OUT/result['full_logits_file'])==result['full_logits_file_sha256']
        collected.append((sample,result))
    assert Counter(s['task'] for s,r in collected)==Counter({t:40 for t in TASKS})
    per=[]
    for sample,result in collected:
        for row in result['metrics']:
            assert all(np.isfinite(row[k]) for k in ['logp_loss','js','full_source_mass'])
            assert row['full_token_id']==result['metrics'][0]['full_token_id']
            assert row['logp_loss']==row['full_token_logp']-row['ablated_full_token_logp']
            per.append(dict(task=sample['task'],question_id=sample['question_id'],image_id=sample['image_id'],
                image_sha256=sample['image_sha256'],physical_gpu=manifest['assignments'][sample['question_id']]+2,**row))
    assert len(per)==23040
    write_csv(OUT/'per_question.csv',per)
    write_csv(OUT/'baseline_comparison.csv',[dict(task=s['task'],question_id=s['question_id'],**r['history']) for s,r in collected])
    freeze_json(OUT/'integrity.json',dict(passed=True,questions=240,interventions=23040,
        per_task=dict(Counter(s['task'] for s,r in collected)),images=manifest['images'],shard_questions=manifest['shard_questions'],
        no_duplicates=True,all_image_clusters_disjoint=True,binding=binding,full_baselines=240,
        native_full_exact_history=sum(r['history']['historical_full_exact'] for s,r in collected),
        historical_full_top1_same=sum(r['history']['historical_full_top1_same'] for s,r in collected),
        historical_full_max_abs_difference=max(r['history']['historical_full_max_abs_difference'] for s,r in collected),
        historical_mass_max_abs_difference=max(r['history']['historical_mass_max_abs_difference'] for s,r in collected),
        max_native_row_error=max(v['row_error'] for s,r in collected for v in r['native_rows']),
        gpu_seconds=[r['seconds'] for r in complete],per_question_csv_sha256=sha256(OUT/'per_question.csv')))
    return collected


def summaries(collected):
    tables=[];correlations=[]
    for task in ['Overall',*TASKS]:
        items=[(s,r) for s,r in collected if task=='Overall' or s['task']==task]
        n=len(items);images=[s['image_sha256'] for s,r in items]
        keys=['logp_loss','top1_flip','js','full_source_mass']
        matrices={k:np.array([[v[k] for v in r['metrics']] for s,r in items],dtype=float) for k in keys}
        joined=np.column_stack([matrices[k] for k in keys])
        ci=bootstrap(joined,images,[1]*joined.shape[1],20000,SEED)
        for index in range(96):
            loss=matrices['logp_loss'][:,index]
            row=dict(task=task,layer=index//3,source=SOURCES[index%3],questions=n,image_clusters=len(set(images)),
                median_logp_loss=float(np.median(loss)),std_logp_loss=float(loss.std(ddof=1)),
                p5_logp_loss=float(np.quantile(loss,.05)),p95_logp_loss=float(np.quantile(loss,.95)),
                min_logp_loss=float(loss.min()),max_logp_loss=float(loss.max()),
                negative_loss_count=int((loss<0).sum()),top1_flip_count=int(matrices['top1_flip'][:,index].sum()))
            for ki,key in enumerate(keys):
                stats=ci[ki*96+index]
                row.update({f'mean_{key}':stats['mean'],f'{key}_ci95_low':stats['ci95_low'],f'{key}_ci95_high':stats['ci95_high']})
            tables.append(row)
            pearson,spearman=descriptive_correlations(matrices['full_source_mass'][:,index],loss)
            correlations.append(dict(task=task,source=SOURCES[index%3],scope='Across questions within one fixed layer',
                layer=index//3,observations=n,pearson=pearson,spearman=spearman))
        for si,source in enumerate(SOURCES):
            x=matrices['full_source_mass'][:,si::3];y=matrices['logp_loss'][:,si::3]
            for scope,a,b in [('Across32 layer means',x.mean(0),y.mean(0)),
                ('Pooled question-layer observations; depth-confounded descriptive only',x.ravel(),y.ravel())]:
                pearson,spearman=descriptive_correlations(a,b)
                correlations.append(dict(task=task,source=source,scope=scope,layer='',observations=len(a),pearson=pearson,spearman=spearman))
    write_csv(OUT/'layer_source_summary.csv',[r for r in tables if r['task']=='Overall'])
    write_csv(OUT/'task_source_summary.csv',[r for r in tables if r['task']!='Overall'])
    write_csv(OUT/'mass_loss_correlations.csv',correlations)
    return tables,correlations


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def decorate(ax):
    for low,high,color in [(-.5,1.5,'#dedede'),(1.5,5.5,'#d9e9e2'),(12.5,16.5,'#f4dfd6')]:
        ax.axvspan(low,high,color=color,alpha=.35,zorder=0)
    ax.axhline(0,color='#777777',linewidth=.6)
    ax.set_xlim(-.5,31.5);ax.set_xticks([0,4,8,12,16,20,24,28,31])
    ax.set_xlabel('Layer index');ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)


def figures(tables):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.titleweight':'medium'})
    overall=[r for r in tables if r['task']=='Overall']
    bounds=[r[k] for r in tables for k in ['logp_loss_ci95_low','logp_loss_ci95_high']]
    lower,upper=min(0,min(bounds)),max(bounds);pad=max((upper-lower)*.07,.001)
    for kind in ['loss','flip']:
        fig,axes=plt.subplots(2,3,figsize=(14,7.2),sharex=True,sharey=True,constrained_layout=True)
        for ax,task in zip(axes.flat,TASKS):
            for source in SOURCES:
                rows=[r for r in tables if r['task']==task and r['source']==source]
                key='logp_loss' if kind=='loss' else 'top1_flip'
                y=[r[f'mean_{key}'] for r in rows]
                ax.plot(range(32),y,label=f'{source} OFF',color=COLORS[source],linewidth=1.5)
                ax.fill_between(range(32),[r[f'{key}_ci95_low'] for r in rows],
                    [r[f'{key}_ci95_high'] for r in rows],color=COLORS[source],alpha=.14,linewidth=0)
            decorate(ax);ax.set_title(f'{task} (n=40)')
            if kind=='loss': ax.set_ylim(lower-pad,upper+pad)
        axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
        for ax in axes[:,0]: ax.set_ylabel('Mean Full-top1 LogP loss (nats)' if kind=='loss' else 'Top-1 flip rate')
        fig.suptitle('RQ2 Figure 1: Layer-wise Source-Specific Attention Necessity' if kind=='loss' else 'First-answer Top-1 Changes (Not Full-answer Accuracy)')
        save(fig,'figure1_source_necessity' if kind=='loss' else 'top1_flip_tasks')
    fig,axes=plt.subplots(1,2,figsize=(12,3.7),constrained_layout=True)
    for ax,key,ylabel in zip(axes,['logp_loss','top1_flip'],['Mean Full-top1 LogP loss (nats)','Top-1 flip rate']):
        for source in SOURCES:
            rows=[r for r in overall if r['source']==source]
            ax.plot(range(32),[r[f'mean_{key}'] for r in rows],label=f'{source} OFF',color=COLORS[source],linewidth=1.6)
            ax.fill_between(range(32),[r[f'{key}_ci95_low'] for r in rows],
                [r[f'{key}_ci95_high'] for r in rows],color=COLORS[source],alpha=.16,linewidth=0)
        decorate(ax);ax.set_ylabel(ylabel);ax.set_title('240-question exploratory aggregate')
    axes[0].legend(frameon=False,ncol=3,fontsize=8)
    save(fig,'overall_source_necessity_and_flip')
    matrix=np.array([[next(r['mean_logp_loss'] for r in overall if r['layer']==l and r['source']==s) for s in SOURCES] for l in range(32)])
    scale=max(float(np.abs(matrix).max()),1e-6)
    fig,ax=plt.subplots(figsize=(4.8,9.2),constrained_layout=True)
    im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale))
    ax.set_xticks(range(3),[s+' OFF' for s in SOURCES]);ax.set_yticks(range(32));ax.set_ylabel('Layer index')
    ax.set_title('Source Necessity: 240 Questions')
    fig.colorbar(im,ax=ax,label='Mean Full-top1 LogP loss (nats)')
    save(fig,'source_sensitivity_heatmap')


def report(tables,correlations):
    overall=[r for r in tables if r['task']=='Overall'];audit=read(OUT/'integrity.json');c=read(OUT/'protocol.json')
    lines=['# Research 2 Experiment 2.1：单层来源 Attention 必要性','',
        '状态：完整完成240题（六类各40）、96条件，共23,040次干预 Prefill。另执行240次同机 Full 配对前向；没有完整答案生成、Decode或训练。','',
        '## 主要结果','',
        '| 来源 | 最大平均 Loss 层 | 平均 Loss（nats） | 探索性95% CI | 首Token翻转 | L14–L16平均 Loss |',
        '| --- | --- | --- | --- | --- | --- |']
    for source in SOURCES:
        rows=[r for r in overall if r['source']==source];peak=max(rows,key=lambda r:r['mean_logp_loss'])
        middle=np.mean([r['mean_logp_loss'] for r in rows if 14<=r['layer']<=16])
        lines.append(f"| {source} OFF | L{peak['layer']} | {peak['mean_logp_loss']:.6f} | [{peak['logp_loss_ci95_low']:.6f}, {peak['logp_loss_ci95_high']:.6f}] | {peak['top1_flip_count']}/240 | {middle:.6f} |")
    lines+=['','## 对研究问题的回答','']
    middle={s:np.mean([r['mean_logp_loss'] for r in overall if r['source']==s and 14<=r['layer']<=16]) for s in SOURCES}
    lead=max(middle,key=middle.get)
    lines.append(f"1. 按本轮直接来源 Hard Mask 的平均首Token Loss，L14–L16最大的是 **{lead}** 路径："+', '.join(f'{s}={middle[s]:.6f}' for s in SOURCES)+' nats。来源不是互斥的因果贡献分解；Softmax重分配与非线性使三个损害不能相加。')
    for source in SOURCES:
        rows=[r for r in overall if r['source']==source]
        top=sorted(rows,key=lambda r:r['mean_logp_loss'],reverse=True)[:3]
        lines.append(f"2. {source}最大损害三层："+'，'.join(f"L{r['layer']} ({r['mean_logp_loss']:.5f})" for r in top)+'。')
    lines+=['','### 分任务峰值','', '| Task | Image OFF peak | Prompt OFF peak | BOS OFF peak |','| --- | --- | --- | --- |']
    for task in TASKS:
        cells=[]
        for source in SOURCES:
            p=max([r for r in tables if r['task']==task and r['source']==source],key=lambda r:r['mean_logp_loss'])
            cells.append(f"L{p['layer']} / {p['mean_logp_loss']:.5f}")
        lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines+=['','### BOS主导层与来源必要性','']
    for source in SOURCES:
        early=[r for r in overall if r['source']==source and 2<=r['layer']<=5]
        lines.append(f"- L2–L5 {source}平均原生来源Mass={np.mean([r['mean_full_source_mass'] for r in early]):.4f}，平均Loss={np.mean([r['mean_logp_loss'] for r in early]):.6f}，翻转总数={sum(r['top1_flip_count'] for r in early)}/960个题层观测（不是960张独立图片）。")
    lines+=['','### Mass与Loss的描述性关系','']
    for s in SOURCES:
        corr=next(r for r in correlations if r['task']=='Overall' and r['source']==s and r['scope']=='Across32 layer means')
        lines.append(f"- {s}：32层均值 Pearson={corr['pearson']:.4f}，Spearman={corr['spearman']:.4f}。这混有层深结构，不是控制层号后的机制证据。固定单层内、严格对齐题目的相关性见CSV。")
    lines+=['','## 协议与数值审计','',
        f"- 模型revision：`{c['model_revision']}`；FP16/eager，Transformers4.49.0；仅物理GPU2/3。",
        f"- 固定seed20261031，哈希每类抽40；共{audit['images']}个图片SHA256簇；整簇分片120/120，题目权重保留。",
        '- Prompt OFF屏蔽所有真实文本Key，但排除BOS、Image和P−1自身。旧Prompt Mass包含自身，本轮单独保存自身Mass及真正被阻断的Prompt Mass，不能混用。',
        '- 直接复用既有VisualAttentionKnockout/knockout_mask；只替换目标Query/Key集合。所有Mask在Softmax之前生效，-inf使目标概率严格为0，其余合法Key重新归一化。',
        '- 两张GPU各2题Smoke：No-op全部有效Logits精确一致；三来源覆盖L0/L15/L31；其他Query的逐层状态、Attention和最终Logits逐位一致；正常及异常Hook清理通过。',
        f"- 历史Full向量精确一致：{audit['native_full_exact_history']}/240；Top1一致：{audit['historical_full_top1_same']}/240；最大Logit差={audit['historical_full_max_abs_difference']:.8g}；原生Mass最大差={audit['historical_mass_max_abs_difference']:.8g}。无论历史是否一致，指标始终使用本轮同卡Full，不放宽容差。",
        f"- 最大原生Head行和误差={audit['max_native_row_error']:.8g}，沿用历史2e-3界限；无NaN、Inf、OOM；输入与参数版本不变；各题checkpoint哈希及输入绑定完整。",
        f"- 两卡正式执行用时：{audit['gpu_seconds'][0]:.1f}s、{audit['gpu_seconds'][1]:.1f}s；代码冻结commit：`{c['code_commit']}`。",
        '- 20,000次按图片SHA256整簇配对Bootstrap，普通逐点95%CI。240题仅为探索，96条件及分任务不作未校正确认性显著性宣称。未做独立确认。','',
        '## 解释边界与下一步','',
        '- Loss固定Full原预测Token，允许负值；不是Gold答案损失。Top1翻转不是完整答案Accuracy。所有数值均来自实际前向。',
        '- 阻断Image直接读取后，视觉信息仍可经残差、Prompt表示和其他层间接传递；低敏感不能证明无需视觉。BOS或Prompt同理。',
        '- 高Mass不等于高功能必要性；接近零或无Flip不能证明路径无功能。来源屏蔽的Softmax重分配也可能产生有利或不利效果。',
        '- 下一步建议仅作为研究决策：预注册本轮中层来源差异，在独立样本上确认后，再考虑完整答案指标。本轮不追加实验。','',
        '## 文件','',
        '- `figure1_source_necessity.pdf/png`：六任务三曲线，同一纵轴尺度。',
        '- `overall_source_necessity_and_flip.pdf/png`：240题探索汇总与翻转。',
        '- `source_sensitivity_heatmap.pdf/png`：32×3真实均值，统一对称色标。',
        '- `top1_flip_tasks.pdf/png`：六任务翻转率。',
        '- `per_question.csv`、`layer_source_summary.csv`、`task_source_summary.csv`、`mass_loss_correlations.csv`。',
        '- `protocol.json`、`sample_manifest.json`、`integrity.json`、两卡Smoke及逐题checkpoint、当前Full向量。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def main():
    lock=(OUT/'aggregation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    collected=collect();print('Complete240 validated',flush=True)
    tables,correlations=summaries(collected);figures(tables);report(tables,correlations)
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['result_hashes.json','aggregation.lock','worker.lock'] and not p.name.endswith('.tmp')}
    freeze_json(OUT/'result_hashes.json',hashes)
    print('All summaries, figures and report complete',flush=True)


if __name__=='__main__': main()
