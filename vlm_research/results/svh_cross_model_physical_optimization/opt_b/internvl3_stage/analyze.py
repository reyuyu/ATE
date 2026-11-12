"""Complete-only aggregation, image-cluster bootstrap, unsmoothed figures."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

from .common import (OUT, ROOT, TASKS, CONDITIONS, HISTORY, SEED, read, write,
                     sha256, csv_write, csv_read, verify, digest)


def bootstrap(values, images, repetitions=20000):
    values=np.asarray(values,float)
    if values.ndim==1:
        values=values[:,None]
    assert len(values)==len(images) and np.isfinite(values).all()
    unique=sorted(set(images))
    lookup={key:i for i,key in enumerate(unique)}
    sums=np.zeros((len(unique),values.shape[1]))
    counts=np.zeros(len(unique))
    for image,row in zip(images,values):
        sums[lookup[image]]+=row
        counts[lookup[image]]+=1
    weights=np.random.default_rng(SEED).multinomial(len(unique),np.ones(len(unique))/len(unique),size=repetitions)
    samples=(weights@sums)/(weights@counts)[:,None]
    return values.mean(0),np.quantile(samples,[.025,.975],axis=0)


def load():
    protocol=verify()
    assert read(OUT/'execution_run.json')['passed'] and read(OUT/'gpu_smoke.json')['passed']
    samples=read(OUT/'sample_manifest.json')['samples']
    assert len(samples)==120
    assert len(list((OUT/'checkpoints').glob('*.json')))==120
    records=[]
    for sample in samples:
        record=read(OUT/'checkpoints'/(sample['task']+'__'+sample['question_id']+'.json'))
        assert record['binding']==sha256(OUT/'protocol.json') and record['input_binding']==digest(sample)
        assert record['complete'] and set(record['windows'])==set(CONDITIONS)
        assert {r['layer'] for r in record['sensitivity']}==set(range(28))
        assert set(record['mass'])==set(map(str,range(28)))
        assert sha256(record['native_first_logits_path'])==record['native_first_logits_file_sha256']
        records.append(record)
    return samples,records,protocol


def summarize(samples,records):
    mass_rows,sensitivity_rows,window_rows=[],[],[]
    sources=['anchor','image','question','template','self']
    for sample,record in zip(samples,records):
        common={k:sample[k] for k in ['task','question_id','image_id','image_sha256']}
        full=record['windows']['Full']
        for layer in range(28):
            head=np.asarray(record['mass'][str(layer)]['head_masses'])
            assert head.shape==(28,5) and np.isfinite(head).all()
            value=dict(**common,layer=layer,input_length=len(sample['input_ids']),visual_tokens=256,
                native_first_token=full['first_token'],row_sum_max_error=record['mass'][str(layer)]['row_sum_max_error'])
            for i,name in enumerate(sources):
                value[name+'_mass']=float(head[:,i].mean())
                value[name+'_head_std']=float(head[:,i].std())
                value[name+'_head_min']=float(head[:,i].min())
                value[name+'_head_max']=float(head[:,i].max())
            value['prompt_mass']=value['question_mass']+value['template_mass']
            mass_rows.append(value)
        sensitivity_rows.extend(dict(**common,**row) for row in sorted(record['sensitivity'],key=lambda r:r['layer']))
        for name,value in record['windows'].items():
            base_correct=int(full['normalized']==sample['answer'])
            correct=int(value['normalized']==sample['answer'])
            row=dict(**common,condition=name,answer=value['answer'],normalized=value['normalized'],gold=sample['answer'],
                full_answer=full['answer'],correct=correct,full_correct=base_correct,delta=correct-base_correct,
                correct_to_wrong=int(base_correct==1 and correct==0),wrong_to_correct=int(base_correct==0 and correct==1),
                first_token_flip=int(value['first_token']!=full['first_token']),text_changed=int(value['answer']!=full['answer']),
                tokens_changed=int(value['token_ids']!=full['token_ids']),token_ids=json.dumps(value['token_ids']),
                eos=value['eos'],truncated=value['truncated'],empty=value['empty'],repeated=value['repeated'],length=value['length'],
                first_logp_loss=value.get('signed_logp_loss',0.))
            window_rows.append(row)
    csv_write('attention_mass_per_question_layer.csv',mass_rows)
    csv_write('single_layer_sensitivity_per_question.csv',sensitivity_rows)
    csv_write('window_per_question.csv',window_rows)
    mass_summary,sensitivity_summary,window_summary=[],[],[]
    for group in ['Overall',*TASKS]:
        selected=[i for i,s in enumerate(samples) if group=='Overall' or s['task']==group]
        identities=[samples[i]['image_sha256'] for i in selected]
        mass_metrics=[name+'_mass' for name in sources]+['prompt_mass']
        sensitivity_metrics=['signed_logp_loss','absolute_logp_change','top1_flip','js_divergence']
        for rows,metrics,result in [(mass_rows,mass_metrics,mass_summary),(sensitivity_rows,sensitivity_metrics,sensitivity_summary)]:
            matrix=np.array([[rows[i*28+layer][metric] for layer in range(28) for metric in metrics] for i in selected])
            mean,ci=bootstrap(matrix,identities)
            for layer in range(28):
                row=dict(group=group,layer=layer,n=len(selected),images=len(set(identities)))
                for j,metric in enumerate(metrics):
                    column=layer*len(metrics)+j
                    values=matrix[:,column]
                    row.update({metric:float(mean[column]),metric+'_ci_low':float(ci[0,column]),metric+'_ci_high':float(ci[1,column])})
                    if rows is sensitivity_rows:
                        row[metric+'_median']=float(np.median(values))
                        row[metric+'_p95']=float(np.quantile(values,.95))
                if rows is sensitivity_rows:
                    row['top1_flip_count']=int(round(row['top1_flip']*len(selected)))
                result.append(row)
        qids={(samples[i]['task'],samples[i]['question_id']) for i in selected}
        for name in CONDITIONS:
            rows=[r for r in window_rows if (r['task'],r['question_id']) in qids and r['condition']==name]
            values=np.array([[r['correct']*100,r['delta']*100] for r in rows])
            mean,ci=bootstrap(values,[r['image_sha256'] for r in rows])
            row=dict(group=group,condition=name,n=len(rows),images=len({r['image_sha256'] for r in rows}),
                accuracy_pct=float(mean[0]),accuracy_ci_low=float(ci[0,0]),accuracy_ci_high=float(ci[1,0]),
                delta_pp=float(mean[1]),delta_ci_low=float(ci[0,1]),delta_ci_high=float(ci[1,1]),
                degenerate_bootstrap=bool(ci[0,1]==ci[1,1]))
            for key in ['correct_to_wrong','wrong_to_correct','first_token_flip','text_changed','tokens_changed','eos','truncated','empty','repeated']:
                row[key]=sum(r[key] for r in rows)
            row['mean_length']=float(np.mean([r['length'] for r in rows]))
            window_summary.append(row)
    csv_write('attention_mass_layer_summary.csv',mass_summary)
    csv_write('single_layer_sensitivity_summary.csv',[r for r in sensitivity_summary if r['group']=='Overall'])
    csv_write('task_layer_sensitivity.csv',[r for r in sensitivity_summary if r['group']!='Overall'])
    csv_write('window_accuracy_summary.csv',window_summary)
    csv_write('window_paired_comparisons.csv',[r for r in window_summary if r['group']=='Overall' and r['condition']!='Full'])
    return mass_summary,sensitivity_summary,window_summary


def cross_model(mass,sensitivity,windows):
    result=[]
    historical=csv_read(HISTORY['llava_sensitivity'])
    for row in historical:
        if row['group']!='Overall':
            continue
        layer=int(row['layer'])
        result.append(dict(kind='layer',model='LLaVA',layer=layer,relative_depth=layer/31,n=3000,visual_tokens=576,
            anchor_name='BOS',anchor_mass=float(row['bos_mass']),image_mass=float(row['image_mass']),
            other_text_including_self_mass=float(row['prompt_mass']),
            signed_logp_loss=-float(row['mean_delta_logp']),absolute_logp_change=float(row['mean_abs_delta_logp']),
            sign_conversion='Negate historical OFF-minus-Full delta_logp to Full-minus-OFF; verified original metric_rows source'))
    intern_mass={r['layer']:r for r in mass if r['group']=='Overall'}
    for row in sensitivity:
        if row['group']!='Overall':
            continue
        layer=row['layer'];m=intern_mass[layer]
        result.append(dict(kind='layer',model='InternVL',layer=layer,relative_depth=layer/27,n=120,visual_tokens=256,
            anchor_name='Start Anchor (not BOS)',anchor_mass=m['anchor_mass'],image_mass=m['image_mass'],
            other_text_including_self_mass=m['prompt_mass']+m['self_mass'],
            signed_logp_loss=row['signed_logp_loss'],absolute_logp_change=row['absolute_logp_change'],sign_conversion='Already Full-minus-OFF'))
    for row in csv_read(HISTORY['llava_windows']):
        if row['group']=='Overall' and row['condition']!='Full':
            result.append(dict(kind='window',model='LLaVA',condition=row['condition'],layer=int(row['start_layer']),
                relative_depth=int(row['start_layer'])/31,n=3000,visual_tokens=576,accuracy_pct=float(row['accuracy_pct']),
                delta_pp=float(row['delta_pp']),delta_ci_low=float(row['ci95_low_pp']),delta_ci_high=float(row['ci95_high_pp'])))
    for row in windows:
        if row['group']=='Overall' and row['condition']!='Full':
            start=CONDITIONS[row['condition']][0]
            result.append(dict(kind='window',model='InternVL',condition=row['condition'],layer=start,
                relative_depth=start/27,n=120,visual_tokens=256,accuracy_pct=row['accuracy_pct'],delta_pp=row['delta_pp'],
                delta_ci_low=row['delta_ci_low'],delta_ci_high=row['delta_ci_high']))
    csv_write('cross_model_comparison.csv',result)
    return result


def figures(mass,sensitivity,windows,cross):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.titleweight':'normal'})
    def save(fig,name):
        fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight')
        plt.close(fig)
    def line(ax,rows,key,label,color,ci=True):
        x=[r['layer'] for r in rows];y=[r[key] for r in rows]
        ax.plot(x,y,label=label,color=color,lw=1.5)
        if ci:
            ax.fill_between(x,[r[key+'_ci_low'] for r in rows],[r[key+'_ci_high'] for r in rows],color=color,alpha=.12,lw=0)
        ax.grid(axis='y',alpha=.17)
        ax.set_xlim(x[0],x[-1])
        ax.set_xlabel('Language decoder layer')
    colors=['#333333','#17835c','#c35435','#536eaa']
    main_mass=[r for r in mass if r['group']=='Overall']
    main_sens=[r for r in sensitivity if r['group']=='Overall']
    fig,ax=plt.subplots(figsize=(7.4,3.0),layout='constrained')
    for key,label,color in zip(['anchor_mass','image_mass','prompt_mass','self_mass'],['Start Anchor','Image','Prompt','Self'],colors):
        line(ax,main_mass,key,label,color)
    ax.set_ylabel('Native P-1 attention mass');ax.set_ylim(0,1)
    ax.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'figure1_attention_mass')
    fig,axes=plt.subplots(1,3,figsize=(11,2.9),layout='constrained')
    for ax,key,label in zip(axes,['signed_logp_loss','absolute_logp_change','top1_flip'],['Signed LogP Loss (nat)','Absolute LogP Change (nat)','Top-1 Flip Rate']):
        line(ax,main_sens,key,label,colors[1]);ax.set_ylabel(label)
        peak=max(main_sens,key=lambda r:r[key])
        ax.scatter(peak['layer'],peak[key],color=colors[2],s=20,zorder=4)
        ax.annotate('L'+str(peak['layer']),(peak['layer'],peak[key]),xytext=(4,6),textcoords='offset points',fontsize=8)
    save(fig,'figure2_layer_sensitivity')
    fig,axes=plt.subplots(2,1,figsize=(10.5,4.9),layout='constrained')
    for ax,key,label in zip(axes,['signed_logp_loss','absolute_logp_change'],['Signed LogP Loss (nat)','Absolute LogP Change (nat)']):
        data=np.array([[r[key] for r in sensitivity if r['group']==task] for task in TASKS])
        limit=max(float(np.abs(data).max()),1e-9)
        im=ax.imshow(data,aspect='auto',cmap='RdBu_r' if key=='signed_logp_loss' else 'viridis',
            vmin=-limit if key=='signed_logp_loss' else 0,vmax=limit,interpolation='nearest')
        ax.set_yticks(range(6),TASKS);ax.set_xticks(range(0,28,2));ax.set_xlabel('Language decoder layer')
        fig.colorbar(im,ax=ax,label=label,pad=.015)
    save(fig,'figure3_task_sensitivity')
    fig,axes=plt.subplots(2,3,figsize=(12,5.7),layout='constrained')
    for j,model in enumerate(['LLaVA','InternVL']):
        rows=[r for r in cross if r['kind']=='layer' and r['model']==model]
        for key,label,color in zip(['anchor_mass','image_mass','other_text_including_self_mass'],
                ['BOS' if model=='LLaVA' else 'Start Anchor','Image','Other Text + Self'],colors):
            axes[0,j].plot([r['layer'] for r in rows],[r[key] for r in rows],label=label,color=color,lw=1.3)
        axes[0,j].set_title(model+(' | 576 visual tokens, n=3000' if model=='LLaVA' else ' | 256 visual tokens, n=120'),fontsize=9)
        axes[0,j].set_ylim(0,1);axes[0,j].legend(frameon=False,fontsize=7)
        axes[0,j].set_ylabel('P-1 source mass')
        for key,label,style in [('signed_logp_loss','Signed','-'),('absolute_logp_change','Absolute','--')]:
            axes[1,j].plot([r['layer'] for r in rows],[r[key] for r in rows],label=label,ls=style,color=colors[1 if style=='-' else 2])
        axes[1,j].legend(frameon=False,fontsize=7);axes[1,j].set_ylabel('LogP change (nat)')
        for ax in axes[:,j]:
            ax.set_xlabel('Original layer index');ax.grid(axis='y',alpha=.15)
    for model,color in [('LLaVA',colors[2]),('InternVL',colors[1])]:
        rows=[r for r in cross if r['kind']=='layer' and r['model']==model]
        axes[0,2].plot([r['relative_depth'] for r in rows],[r['anchor_mass'] for r in rows],label=model+(' BOS' if model=='LLaVA' else ' Anchor'),color=color)
        axes[1,2].plot([r['relative_depth'] for r in rows],[r['signed_logp_loss'] for r in rows],label=model,color=color)
    axes[0,2].set_title('Relative network depth');axes[0,2].set_ylabel('BOS / Start Anchor Mass');axes[0,2].set_ylim(0,1)
    axes[1,2].set_ylabel('Signed LogP Loss (nat)')
    for ax in axes[:,2]:
        ax.set_xlabel('Layer / (decoder layers - 1)');ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.15)
    save(fig,'figure4_llava_vs_internvl')
    main_windows=[r for r in windows if r['group']=='Overall']
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.2),layout='constrained')
    x=np.arange(7);acc=np.array([r['accuracy_pct'] for r in main_windows])
    axes[0].bar(x,acc,color=[colors[0]]+[colors[1]]*4+[colors[2],colors[3]],width=.65)
    axes[0].errorbar(x,acc,yerr=[acc-[r['accuracy_ci_low'] for r in main_windows],[r['accuracy_ci_high'] for r in main_windows]-acc],fmt='none',ecolor='black',capsize=2,lw=.8)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Complete-answer Accuracy (%)')
    delta=np.array([r['delta_pp'] for r in main_windows])
    axes[1].errorbar(x,delta,yerr=[delta-[r['delta_ci_low'] for r in main_windows],[r['delta_ci_high'] for r in main_windows]-delta],fmt='o',color=colors[1],capsize=3)
    axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel('Accuracy difference vs Full (pp)')
    axes[2].bar(x-.16,[r['correct_to_wrong'] for r in main_windows],width=.32,label='Correct to wrong',color=colors[2])
    axes[2].bar(x+.16,[r['wrong_to_correct'] for r in main_windows],width=.32,label='Wrong to correct',color=colors[1])
    axes[2].legend(frameon=False,fontsize=7);axes[2].set_ylabel('Question count')
    labels=['Full','0-3','1-4','2-5','3-6','12-15','24-27']
    for ax in axes:
        ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_xlabel('Four-layer Attention output OFF window');ax.grid(axis='y',alpha=.15)
    save(fig,'figure5_window_accuracy')
    fig,axes=plt.subplots(2,3,figsize=(10,5),layout='constrained',sharex=True,sharey=True)
    for ax,task in zip(axes.flat,TASKS):
        rows=[r for r in mass if r['group']==task]
        for key,label,color in zip(['anchor_mass','image_mass','prompt_mass','self_mass'],['Anchor','Image','Prompt','Self'],colors):
            line(ax,rows,key,label,color)
        ax.set_title(task+' (n=20)');ax.set_ylim(0,1)
    axes[0,0].legend(frameon=False,fontsize=7,ncol=2)
    save(fig,'supplement_task_mass')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.0),layout='constrained')
    for ax,relative in zip(axes,[False,True]):
        for model,color in [('LLaVA',colors[2]),('InternVL',colors[1])]:
            rows=[r for r in cross if r['kind']=='window' and r['model']==model]
            field='relative_depth' if relative else 'layer'
            ax.plot([r[field] for r in rows],[r['delta_pp'] for r in rows],'o-',label=model,color=color,markersize=3,lw=1)
        ax.axhline(0,color='black',lw=.5);ax.set_ylabel('Delta Accuracy (pp)');ax.legend(frameon=False,fontsize=8)
        ax.set_xlabel('Relative window start depth' if relative else 'Original four-layer window start')
    save(fig,'supplement_cross_model_windows')


def report(mass,sensitivity,windows,cross,samples,records):
    m=[r for r in mass if r['group']=='Overall'];s=[r for r in sensitivity if r['group']=='Overall']
    w=[r for r in windows if r['group']=='Overall']
    peak=max(s,key=lambda r:r['signed_logp_loss']);absolute=max(s,key=lambda r:r['absolute_logp_change'])
    task_peaks={task:dict(signed=max((r for r in sensitivity if r['group']==task),key=lambda r:r['signed_logp_loss'])['layer'],
        absolute=max((r for r in sensitivity if r['group']==task),key=lambda r:r['absolute_logp_change'])['layer']) for task in TASKS}
    high=[r['layer'] for r in m if r['anchor_mass']>.5]
    stats=dict(exploratory=True,forced_stages=False,majority_anchor_layers=high,
        first_two_layers=[{k:r[k] for k in ['layer','anchor_mass','image_mass','prompt_mass','self_mass']} for r in m[:2]],
        signed_peak=peak,absolute_peak=absolute,task_peaks=task_peaks,
        late_four_mean_anchor=float(np.mean([r['anchor_mass'] for r in m[-4:]])),
        early_high_anchor_low_sensitivity=[r['layer'] for r in s[:8] if m[r['layer']]['anchor_mass']>.5 and r['absolute_logp_change']<=np.median([a['absolute_logp_change'] for a in s])],
        criteria='Anchor >0.5 means majority mass, low sensitivity means below across-layer median absolute change; descriptive labels only, not tested stage boundaries',
        four_layer_windows=w,cross_model_limitations='Different models/prompts/n/precision/visual token budgets; BOS != Start Anchor; no independent confirmation')
    write(OUT/'stage_assessment.json',stats)
    lines=['# InternVL3-8B 层级 Attention 快速探索','',
        '## 首页回答','',
        f'**Q1 层级趋势：** 原始曲线见 Figure 1/2；主要 Signed 敏感峰在 L{peak["layer"]}，Absolute 峰在 L{absolute["layer"]}。以下报告实际来源分配与功能损害，不强制拟合三阶段或四阶段。',
        f'**Q2 早期高 Anchor：** 全网络 Anchor Mass >0.5 的层为 {high}；L0-L7 中同时低于全层 Absolute 敏感性中位数的高 Anchor 层为 {stats["early_high_anchor_low_sensitivity"]}。这是描述性重合，不证明因果。',
        f'**Q3 中层峰与任务：** Signed 峰 L{peak["layer"]}，Mean Loss={peak["signed_logp_loss"]:.6f} nat；Absolute 峰 L{absolute["layer"]}。六任务峰位见下表，每任务仅20题，不能确认任务特异性。',
        f'**Q4 深层：** L24-L27 平均 Anchor Mass={stats["late_four_mean_anchor"]:.4f}。WL完整答案损害和深层单层损害分别报告，不能用Mass大小代替必要性。',
        '**Q5 连续四层：** 六个冻结窗口实测结果如下；小样本CI跨零或退化为[0,0]不证明等价、安全或非劣效。',
        '**Q6 跨模型：** Figure 4与补充图展示LLaVA/InternVL原始层号与相对深度。两模型统计不是同样本配对；来源锚点、Prompt、精度和视觉Token预算不同，不能视为机制等价。','',
        '## 完整答案窗口结果','',
        '| 条件 | 层 | Accuracy | ΔFull pp | 图片簇配对95%CI pp | 对→错 | 错→对 | 首Token翻转 | EOS / 截断 |',
        '|---|---|---:|---:|---|---:|---:|---:|---|']
    for r in w:
        layers=CONDITIONS[r['condition']]
        label=f'L{layers[0]}-L{layers[-1]}' if layers else '原生'
        lines.append(f'| {r["condition"]} | {label} | {r["accuracy_pct"]:.3f}% | {r["delta_pp"]:+.3f} | [{r["delta_ci_low"]:+.3f}, {r["delta_ci_high"]:+.3f}] | {r["correct_to_wrong"]} | {r["wrong_to_correct"]} | {r["first_token_flip"]} | {r["eos"]} / {r["truncated"]} |')
    lines += ['', '## 原始来源与敏感性','', '| 层 | Anchor | Image | Prompt | Self | Signed Loss nat | Absolute Change nat | Top-1 Flip |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(m,s):
        lines.append(f'| L{a["layer"]} | {a["anchor_mass"]:.4f} | {a["image_mass"]:.4f} | {a["prompt_mass"]:.4f} | {a["self_mass"]:.4f} | {b["signed_logp_loss"]:.6f} | {b["absolute_logp_change"]:.6f} | {b["top1_flip_count"]}/120 |')
    lines+=['','## 六任务峰位','','| 任务 | Signed峰 | Absolute峰 |','|---|---:|---:|']
    for task,p in task_peaks.items():
        lines.append(f'| {task} | L{p["signed"]} | L{p["absolute"]} |')
    lines += ['', '## 协议、校验和解释边界','',
        f'- {len(samples)}题，六任务各20，真实图片SHA256簇 {len({r["image_sha256"] for r in samples})}。样本仅按固定seed与输入哈希选取，未使用模型表现。',
        '- 单GPU1、BF16/Eager、官方448单图块，max_num=1，256视觉Token；不代表默认动态高分辨率配置。',
        '- 全部840个完整生成与3360个单层干预均进入统计。Full生成同时提供首Token Logits，Mass Observer逐题与Full严格一致；单层使用原生视觉Embedding和全新Cache，不是跳过视觉信息。',
        '- 最短、典型、最长实际输入通过Smoke：No-op/Observer Logits差0，非目标Prefill Query不变，L2/L27单层和W2真实Decode Hook有效，MLP/Residual/Cache正常，异常清理通过。',
        '- 全部层统计仅P−1；每层当前行立即归约，不保存全序列Attention矩阵。Head仅用于同题归约，不作为独立统计样本。',
        '- 20,000次Image SHA256整簇Bootstrap，固定seed=20261101，题目加权普通95%CI，非同时区间。Median/P95见CSV。',
        '- **重要符号校正：** 历史LLaVA `delta_logp` 实现是OFF−Full，本轮对照明确取负，统一Signed Loss为Full−OFF。Absolute不变。详见cross_model_comparison.csv和绑定的历史metric_rows源码。',
        '- Figure 1的Prompt不含Self；跨模型Figure 4为对齐历史分类使用Other Text + Self。LLaVA BOS与InternVL Start Anchor分别标注，不假装等价。',
        '- 单层首Token概率变化不等于最终Accuracy变化；多层联合损害不能由单层均值简单相加。窗口消融保留全部MLP、QKV、Cache、视觉计算，不代表真实省略或GPU加速。',
        '- 本轮不引用Qwen四阶段结论，不声称独立确认、非劣效或统计等价。未新增来源Mask、层组合、训练或大样本。','',
        '## 图表','',
        '- Figure 1: figure1_attention_mass.pdf/png',
        '- Figure 2: figure2_layer_sensitivity.pdf/png',
        '- Figure 3: figure3_task_sensitivity.pdf/png',
        '- Figure 4: figure4_llava_vs_internvl.pdf/png',
        '- Figure 5: figure5_window_accuracy.pdf/png',
        '- 补充：supplement_task_mass、supplement_cross_model_windows。全部无平滑，CSV在同目录。','',
        'GPU运行耗时、显存、源码Commit见execution_run.json与gpu1_resources_*.json；资源释放见gpu1_release.json。完成后停止，不自动扩展。']
    # Report prose is an artifact, not model code or a post-hoc protocol adjustment.
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    samples,records,protocol=load()
    mass,sensitivity,windows=summarize(samples,records)
    cross=cross_model(mass,sensitivity,windows)
    figures(mass,sensitivity,windows,cross)
    report(mass,sensitivity,windows,cross,samples,records)
    write(OUT/'validation.json',dict(passed=True,samples=120,images=len({s['image_sha256'] for s in samples}),
        mass_rows=3360,single_layer_rows=3360,generations=840,conditions=list(CONDITIONS),
        all_checkpoints_valid=True,all_samples_included=True,source_hashes_unchanged=True,
        gpu_smoke_passed=True,numerical_errors=[],bootstrap_repetitions=20000,
        ordinary_intervals_only=True,no_noninferiority_claim=True))
    print('AGGREGATED 120 samples, 3360 layers, 840 generations; all figures saved',flush=True)


if __name__=='__main__':
    main()
