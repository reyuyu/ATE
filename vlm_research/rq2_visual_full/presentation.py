"""Publication figures and literal reporting of the frozen NI decision."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from .run import OUT, PRIMARY, SECONDARY, LABELS, CONDITIONS, previous, read

KEYS=['Full',*CONDITIONS]
COLORS=['#555555','#bd4a49','#c48c28','#22866a','#547cad']
SHORT=['Full','All OFF','Keep14','Keep14+19','Drop14+19']


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=240)
    plt.close(fig)


def figures(summary,tasks):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
        'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    order=[PRIMARY,'validation','exploration','all3000']
    titles={PRIMARY:'Primary: image-disjoint', 'validation':'Supporting: all validation',
            'exploration':'Exploration (reused)', 'all3000':'Merged (not independent)'}
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.5),sharex=True)
    for ax,pop in zip(axes.flat,order):
        rr=[next(r for r in summary if r['population']==pop and r['condition']==k) for k in KEYS]
        for x,r in enumerate(rr):
            delta=r['delta_pp']
            ax.errorbar(x,delta,yerr=np.array([[delta-r['delta_ci95_low']],[r['delta_ci95_high']-delta]]),
                fmt='o',color=COLORS[x],ms=6 if x==3 else 4,capsize=3,lw=1.2)
        ax.axhline(0,color='#888888',ls=':',lw=.8)
        ax.axhline(-1,color='#a65050',ls='--',lw=.9)
        ax.set_title(f"{titles[pop]} (n={rr[0]['n']:,})",fontsize=8.8,pad=8)
        ax.set_xticks(range(5),SHORT,rotation=25,ha='right',fontsize=7)
        ax.tick_params(labelbottom=True)
        ax.grid(axis='y',alpha=.15);ax.set_xlim(-.5,4.5);ax.margins(y=.22)
        ax.set_ylabel('Accuracy change vs Full (pp)',fontsize=8)
    fig.suptitle('Full-Scale Visual Attention Sufficiency',fontsize=12,fontweight='bold',y=.98)
    fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.13,hspace=.55,wspace=.26)
    save(fig,'figure_a_visual_sufficiency')
    fig,ax=plt.subplots(figsize=(8.3,4.0))
    names=list(previous.old.TASKS)
    rr=[r for r in tasks if r['population']=='all3000']
    for j,key in enumerate(KEYS):
        values=[next(r['accuracy_pct'] for r in rr if r['task']==t and r['condition']==key) for t in names]
        ax.bar(np.arange(6)+(j-2)*.145,values,width=.14,color=COLORS[j],label=LABELS[key])
    labels=[]
    for task in names:
        delta=next(r['delta_pp'] for r in rr if r['task']==task and r['condition']=='C')
        labels.append(f'{task}\nC-Full: {delta:+.2f}pp')
    ax.set_xticks(np.arange(6),labels,fontsize=8)
    for label,task in zip(ax.get_xticklabels(),names):
        if next(r['delta_pp'] for r in rr if r['task']==task and r['condition']=='C')<0:label.set_color('#a63838')
    ax.set_ylabel('Complete-answer Accuracy (%)');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Six GQA Task Accuracy',fontsize=12,fontweight='bold',y=.98)
    fig.legend(loc='upper center',bbox_to_anchor=(.53,.92),ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.73,bottom=.18)
    save(fig,'figure_b_six_tasks')
    fig,ax=plt.subplots(figsize=(7.6,2.25))
    values=np.array([[int(k not in CONDITIONS[key]) for k in range(32)] for key in ['B','C','D']])
    ax.imshow(values,cmap=ListedColormap(['#dddddd','#22866a']),vmin=0,vmax=1,aspect='auto',interpolation='none')
    ax.set_yticks(range(3),[LABELS[k] for k in ['B','C','D']],fontsize=8)
    ax.set_xticks(range(32),[str(k) for k in range(32)],fontsize=6.5)
    for k,label in enumerate(ax.get_xticklabels()):
        if k in (14,19):label.set_fontweight('bold');label.set_color('#176649')
    ax.set_xticks(np.arange(-.5,32,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_xlabel('Decoder layer',fontsize=8)
    fig.suptitle('Visual Layer Sparsity',fontsize=11,fontweight='bold',y=.98)
    fig.legend(handles=[Patch(color='#22866a',label='Image edges retained'),Patch(color='#dddddd',label='Image edges blocked')],
        loc='upper center',bbox_to_anchor=(.57,.88),ncol=2,frameon=False,fontsize=7)
    fig.subplots_adjust(left=.17,right=.99,top=.64,bottom=.25)
    save(fig,'figure_c_layer_sparsity')
    from .analyze import write_csv
    write_csv(OUT/'visual_layer_sparsity.csv',[dict(condition=key,layer=k,image_retained=k not in CONDITIONS[key]) for key in ['B','C','D'] for k in range(32)])
    captions=['# Figure Captions','',
        'Figure A. 各面板显示相对同题Full的Accuracy变化及整图像簇配对普通95%区间；红色虚线为预声明−1pp界限。唯一主非劣效检验使用图像隔离验证集C−Full下界；其他面板为支持性或描述性结果，不因重复展示形成新的确认性检验。次要四项比较的校正区间见paired_comparisons.csv。',
        'Figure B. 完整3,000题合并、每任务500题的五条件Accuracy；包含探索样本，不是独立确认。红色任务标签标记C相对Full负向点估计，不表示任务分组统计显著。图下显示C−Full百分点变化。',
        'Figure C. 实际固定配置的32层直接P−1→Image边保留/阻断映射。只在首次Prefill最后Query生效；不表示视觉Token被删除，也不表示其他Prefill Query、MLP或Decode被关闭。', '']
    (OUT/'captions_zh.md').write_text('\n'.join(captions))


def report(summary,paired,tasks,completed):
    primary=next(r for r in paired if r['population']==PRIMARY and r['condition']=='C' and r['reference']=='Full')
    support=next(r for r in paired if r['population']=='validation' and r['condition']=='C' and r['reference']=='Full')
    passed=primary['ci95_low_pp']>-1.
    def interval(r,adjusted=False):
        lo,hi=('family95_low_pp','family95_high_pp') if adjusted else ('ci95_low_pp','ci95_high_pp')
        return f"[{r[lo]:+.3f}, {r[hi]:+.3f}]"
    def comparison(a,b,pop=PRIMARY):
        return next(r for r in paired if r['population']==pop and r['condition']==a and r['reference']==b)
    lines=['# Research 2 Experiment 2.3：完整3,000题视觉Attention充分性验证','',
        '**状态：全部完成。仅新增2,760题×4条件＝11,040次完整自由生成；Full3,000题及探索240题四条件全部复用。**','',
        '## 主要结论','',
        f"**预声明主检验：{'通过' if passed else '未通过'}−1pp非劣效。** 图像隔离验证集2,656题、2,174个图像簇，C−Full={primary['delta_pp']:+.3f}pp，双侧95%CI={interval(primary)}pp。下界{primary['ci95_low_pp']:+.3f}pp{'严格大于' if passed else '未严格大于'}−1pp。该下界等价于单侧97.5%下置信界。",
        f"全部2,760题验证结果作为支持：C−Full={support['delta_pp']:+.3f}pp，95%CI={interval(support)}pp；其下界{'满足' if support['ci95_low_pp']>-1 else '不满足'}同一数值判据，但不是另一个可替换主检验。",
        '非劣效不是零损害或等价；仍须报告逐题损害。若主检验不通过，不以合并3,000题或其他子集替代，不重选层、不放宽标准。','',
        '## 样本与独立性','',
        '冻结原六类各500题，原Question ID、Image ID、Prompt及视觉处理不变。探索240题覆盖235个图像簇。剩余2,760题中104题（80个图像SHA256）与探索集共享图片，保留在整体验证结果，但从优先主验证集剔除。图像隔离集为2,656题、2,174簇；完整3,000题共2,409簇。',
        '图像隔离与拆分仅由冻结Question ID/Image SHA256决定，不读取本轮模型输出；同图问题整簇配对Bootstrap。剩余问题仍来自原GQA冻结样本，不是独立外部数据集。3,000题合并含探索数据，不能称为独立确认。详见split_integrity.json。','']
    titles={'exploration':'探索240题（历史复用）','validation':'剩余验证2,760题（支持性）',PRIMARY:'图像隔离验证2,656题（唯一主检验）','all3000':'全部3,000题（合并描述性）'}
    for pop in ['exploration','validation',PRIMARY,'all3000']:
        lines += [f'## {titles[pop]}','','| 条件 | 正确/题数 | Accuracy | ΔFull pp | 普通配对95%CI pp | 原对→错 | 原错→对 | 首Token翻转 | 平均LogP Loss nat |',
                  '|---|---:|---:|---:|---|---:|---:|---:|---:|']
        for r in [r for r in summary if r['population']==pop]:
            lines.append(f"| {r['label']} | {r['correct']}/{r['n']} | {r['accuracy_pct']:.3f}% | {r['delta_pp']:+.3f} | [{r['delta_ci95_low']:+.3f}, {r['delta_ci95_high']:+.3f}] | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_top1_flips']} | {r['mean_logp_loss']:.6f} |")
        lines.append('')
    lines+=['## 次要比较：主验证集的预声明四项族','',
        '整簇配对Bootstrap200,000次；四项次要比较采用Bonferroni4同时双侧95%区间。排除0才支持对应方向差异；普通区间或点估计不能替代校正结论。六任务和其他总体仅支持性/描述性，不另作确认性发现。','',
        '| 比较 | ΔAccuracy pp | 普通95%CI pp | Bonferroni4同时95%CI pp | 参考对→条件错 | 参考错→条件对 |',
        '|---|---:|---|---|---:|---:|']
    for a,b in SECONDARY:
        r=comparison(a,b)
        lines.append(f"| {LABELS[a]} − {LABELS[b]} | {r['delta_pp']:+.3f} | {interval(r)} | {interval(r,True)} | {r['correct_to_wrong']} | {r['wrong_to_correct']} |")
    lines += ['','## 六任务（描述性）','']
    for pop in [PRIMARY,'all3000']:
        lines += [f'### {titles[pop]}','','| 任务 | 题数 | Full | All OFF | Keep14 | Keep14+19 | Drop14+19 | C−Full pp | C损害/纠正 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for task in previous.old.TASKS:
            rr=[next(r for r in tasks if r['population']==pop and r['task']==task and r['condition']==k) for k in KEYS]
            core=rr[3]
            lines.append(f"| {task} | {core['n']} | "+' | '.join(f"{r['accuracy_pct']:.3f}%" for r in rr)+f" | {core['delta_pp']:+.3f} | {core['correct_to_wrong']}/{core['wrong_to_correct']} |")
        lines.append('')
    ca=comparison('C','A');cb=comparison('C','B');drop=comparison('D','Full')
    allca=comparison('C','A','all3000');allcb=comparison('C','B','all3000')
    worst=sorted([r for r in tasks if r['population']==PRIMARY and r['condition']=='C'],key=lambda r:r['delta_pp'])
    lines+=['## 对研究问题的直接回答','',
        f"1. **{'达到' if passed else '未达到'}预声明总体非劣效标准**，依据唯一主集下界{primary['ci95_low_pp']:+.3f}pp，而不是根据点估计或CI含0判定。全部2,760题支持结果及全部3,000题分开展示。",
        '2. 全部3,000题五条件Accuracy见合并表；这是论文总体描述，不是独立确认。',
        f"3. 图像隔离验证中，两层相对All OFF纠正{ca['wrong_to_correct']}题、损害{ca['correct_to_wrong']}题，净{ca['delta_pp']:+.3f}pp；校正CI={interval(ca,True)}。全3,000题对应纠正{allca['wrong_to_correct']}、损害{allca['correct_to_wrong']}题。",
        f"4. L19额外贡献：在Keep14上加L19，主集纠正{cb['wrong_to_correct']}、损害{cb['correct_to_wrong']}题，净{cb['delta_pp']:+.3f}pp，校正CI={interval(cb,True)}。"+('校正区间排除0，支持正向恢复。' if cb['family95_low_pp']>0 else '校正区间未严格支持正向收益，不能仅凭正向点估计宣布贡献已确证。')+f"全3,000题纠正{allcb['wrong_to_correct']}、损害{allcb['correct_to_wrong']}题。",
        f"5. Drop14+19主集净{drop['delta_pp']:+.3f}pp，损害{drop['correct_to_wrong']}、纠正{drop['wrong_to_correct']}题，校正CI={interval(drop,True)}。"+('校正上界小于0，支持图像隔离样本上的删除损害复现。' if drop['family95_high_pp']<0 else '校正区间未严格排除0，不能宣布准确率损害已统计复现。'),
        '6. C条件主集负向差异较大的任务（描述性排序）：'+ '；'.join(f"{r['task']} {r['delta_pp']:+.3f}pp（损害{r['correct_to_wrong']}、纠正{r['wrong_to_correct']}）" for r in worst[:2])+'。完整六任务与所有条件见task_accuracy.csv，不能据排序再选层。',
        '7. '+('结果支持在本模型/短答案/固定其他通路的条件下，直接P−1视觉读取可集中保留在L14/L19而满足预声明总体非劣效界限。' if passed else '当前结果不足以支持两层保留在预声明损害界限内充分；即使点估计接近，也不能推广单层低敏感为联合充分。')+'这不是所有视觉信息处理只需两层的结论。',
        '8. 其他Prefill Query仍正常读取图像，视觉信息可经文本表示与残差间接传递；Decode仍正常读Image。本轮没有验证所有Query的视觉稀疏性、Decode关闭、跨模型普适性、物理K/V删除或实际GPU加速。','',
        '## 执行、正确性与异常','',
        'LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0，Greedy、max_new_tokens32，独立真实KV Cache。直接复用RQ2.2源码，不改变其他Prefill Query、BOS/Prompt/Self、MLP、Residual和位置编码；Mask只在首次Prefill有效，Decode前全部清理。',
        '18项CPU回归通过；每张正式GPU真实Smoke通过，No-op全部有效Logits及生成精确一致，历史Full和探索集四条件逐Token/首分布复用精确一致，576视觉位置与动态P−1、非目标Query、正常Decode、Hook清理及缓存隔离均通过。未放宽容差。',
        '启动时GPU2被外部任务占用，预检拒绝启动，未触碰该进程。正式仅使用GPU1、GPU3，每卡1,380题、5,520次新生成；原三卡尝试及CPU合成分片测试修正记录保留在startup_three_gpu_attempt/，不是正式条件的缺失或模型失败。',
        f"正式并行墙钟{(max(r['finished_epoch'] for r in completed)-min(r['started_epoch'] for r in completed))/60:.2f}分钟，累计Worker时间{sum(r['seconds'] for r in completed)/60:.2f}分钟（含各自加载）。逐卡及Smoke耗时见validation.json和日志。",'']
    lines += ['| 总体 | 条件 | EOS | 截断 | 空答案 |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['population']=='all3000':lines.append(f"| 全3,000题 | {r['label']} | {r['eos']} | {r['truncated']} | {r['empty']} |")
    lines += ['','异常输出全部保留并标记，不剔除问题。完整逐题文本、Token、损益、首Token指标、复用来源及绑定见per_question.csv；每题原始分布和检查点保留。CI包含0不等于无损，首Token Loss不是完整答案Accuracy。','',
        '## 产物与边界','',
        'Figure A/B/C提供PNG及矢量PDF；图外详细Caption见captions_zh.md。所有统计图底表为accuracy_summary.csv、paired_comparisons.csv、task_accuracy.csv；Figure C底表为visual_layer_sparsity.csv。',
        f"执行源码Commit：`{read(OUT/'protocol.json')['code_commit']}`；最终交付Commit由Git记录。只运行预先固定五条件，无训练、无额外组合、无重新扫层；完成后停止。",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
