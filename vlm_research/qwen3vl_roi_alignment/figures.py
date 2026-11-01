import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .preflight import OUT
from .analyze import METRICS,QUERIES,TASKS,rho

COLORS=('#d04a39','#16877f')


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def curves(summary,task,query,metric):
    rows=sorted([s for s in summary if s['task']==task and s['query']==query and s['metric']==metric],key=lambda r:r['layer'])
    assert len(rows)==36
    return tuple(np.array([r[k] for r in rows]) for k in ('mean','ci_low','ci_high'))


def line(ax,summary,task,query,metric,color,label,style='-'):
    mean,low,high=curves(summary,task,query,metric)
    ax.plot(range(36),mean,color=color,label=label,ls=style,lw=1.7)
    ax.fill_between(range(36),low,high,color=color,alpha=.14,lw=0)


def decorate(ax):
    ax.set_xlim(0,35);ax.set_xticks([0,5,10,15,20,26,30,35])
    ax.axvline(26,color='#777777',ls=':',lw=.8)
    ax.set_xlabel('Decoder layer');ax.grid(axis='y',alpha=.18)


def plot(samples,cube,summary,individual_summary,peaks):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.titlesize':10})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,metric,title in zip(axes.flat,METRICS[:4],
            ('Image attention mass','Target ROI attention mass','Conditional ROI hit ratio','Area-normalized ROI enrichment')):
        for q,query in enumerate(QUERIES):line(ax,summary,'Overall',query,metric,COLORS[q],query)
        ax.set_title(title);ax.set_ylabel('Mass' if 'mass' in metric else 'Ratio' if metric=='roi_hit_ratio' else 'Enrichment')
        decorate(ax)
        if metric=='roi_hit_ratio':
            line(ax,summary,'Overall','P-1','expected_random_roi_hit_ratio','#777777','Uniform area expectation',':')
        if metric=='roi_enrichment':
            ax.axhline(1,color='#888888',ls='--',lw=.8,label='Uniform expectation')
            for q,query in enumerate(QUERIES):
                mean,_,_=curves(summary,'Overall',query,'random_roi_enrichment')
                ax.plot(range(36),mean,color=COLORS[q],ls=':',lw=1,label=f'{query} random patches')
            twin=ax.twinx();line(twin,summary,'Overall','P-1','absolute_logp_change','#262626','P-1 Image OFF sensitivity','--')
            twin.set_ylabel('Mean absolute LogP change (nat)');twin.spines['right'].set_visible(True)
        handles,labels=ax.get_legend_handles_labels()
        if metric=='roi_enrichment':
            h,l=twin.get_legend_handles_labels();handles+=h;labels+=l
        ax.legend(handles,labels,fontsize=7,loc='best',frameon=False)
    fig.suptitle('Native visual ROI attention and paired functional sensitivity | GQA, 240 questions',fontsize=12)
    save(fig,'overall_roi_alignment')

    fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for ax,task in zip(axes.flat,TASKS):
        for q,query in enumerate(QUERIES):line(ax,summary,task,query,'roi_enrichment',COLORS[q],query)
        ax.axhline(1,color='#888888',ls=':',lw=.8)
        ax.set_title(f'{task} (n=40)');ax.set_ylabel('ROI enrichment');decorate(ax)
        twin=ax.twinx();line(twin,summary,task,'P-1','absolute_logp_change','#262626','Image OFF sensitivity','--')
        twin.set_ylabel('Absolute LogP change (nat)');twin.spines['right'].set_visible(True)
        if task=='ChooseAttr':
            handles,labels=ax.get_legend_handles_labels();h,l=twin.get_legend_handles_labels()
            ax.legend(handles+h,labels+l,fontsize=7,frameon=False,loc='upper left')
    fig.suptitle('Six-task alignment | ordinary image-cluster 95% CI; right-axis scales vary',fontsize=12)
    save(fig,'six_task_roi_alignment')

    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    for q,query in enumerate(QUERIES):
        values=rho(cube[:,:,q,3],cube[:,:,0,-2]);valid=np.isfinite(values)
        ax=axes[0,q];ax.hist(values[valid],bins=np.linspace(-1,1,25),color=COLORS[q],alpha=.85)
        ax.axvline(0,color='#666666',ls=':',lw=1);ax.set_xlabel('Per-question cross-layer Spearman');ax.set_ylabel('Questions')
        s=next(s for s in individual_summary if s['task']=='Overall' and s['query']==query and s['metric']=='roi_enrichment' and s['sensitivity']=='absolute_logp_change')
        ax.set_title(f'{query}: {valid.sum()} valid / {(~valid).sum()} undefined')
        ax.text(.03,.94,f'Mean {s["mean_spearman"]:.3f}\n95% CI [{s["mean_ci_low"]:.3f}, {s["mean_ci_high"]:.3f}]',
                transform=ax.transAxes,va='top',fontsize=8)
        delta=[r['peak_difference'] for r in peaks if r['query']==query]
        ax=axes[1,q];ax.hist(delta,bins=np.arange(-36,38,2),color=COLORS[q],alpha=.85)
        ax.axvline(0,color='#666666',ls=':',lw=1);ax.set_xlabel('ROI enrichment peak minus sensitivity peak (layers)')
        ax.set_ylabel('Questions');ax.set_title(f'{query}: peak offsets (all 240; first maximum for ties)')
    fig.suptitle('Sample-level alignment is distinct from correlation of layer means',fontsize=12)
    save(fig,'sample_correlations_and_peaks')

    fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
    colors=plt.get_cmap('tab10')
    mean=cube.mean(0)
    for q,query in enumerate(QUERIES):
        ax=axes[0,q]
        sc=ax.scatter(mean[:,q,3],mean[:,0,-2],c=np.arange(36),cmap='viridis',s=28)
        for layer in (1,14,19,26,35):ax.annotate(f'L{layer}',(mean[layer,q,3],mean[layer,0,-2]),xytext=(4,3),textcoords='offset points',fontsize=7)
        ax.set_title(f'{query}: 36 native layer means');ax.set_xlabel('Mean ROI enrichment');ax.set_ylabel('Mean absolute LogP change (nat)')
        fig.colorbar(sc,ax=ax,label='Layer',shrink=.8)
        ax=axes[1,q]
        for i,task in enumerate(TASKS):
            selected=[j for j,s in enumerate(samples) if s['task']==task]
            ax.scatter(cube[selected,26,q,3],cube[selected,26,0,-2],s=17,alpha=.7,color=colors(i),label=task)
        ax.set_title(f'{query}: same-layer alignment at L26');ax.set_xlabel('Per-question ROI enrichment')
        ax.set_ylabel('Absolute LogP change (nat; symlog)');ax.set_yscale('symlog',linthresh=1e-4)
        ax.grid(axis='y',alpha=.18)
        if q==0:ax.legend(fontsize=6,frameon=False,ncol=2)
    fig.suptitle('Descriptive alignment, not causal evidence of ROI necessity',fontsize=12)
    save(fig,'layer_and_l26_alignment')


def report(samples,cube,summary,layer_corr,individual_summary,peak_summary,l26,validation):
    def choose(rows,**kwargs):return next(r for r in rows if all(r[k]==v for k,v in kwargs.items()))
    def ci(r,value='spearman',lo='ci_low',hi='ci_high'):
        return f'{r[value]:.3f} [{r[lo]:.3f}, {r[hi]:.3f}]'
    resolution=json.loads((OUT/'annotation_resolution.json').read_text())
    text=['# Qwen3-VL 视觉 ROI Attention 与功能敏感性逐层对齐（完整 240 题）','',
        '## 完成与有效性','',
        '- 同一批六类 GQA 各 40 题，共 240 题；36 层、P−1/真实 Question 两类 Query，得到 17,280 行配对记录。未换题、未重新筛选。',
        '- 固定 Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/eager、官方 Chat Template、Transformers 4.57.6；模型文件逐份 SHA256 同历史一致。',
        '- 本轮仅原生 Full Prefill，无 Mask、无新生成、无消融重跑，DeepStack 全原生。240 次观测前向，加 24 次无 Hook 的 Smoke 对照；不保存完整 Attention 矩阵。',
        '- 9 项采集前 CPU 测试、2 项统计测试通过（含重复图片整簇重采样检查）；240 题真实 Processor 输入和像素重建检查通过。每任务 4 题，共 24 题 GPU Smoke 通过，带只读 Hook 与无 Hook logits 精确一致。',
        '- 240/240 原生首 Token 全词表 BF16 logits SHA256 与同机历史 Full 精确一致；原始图片 SHA256、输入绑定、8,640 条历史 Image OFF 敏感性严格配对。三卡 DeepStack 实际加法、正常/异常 Hook 清理、权重不变均通过。',
        f'- Attention 行和最大误差 {validation["row_sum_max_error"]:.8f}；沿用原 BF16 阈值 {validation["row_sum_tolerance"]:.8f}，未放宽容差；无 NaN/Inf/OOM。','',
        '## ROI 与统计口径','',
        '- 官方 Cross-modal CSV 120 题；按用户授权，以官方 GQA val scene graph 同图、同对象 ID 补齐另外 120 题。终端语义程序依赖用于定位目标，不按词名猜测。120 个 CSV 题的目标对象集合与 scene graph 坐标全部精确交叉验证一致。',
        f'- 共 {resolution["total_target_boxes"]} 个目标框，3 个框越界 1 像素，均显式裁剪并保留原框、图像/对象 ID 和来源；不存在的 LogicalObj 备选对象不补框。详见 `bbox_provenance.csv`。',
        '- Qwen 的动态网格不是固定 576 Token：16×16 raw patches 经 2×2 spatial merge；验证真实 Processor 打包顺序和官方视觉 rotary 坐标顺序。ROI 为框与 merged 单元正面积相交的并集，边界单元整个计入；细小框的离散覆盖会大于连续像素面积。',
        '- Question 指真实问题子词 Token；先对所有真实 Question Query 与各 Head 的原生 Attention 做算术平均，再计算视觉内部比值。P−1 单独统计；不将模板当作问题。',
        '- ROI Hit Ratio = ROI Mass / Image Mass；ROI Enrichment = Hit Ratio / ROI Token 面积比例。随机基线为每题 128 组等 Token 数均匀无放回随机 patch 集合（允许与真实 ROI 重叠），跨层/Query 共用随机集合；不是空间连续或形状匹配的区域。均匀期望 Enrichment 恒为 1。',
        '- 所有敏感性均来自历史 **P−1 Image OFF**；Question ROI 与之配对不表示测量了 Question 路径的干预必要性。主指标为 Mean Absolute LogP Change，Signed Loss 为辅；不是 Accuracy 或 Gold Answer Loss。',
        '- 图片 SHA256 整簇配对 Bootstrap 2,000 次，保留题目加权估计。普通 95% 区间仅作探索性描述，不以跨层点、Head 或 Query 作独立图像，不报告独立层显著性。','',
        '## 核心对齐结果','',
        '| Query | 层均值 Enrichment–绝对变化 Spearman [95%CI] | 逐题跨层 Spearman 均值 [95%CI] | L26 同层跨题 Spearman [95%CI] |',
        '|---|---|---|---|']
    for query in QUERIES:
        a=choose(layer_corr,task='Overall',query=query,metric='roi_enrichment',sensitivity='absolute_logp_change')
        b=choose(individual_summary,task='Overall',query=query,metric='roi_enrichment',sensitivity='absolute_logp_change')
        c=choose(l26,task='Overall',query=query,metric='roi_enrichment',sensitivity='absolute_logp_change')
        text.append(f'| {query} | {ci(a)} | {ci(b,"mean_spearman","mean_ci_low","mean_ci_high")} | {ci(c)} |')
    text+=['','层均值相关描述“平均曲线是否一起起伏”；逐题相关描述“同一道题的两条曲线是否对齐”；L26 同层相关描述“跨图片 ROI 关注强度是否对应较大 LogP 改变”。这三者不可互相替代，且跨层结构、任务差异与原预测置信度都可能影响关联。','',
           '| 任务 | 绝对敏感性峰层 | P−1 Enrichment 峰层 | Question Enrichment 峰层 | P−1 层均值 Spearman | Question 层均值 Spearman |','|---|---:|---:|---:|---:|---:|']
    for task in ('Overall',)+TASKS:
        selected=[i for i,s in enumerate(samples) if task=='Overall' or s['task']==task];mean=cube[selected].mean(0)
        aa=choose(layer_corr,task=task,query='P-1',metric='roi_enrichment',sensitivity='absolute_logp_change')
        bb=choose(layer_corr,task=task,query='Question',metric='roi_enrichment',sensitivity='absolute_logp_change')
        text.append(f'| {task} | L{mean[:,0,-2].argmax()} | L{mean[:,0,3].argmax()} | L{mean[:,1,3].argmax()} | {aa["spearman"]:.3f} | {bb["spearman"]:.3f} |')
    text+=['','## L26 专项','', '| Query | Image Mass [95%CI] | ROI Mass [95%CI] | Hit Ratio [95%CI] | Enrichment [95%CI] |','|---|---|---|---|---|']
    for query in QUERIES:
        vals=[choose(summary,task='Overall',query=query,metric=m,layer=26) for m in METRICS[:4]]
        text.append('| '+query+' | '+' | '.join(ci(v,'mean') for v in vals)+' |')
    mean=cube.mean(0)
    text += ['',f'L26 历史 Mean Absolute LogP Change = **{mean[26,0,-2]:.8f} nat**，Signed Loss = **{mean[26,0,-1]:.8f} nat**。',
        f'CompareAttr 对 L26 总绝对变化之和贡献为 {100*cube[[i for i,s in enumerate(samples) if s["task"]=="CompareAttr"],26,0,-2].sum()/cube[:,26,0,-2].sum():.1f}%；因此不能仅凭总体 L26 峰推广到六种任务。',
        '', '## 逐题峰值与随机对照','']
    for q,query in enumerate(QUERIES):
        r=choose(peak_summary,task='Overall',query=query)
        b=choose(individual_summary,task='Overall',query=query,metric='roi_enrichment',sensitivity='absolute_logp_change')
        signed=choose(layer_corr,task='Overall',query=query,metric='roi_enrichment',sensitivity='signed_logp_loss')
        rand=cube[:,:,q,6].mean(0)
        text += [f'- {query}：逐题 ROI 峰减敏感峰平均 {r["mean_peak_difference"]:.2f} 层 [95%CI {r["ci_low"]:.2f}, {r["ci_high"]:.2f}]，中位数 {r["median_peak_difference"]:.1f}；峰层完全相同 {r["exact_peak_matches"]}/240，差不超过 2 层 {r["within_two_layers"]}/240。',
            f'- {query}：总体均值曲线的 ROI 峰 L{r["mean_curve_roi_peak"]}、敏感峰 L{r["mean_curve_sensitivity_peak"]}，差 {r["mean_curve_peak_difference"]} 层 [图片簇 Bootstrap 95%CI {r["mean_curve_peak_difference_ci_low"]:.0f}, {r["mean_curve_peak_difference_ci_high"]:.0f}]。总体曲线峰差与逐题峰差是不同估计量。',
            f'- {query}：逐题 Enrichment–绝对变化 Spearman 中位数 {b["median_spearman"]:.3f}；{b["undefined_questions"]} 条常量曲线相关无定义，保留并明确标记，不填 0；峰值平局预先规定取最早层，常量 ROI 曲线仍列入全部 240 题峰差统计（{r["constant_roi_curves"]} 条），应谨慎解释。',
            f'- {query}：总体各层随机 patch Enrichment 均值范围 {rand.min():.3f}–{rand.max():.3f}；真实与随机对照见原始逐题表及曲线。Signed Loss 的层均值 Spearman = {ci(signed)}。']
    primary=choose(layer_corr,task='Overall',query='P-1',metric='roi_enrichment',sensitivity='absolute_logp_change')
    within=choose(individual_summary,task='Overall',query='P-1',metric='roi_enrichment',sensitivity='absolute_logp_change')
    fixed=choose(l26,task='Overall',query='P-1',metric='roi_enrichment',sensitivity='absolute_logp_change')
    text += ['', '## 中文研究结论','',
        f'- P−1 的总体层均值 ROI Enrichment 与绝对敏感性有正向描述性关联（ρ={primary["spearman"]:.3f}），但逐题跨层关联很弱（平均 ρ={within["mean_spearman"]:.3f}）。总体曲线的关联不能直接推广为单题功能敏感性的可靠指示器。',
        f'- L26 的 P−1 Enrichment 高于均匀面积基线，但同层跨题相关仅 ρ={fixed["spearman"]:.3f}；Question 的同层关联接近零。没有证据将 L26 的较大 Image OFF 改变直接归因于 ROI 读取。',
        '- 总体 P−1 ROI 富集峰在 L23、Question 富集峰在 L21，不与总体功能敏感峰 L26 重合。ChooseRel/CompareAttr 的 P−1 富集峰与敏感峰均为 L26，但 ChooseRel 全层曲线相关接近零；单个峰重叠不足以说明整条曲线或逐题机制一致。',
        '- 六任务并不一致，LogicalObj 的 P−1 层均值相关为负；总体 L26 敏感峰的主要贡献来自 CompareAttr。应保留分任务和逐题结果，不使用总体均值替代全部任务结论。',
        '', '## 结论边界','',
        'ROI 关注、直接 Image 路径敏感性与预测表现是不同概念。图中的曲线关联及峰值重叠仅为描述性证据；跨层统计未控制所有混杂，不证明某 ROI 路径导致首 Token 改变，更不能推导整层可跳过或答案 Accuracy 不变。',
        'L26 必须同时结合六任务曲线、逐题相关、同层相关和峰差观察，不把总体绝对变化峰自动称为 ROI 功能峰。Image 来源中还包含 ROI 外视觉信息，间接视觉信息也可经其他 Token/残差流传播。',
        '', '## 文件与运行','',
        '- 主图：`overall_roi_alignment`；六任务：`six_task_roi_alignment`；逐题相关/峰差：`sample_correlations_and_peaks`；层均值与 L26 散点：`layer_and_l26_alignment`，均提供 PNG/PDF。',
        '- `per_question.csv`、`layer_summary.csv`、`task_layer_summary.csv`、`layer_curve_correlations.csv`、`per_question_correlations.csv`、`per_question_correlation_summary.csv`、`peak_layers.csv`、`peak_summary.csv`、`l26_correlations.csv`、`l26_layer_summary.csv`。',
        '- `shard_*/rows/*.npz` 只保留 Head/Question 聚合后的 Image Attention 行，不保留完整矩阵；随机种子、输入、框与模型协议均绑定可重算。',
        f'- 采集冻结代码 `{validation["code_commit"]}`。GPU0/2/3 每卡 80 题；正式采集最长 {max(r["formal_seconds"] for r in validation["gpu_runs"]):.2f} 秒，Smoke 最长 {max(r["smoke_seconds"] for r in validation["gpu_runs"]):.2f} 秒，均不含模型加载、CPU 数据准备、Bootstrap 与绘图。',
        f'- 本轮统计阶段 {validation["statistics_seconds"]:.2f} 秒。GPU0/2/3 已退出并释放模型显存，GPU1 未使用。初次标注不足的 partial 审计保留在提交 `812f3e495`，本次在用户授权后补齐，未将此前 partial 当作完成结果。']
    (OUT/'report_zh.md').write_text('\n'.join(text)+'\n')
