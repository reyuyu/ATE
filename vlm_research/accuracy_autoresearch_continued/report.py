"""Descriptive reporting for the explicitly non-independent continuation."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .runtime import *


def rows(path):
    with path.open() as f: return list(csv.DictReader(f))


def main():
    selection=read(OUT/'selection.json'); primary=selection['primary']
    quality=rows(OUT/'analysis/05_recheck/accuracy_summary.csv')
    perf=rows(OUT/'analysis/06_recheck_performance/latency_flops_kv.csv')
    paired=rows(OUT/'analysis/06_recheck_performance/performance_paired.csv')
    candidates=[config('native'),config(),config(answer='both'),primary]
    names=['Native','C1','C3-V2','Continued selection']
    if selection.get('speed_secondary'):
        candidates.append(selection['speed_secondary']); names.append('Speed secondary')
    count=len(candidates)
    combined=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,d in enumerate(('GQA','VQAv2')):
        values=[]; lows=[]; highs=[]; speeds=[]
        for c,label in zip(candidates,names):
            h=dict(c,logits_to_keep=1); cid=config_id(h)
            q=next(r for r in quality if r['dataset']==d and r['config_id']==cid)
            p=next(r for r in perf if r['dataset']==d and r['config_id']==cid)
            speed=next(r for r in paired if r['dataset']==d and r['condition']==cid and r['baseline']==config_id(config('native',head=1)))
            combined.append(dict(label=label,**q,**{k:v for k,v in p.items() if k not in q},
                **{k:v for k,v in speed.items() if k not in q and k not in p}))
            values.append(float(q['delta_native_pp'])); lows.append(float(q['delta_ci_low'])); highs.append(float(q['delta_ci_high']))
            speeds.append(float(speed['all_e2e_speedup']))
        axes[0,col].errorbar(range(count),values,yerr=[np.array(values)-lows,np.array(highs)-values],fmt='o',capsize=4,color='#16786a')
        axes[0,col].axhline(0,lw=.8,color='.5'); axes[0,col].axhline(-1,ls='--',color='#b83c37',lw=1)
        axes[0,col].set_title(d+' (previously inspected cohort)')
        axes[0,col].set_ylabel('Accuracy difference (pp, H1)')
        axes[1,col].bar(range(count),speeds,color=['#777777','#c59632','#c59632','#16786a','#497ca5'][:count])
        axes[1,col].axhline(1.1,ls='--',color='#b83c37',lw=1)
        axes[1,col].set_ylim(0,max(speeds)*1.2); axes[1,col].set_ylabel('Measured E2E speedup (H1)')
        for i,v in enumerate(speeds): axes[1,col].text(i,v+.02,f'{v:.2f}x',ha='center',fontsize=9)
        for ax in axes[:,col]:
            ax.set_xticks(range(count),names,rotation=20,ha='right',fontsize=9)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'quality_speed.png',dpi=240); fig.savefig(OUT/'quality_speed.pdf'); plt.close(fig)
    write_csv(OUT/'quality_performance_H1.csv',combined)
    # The complete original DEV grid is descriptive; missing conditions are never filled in.
    pool=read(PRIOR/'sample_manifest.json'); parts=read(PRIOR/'split_manifest.json'); depth=[]
    devkeys=[k for k,s in pool.items() if parts['image_split'][s['image_sha256']]=='DEV']
    for d in ('GQA','VQAv2'):
        keys=[k for k in devkeys if pool[k]['dataset']==d]
        for cutoff in (20,24,28,32):
            for mode in ('none','decode'):
                c=config('all',mode,cutoff); records=[read_record(c,k,pool[k]) for k in keys]
                assert all(r is not None for r in records)
                depth.append(dict(dataset=d,cutoff=cutoff,answer_attention=mode,n=len(keys),
                    accuracy=100*np.mean([r['generation']['score'] for r in records])))
    write_csv(OUT/'DEV_depth_comparison.csv',depth)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,d in zip(axes,('GQA','VQAv2')):
        for mode,color in (('none','#ba593e'),('decode','#16786a')):
            v=[r for r in depth if r['dataset']==d and r['answer_attention']==mode]
            ax.plot([r['cutoff'] for r in v],[r['accuracy'] for r in v],'o-',label=mode,color=color)
        ax.set_xticks([20,24,28,32]); ax.set_title(d+' DEV'); ax.set_xlabel('Physical handoff boundary (32: no in-layer exit)')
        ax.set_ylabel('Accuracy (%)'); ax.legend(); ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'DEV_depth_comparison.png',dpi=240); fig.savefig(OUT/'DEV_depth_comparison.pdf'); plt.close(fig)
    changes=[]
    for k in read(OUT/'jobs/05_recheck.json')['samples']:
        s=pool[k]
        for candidate in [primary]+([selection['speed_secondary']] if selection.get('speed_secondary') else []):
            a=read_record(candidate,k,s)['generation']
            for c in (config(answer='both'),config('qsa','decode')):
                b=read_record(c,k,s)['generation']
                changes.append(dict(sample=k,dataset=s['dataset'],category=s.get('answer_type',s['task']),
                    task=s['task'],condition=config_id(candidate),baseline=config_id(c),prediction=a['text'],baseline_prediction=b['text'],
                    score=a['score'],baseline_score=b['score'],harm=a['score']<b['score'],rescue=a['score']>b['score'],
                    delta=a['score']-b['score'],question=s['question'],image_sha256=s['image_sha256']))
    write_csv(OUT/'case_changes.csv',changes)
    lines=['# 六小时原时钟内：继续探索结果','',
        '**本轮是已授权的探索续篇，不是新的独立确认。** 旧FINAL已被查看，续篇仅用原DEV/CHECK选型，随后在同一历史队列RECHECK；不能把普通配对CI称为新的非劣效证明。','',
        '## 冻结选择','',f"- 配置：`{config_id(primary)}`；冻结时间：{selection['at']}。",
        f"- 速度备选：`{config_id(selection['speed_secondary']) if selection.get('speed_secondary') else '无'}`；选择附录在读取本轮计时分数前记录。",
        '- 仅LLaVA固定FP16/eager，P−1和全部真实Decode仍在所有层禁止直接访问原始Image Key。',
        '- 检查4个深度/答案阶段扩展配置，其中L32/no-bypass为原粗筛诊断的扩大检查：新增3种不同候选，累计15种；未重扫Token或Prompt规则、未训练。',
        '- L32仅为无层内视觉Hidden退出的容量诊断，不默认称为真实加速方案。',
        '- 原时钟04:31:15启动、08:31:15冻结线、10:01:15 GPU线、10:31:15交付线全部沿用。','',
        '## RECHECK质量：原始H0','', '| 数据集 | 条件 | 题数 | Accuracy % | ΔNative pp [普通配对95%CI] | 损害/纠正 |',
        '|---|---|---:|---:|---|---|']
    for d in ('GQA','VQAv2'):
        for c,label in zip(candidates,names):
            q=next(r for r in quality if r['dataset']==d and r['config_id']==config_id(c))
            lines.append(f"| {d} | {label} | {q['n']} | {float(q['accuracy']):.3f} | {float(q['delta_native_pp']):+.3f} [{float(q['delta_ci_low']):+.3f}, {float(q['delta_ci_high']):+.3f}] | {q['score_down']}/{q['score_up']} |")
    lines+=['','## 独立H1质量与统一计时','',
        '以下质量和性能均来自同一H1工程绑定。每数据集20题，同GPU、冷KV、重算Vision、3预热＋5次条件交错重复。先取逐题中位数再平均；与旧延迟不拼接。','',
        '| 数据集 | 条件 | H1 Accuracy % | Prefill ms | E2E ms | Decode ms/token | E2E加速 | 实际矩阵TFLOPs | KV Live/Capacity MiB | Peak Allocated MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in combined:
        lines.append(f"| {r['dataset']} | {r['label']} | {float(r['accuracy']):.3f} | {float(r['language_prefill_ms']):.2f} | {float(r['latency_ms']):.2f} | {float(r['decode_ms_per_token']):.2f} | {float(r['all_e2e_speedup']):.3f}x | {float(r['total_flops'])/1e12:.3f} | {float(r['kv_live_mib']):.2f}/{float(r['kv_capacity_mib']):.2f} | {float(r['peak_allocated'])/2**20:.1f} |")
    lines+=['','## 阅读与复现','',
        '- 20,000次图片SHA256整簇配对Bootstrap，固定原seed20261105，题目等权；全部区间为探索性。',
        '- GQA为原冻结评分器；VQAv2为官方十答案软评分，损害/纠正表示软分降低/提高。',
        '- 所有单独Matrix FLOPs来自实际执行Shape，2 FLOPs/MAC，未把Dense Mask当节省；硬件TFLOPS/带宽N/A。',
        '- E2E不含Processor/I/O/布局定位，含冷KV推理和重算Vision；TTFT是CUDA模型阶段。',
        '- 共享模型、图片、旧清单/原子Checkpoint仅引用未复制，旧研究路径与main不改动。',
        '- 原始清单/拆分与基线来自上一Commit `60d262f83a981e49a614ef8213e5311c3c7c6469`，对应服务器路径记录在`execution_freeze.json`。',
        '- 执行：`python -m vlm_research.accuracy_autoresearch_continued.runtime --rank 0/1`，分别只暴露冻结的GPU2/3 UUID。使用旧时钟，不得删除截止或结束标记绕过预算。',
        '- 统计：`manage analyze 05_recheck`、`manage performance 06_recheck_performance`；检查：`finish audit`。新GPU重现需要另行授权和独立运行目录。',
        '- 逐题答案、Token、EOS等在`analysis/05_recheck/per_question.csv`及原子Checkpoint；对C3和旧主候选的全部损益在`case_changes.csv`。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    print('Continuation tables, figures and draft report written; append actual reviewed conclusions.')


if __name__=='__main__':
    configure(); main()
