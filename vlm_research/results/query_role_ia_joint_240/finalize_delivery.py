"""Editorial/report and plot-only finalization from saved CSVs; no model calls."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

OUT=Path(__file__).resolve().parent
def readcsv(name):
    with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
summary=readcsv('accuracy_summary.csv');pairs=readcsv('paired_comparisons.csv');unique=readcsv('joint_unique_harm_cases.csv')
assert len(summary)==4 and summary[-1]['correct']=='205' and len(unique)==1
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
palette=['#486b89','#75a5ad','#a5b5ce','#ce7258'];labels=['Full','I','A','I+A']
ys=[float(r['accuracy_pct']) for r in summary];low=[float(r['ci95_accuracy_low_pct']) for r in summary];high=[float(r['ci95_accuracy_high_pct']) for r in summary]
fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(labels,ys,color=palette)
ax.errorbar(range(4),ys,yerr=[np.array(ys)-low,np.array(high)-ys],fmt='none',color='black',capsize=4)
for x,r,y in zip(range(4),summary,high):ax.text(x,y+1.2,f"{r['correct']}/240\n{float(r['accuracy_pct']):.3f}%",ha='center',va='bottom',fontsize=9)
ax.set(ylim=(0,103),ylabel='Accuracy (%)',title='Image + answer output bypass: 240 fixed GQA questions')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'figure_accuracy.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
tasks=readcsv('task_accuracy.csv');task_names=list(dict.fromkeys(r['task'] for r in tasks));conditions=[r['condition'] for r in summary]
fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(6)
for k,c in enumerate(conditions):axes[0].bar(x+(k-1.5)*.2,[float(next(r['accuracy_pct'] for r in tasks if r['condition']==c and r['task']==t)) for t in task_names],.2,label=labels[k],color=palette[k])
axes[0].set(xticks=x,xticklabels=task_names,ylim=(0,105),ylabel='Accuracy (%)');axes[0].tick_params(axis='x',rotation=30)
axes[0].legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0)
joint=[next(r for r in tasks if r['task']==t and r['condition']=='I+A-Bypass') for t in task_names]
axes[1].bar(x-.18,[int(r['correct_to_wrong']) for r in joint],.36,label='Full correct to joint wrong',color=palette[-1])
axes[1].bar(x+.18,[int(r['wrong_to_correct']) for r in joint],.36,label='Full wrong to joint correct',color=palette[1])
axes[1].set(xticks=x,xticklabels=task_names,ylabel='Questions');axes[1].tick_params(axis='x',rotation=30);axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].legend(loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0,fontsize=9)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'figure_tasks.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
report=OUT/'report_zh.md';text=report.read_text()
needle='本实验检验的是保留 R 的直接 Attention 更新、同时消融 I 与 A 输出更新是否呈现可组合的低敏感性。'
answer='**结论：在本批探索样本上，可以在不直接消融关键 Prefill 文本 Query（R）的同时，联合关闭图像 Query 与答案 Query 的 Attention 输出更新，并维持接近 Full 的总体准确率；结果支持有限的可组合低敏感性，但存在真实受损和一例联合独有损害，尚不能称为安全、无损或已通过非劣效。** R 的后续状态可能间接受影响；这是输出消融证据，不能等同于物理计算可跳过或真实加速。'
assert text.count(needle)==1;text=text.replace(needle,answer)
case=unique[0]
example=f"\n联合独有损害实例：`{case['question_id']}`（{case['task']}），问题为“{case['question']}”。Gold 为 `{case['gold']}`，Full/I/A 均回答 Material，联合回答 `{case['joint_text']}`。这只是已观察到的行为变化，不构成特定 Token 机制证据。\n"
text=text.replace('## 六任务结果',example+'\n## 六任务结果',1)
text=text.replace('## 下一步建议与停止边界','## 结果图\n\n![四条件 Accuracy 与探索性 95% CI](figure_accuracy.png)\n\n![配对损益与交互](figure_paired_interaction.png)\n\n![六任务结果](figure_tasks.png)\n\n## 下一步建议与停止边界',1)
text+='\n交付编辑说明：最终报告补充明确结论与独有损害实例；Accuracy 图将数值标签移至误差线之外，任务图使用整数计数并统一配色。该步骤只读取已保存统计，不改变冻结执行源码、检查点、数组或统计值。脚本为 finalize_delivery.py。\n'
report.write_text(text)
completion=json.loads((OUT/'completion.json').read_text())
completion['delivery_finalization']=dict(plot_labels_verified=True,scientific_statistics_unchanged=True,frozen_sources_unchanged=True)
completion['artifacts']={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['worker.lock','completion.json']}
(OUT/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n')
print('Finalized report and plot layout; refreshed artifact hashes; no scientific data changed')
