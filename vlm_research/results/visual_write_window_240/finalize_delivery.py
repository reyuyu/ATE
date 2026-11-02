"""Presentation-only finalization from audited CSV; frozen execution untouched."""
from pathlib import Path
import csv,json,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path('/data/lk/ATE-worktrees/visual-write-window-240')
p=r/'vlm_research/results/visual_write_window_240'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def table(n):
    with (p/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(p/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

completion=json.loads((p/'completion.json').read_text())
for n,h in completion['artifacts'].items():assert sha(p/n)==h,n
before={n:sha(p/n) for n in completion['artifacts'] if n.endswith('.csv') or n.startswith(('checkpoints/','swap_checkpoints/','smoke_checkpoints/'))}
protocol=json.loads((p/'protocol.json').read_text())
for n,h in protocol['sources'].items():assert sha(r/'vlm_research'/n)==h
assert not subprocess.check_output(['git','diff',protocol['code_commit'],'--','vlm_research/visual_write_window'],cwd=r,text=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
sums=table('accuracy_summary.csv');pairs=table('paired_comparisons.csv');tasks=table('task_accuracy.csv');conds=[v['condition'] for v in sums]
fig,ax=plt.subplots(figsize=(13,5));x=np.arange(18);ys=[float(v['accuracy_pct']) for v in sums]
upper=[float(v['ci95_accuracy_high_pct']) for v in sums];lower=[float(v['ci95_accuracy_low_pct']) for v in sums]
ax.bar(x,ys,color=['#486b89','#999999']+['#ce7258']*8+['#75a5ad']*8)
ax.errorbar(x,ys,yerr=[np.array(ys)-lower,np.array(upper)-ys],fmt='none',color='black',capsize=2)
for k,v in enumerate(sums):ax.text(k,upper[k]+1.2,v['correct'],ha='center',va='bottom',fontsize=8)
ax.axhline(ys[0],ls='--',color='#486b89',label='All-Writer');ax.axhline(ys[1],ls=':',color='gray',label='No-Writer')
ax.set(xticks=x,xticklabels=conds,ylabel='Accuracy (%)',ylim=(0,103));ax.tick_params(axis='x',rotation=50);ax.legend(loc='upper right')
ax.set_title('240 questions per condition; exploratory 95% image-cluster bootstrap intervals',fontsize=10)
fig.tight_layout();save(fig,'figure_accuracy')

primary=[v for v in pairs if v['role'] in ['sufficiency','necessity']]
matrix=np.array([[float(v['delta_pp']) for v in primary if v['role']==role] for role in ['sufficiency','necessity']])
lim=max(1.,abs(matrix).max());fig,ax=plt.subplots(figsize=(13,4.1),layout='constrained')
im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
for y,role in enumerate(['sufficiency','necessity']):
    for k,v in enumerate([v for v in primary if v['role']==role]):
        color='white' if abs(float(v['delta_pp']))>.7*lim else 'black'
        ax.text(k,y,f"{float(v['delta_pp']):+.2f}\n[{float(v['ci95_low_pp']):+.2f}, {float(v['ci95_high_pp']):+.2f}]",ha='center',va='center',fontsize=9,color=color)
ax.set(xticks=range(8),xticklabels=[f'W{k}\nL{4*k}-{4*k+3}' for k in range(8)],yticks=[0,1],yticklabels=['Only - No\nSufficiency','All - Drop\nNecessity'])
ax.set_title('Exploratory 95% intervals | 20,000 paired image-cluster bootstrap resamples',fontsize=10,pad=12)
fig.colorbar(im,ax=ax,label='Accuracy difference (pp)',shrink=.85,pad=.025);save(fig,'figure_window_effects')

tasknames=['ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr'];mats=[]
for role in ['sufficiency','necessity']:
    mats.append(np.array([[next(float(v['delta_no_pp']) if role=='sufficiency' else -float(v['delta_all_pp']) for v in tasks if v['task']==t and v['condition']==('Only-' if role=='sufficiency' else 'Drop-')+f'W{k}') for k in range(8)] for t in tasknames]))
lim=max(1.,max(abs(m).max() for m in mats));fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
for ax,m,title in zip(axes,mats,['Only - No: sufficiency by task','All - Drop: necessity by task']):
    im=ax.imshow(m,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    for y in range(6):
        for k in range(8):
            value=m[y,k];label='0.0' if abs(value)<1e-12 else f'{value:+.1f}'
            ax.text(k,y,label,ha='center',va='center',fontsize=9,color='white' if abs(value)>.7*lim else 'black')
    ax.set(xticks=range(8),xticklabels=[f'W{k}' for k in range(8)],yticks=range(6),yticklabels=tasknames,title=title)
fig.colorbar(im,ax=axes.tolist(),label='Accuracy difference (pp)',shrink=.8,pad=.025);save(fig,'figure_task_effects')

report=p/'report_zh.md';text=report.read_text();assert '## 完成后的科学判读' not in text
extra='''## 完成后的科学判读

**没有发现一个四层窗口能够完整保留All-Writer的表现，但发现了以L8–L11为峰值的不均匀写入能力。** Only-W2为188/240（78.333%），相对No-Writer增加14.583 pp，95%CI [+8.750, +20.417]；它仍比All-Writer少14题，差值−5.833 pp，CI [−10.833, −0.833]，具体为25题受损、11题纠正。全部八个Only条件相对All的普通探索性区间均为负，因此本轮不支持“任意一个四层窗口就足够达到全层表现”。多个窗口联合开放是否足够尚未测试。

L4–L7（Only-W1）和L12–L15（Only-W3）也有部分充分性：相对No分别+6.667 pp [2.083, 11.250]、+4.167 pp [0.417, 8.333]。W0、W4–W7未显示同等强的总体增益，不能将其解释为没有视觉信息流。W7虽然完整答案和首Token Top-1与No-Writer相同，其完整Logits仍可变化，不能由Accuracy相同推导状态相同。

必要性证据同样在W2最强：All−Drop-W2为+4.167 pp [0.417, 7.917]，移除后15题受损、5题纠正，是八个窗口中唯一普通95%区间完全为正的净损害。**这是其他28层开放时的条件必要性线索，不是不可替代或唯一写入节点的证明。** Drop-W2仍有192/240（80.000%），比No-Writer高16.250 pp [9.583, 22.917]。因此其他层能承担相当一部分功能，且Only-W2和Drop-W2都未恢复All的202/240；这与分布不均、部分可替代并需要跨层协作的写入机制相容，尚未定位具体补偿中介。16项窗口比较未作确认性多重校正，边界性的区间尤其不应外推为确认结论。

任务差异也反对“W2普遍充分”：Only-W2相对No在ChooseCat为+40.0 pp、ChooseRel为+22.5 pp、QueryAttr为+10.0 pp；但CompareAttr为−5.0 pp。相对All，Only-W2在QueryAttr下降22.5 pp、CompareAttr下降7.5 pp。移除W2的损害则分布在ChooseAttr、ChooseRel、CompareAttr、LogicalObj及QueryAttr，而ChooseCat净变化为0。全部任务每类仅40题，这些差异为描述性。

两条说明性的反例同时在Only-W2和Drop-W2中出现：`001020372`问“长凳左边的斑马是黑还是白”，Gold与All均为White，两种条件都答Black；`02813015`问“男人在哪一侧”，Gold与All均为Right，两种条件都答Left。保留W2本身或只保留其他窗口，都不足以挽回这些题；这只是逐题协作现象的线索，不能据此证明某种确定交互结构。完整反例没有删去。

对Look Hard, Answer Cheap的机制贡献是：在答案阶段32层都不能直接读取Image Key时，部分中前层R读图仍能显著提高完整VQA准确率，说明视觉读取的功能作用在层间高度不均匀。最有希望的写入候选是L8–L11及其邻近窗口，但本轮没有证明只保留少数层就能无损，也没有证明任何窗口不可替代。后续“稀疏写入、轻量读取”应视为待独立验证的设计方向。本轮未运行任何多窗口联合、Activation Patching、跨模型、Router或Kernel实验。

'''
text=text.replace('## 协议与因果范围\n',extra+'## 协议与因果范围\n',1)
report.write_text(text,encoding='utf-8')
assert all(sha(p/n)==h for n,h in before.items())
audit=dict(passed=True,scope='Presentation only: Chinese interpretation and figure layout/contrast',frozen_execution_unchanged=True,scientific_csv_and_checkpoints_unchanged=True,
    scientific_files_verified=len(before),changed=['report_zh.md']+[f'figure_{stem}.{ext}' for stem in ['accuracy','window_effects','task_effects'] for ext in ['png','pdf']],new_model_generations=0)
(p/'presentation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
completion['delivery_presentation_audit']='presentation_audit.json'
completion['artifacts']={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['completion.json','worker.lock']}
(p/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PRESENTATION PASS: statistics, checkpoint files and frozen sources unchanged')
