"""Measured layer curves and Chinese report; no GPU imports or generation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.environ['MPLCONFIGDIR']='/data/lk/ATE-resources/experiments/textvqa_single_layer_visual_ablation/mplconfig'
import pathlib,json,csv,collections,datetime,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .policy import ROOT,OUT,SEED,MIDDLE
from receiver_cross_dataset.common import read,write,sha
def rows(p):return list(csv.DictReader(p.open()))
def f(r,k):return float(r[k])
COLORS={'All-Text':'#386CB0','Ponly':'#C05A26','Donly':'#198474','Both':'#8255A5','Middle-minus':'#2C668C','Addback':'#8B6324'}
def tab(rr,fields):
 out=['| '+' | '.join(title for _,title in fields)+' |','|'+'|'.join(['---']*len(fields))+'|']
 for r in rr:
  vals=[]
  for key,title in fields:
   v=r[key];vals.append(f'{float(v):.3f}' if key in ['accuracy_pct','delta_pp','CI95_low_pp','CI95_high_pp','p_Holm','reference_accuracy_pct','mean_layer_delta_pp','worst_layer_delta_pp'] else str(v))
  out.append('| '+' | '.join(vals)+' |')
 return '\n'.join(out)
def main():
 stat=rows(OUT/'accuracy_by_layer.csv');base=rows(OUT/'baseline_accuracy.csv');av=read(OUT/'analysis_validation.json');account=read(OUT/'generation_accounting.json');cov=rows(OUT/'coverage.csv');F=OUT/'figures';F.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':130,'savefig.dpi':230,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 figures=[]
 def choose(family):
  rr=[r for r in stat if r['family']==family];full=[r for r in rr if r['sample_set']=='Full300'];pilot=[r for r in rr if r['sample_set']=='Pilot120']
  # Never silently mix sample sizes in one curve. Use largest complete family.
  expected=11 if family=='Middle-minus' else 32
  if len(full)==expected:return sorted(full,key=lambda r:int(r['layer']))
  return sorted(pilot,key=lambda r:int(r['layer'])) if len(pilot)==expected else sorted(full or pilot,key=lambda r:int(r['layer']))
 def save(fig,name):
  fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig);figures.append(name)
 familyselected={}
 for family in ['All-Text','Ponly','Donly','Both','Middle-minus','Addback']:
  rr=choose(family)
  if not rr:continue
  familyselected[family]=rr;x=np.array([int(r['layer']) for r in rr]);acc=np.array([f(r,'accuracy_pct') for r in rr]);delta=np.array([f(r,'delta_pp') for r in rr]);n=rr[0]['n'];ref=rr[0]['reference'];baseline=f(rr[0],'reference_accuracy_pct');col=COLORS[family]
  fig,axes=plt.subplots(2,1,figsize=(11.8,6.7),sharex=True,gridspec_kw={'height_ratios':[1,1]},layout='constrained')
  title={'All-Text':'One layer: R + P-1 + true Decode image access removed','Ponly':'One layer: only P-1 image access removed','Donly':'One layer: only true Decode image access removed','Both':'One layer: P-1 + true Decode image access removed','Middle-minus':'COCO-Middle: leave one allowed Decode layer out','Addback':'Decode-None: restore one Decode image-reading layer'}[family]
  fig.suptitle(f'{title}\nTextVQA | n={n} images | official soft accuracy',fontsize=13)
  al=np.array([f(r,'accuracy_CI95_low_pct') for r in rr]);ah=np.array([f(r,'accuracy_CI95_high_pct') for r in rr]);dl=np.array([f(r,'CI95_low_pp') for r in rr]);dh=np.array([f(r,'CI95_high_pp') for r in rr])
  axes[0].errorbar(x,acc,yerr=[np.maximum(0,acc-al),np.maximum(0,ah-acc)],fmt='o',markersize=4,color=col,ecolor=col,elinewidth=.8,capsize=2,label='Measured accuracy; pointwise 95% CI')
  axes[0].axhline(baseline,color='#454545',ls='--',lw=1.1,label=f'{ref}: {baseline:.3f}%');axes[0].set_ylabel('TextVQA accuracy (%)');axes[0].legend(loc='best',fontsize=8)
  axes[1].errorbar(x,delta,yerr=[np.maximum(0,delta-dl),np.maximum(0,dh-delta)],fmt='o',markersize=4,color=col,elinewidth=.8,capsize=2,label='Measured paired effect; pointwise 95% CI');axes[1].axhline(0,color='#454545',ls='--',lw=1)
  axes[1].set_ylabel(f'Delta vs {ref} (pp)');axes[1].set_xlabel('Intervened layer (0-based)');axes[1].legend(loc='best',fontsize=8)
  for ax in axes:
   ax.axvspan(10.5,20.5,color='#4E9A77',alpha=.07);ax.grid(axis='y',alpha=.18);ax.set_xlim(-.8,31.8);ax.set_xticks(range(32));ax.tick_params(axis='x',labelsize=8)
   if family=='Middle-minus':ax.set_xlim(-.8,20.8);ax.set_xticks(MIDDLE)
  axes[1].text(.01,.04,'20,000 paired image-cluster bootstrap draws. Ordinary CIs; Holm-adjusted p-values in CSV.\nMeasured points only; shaded band L11-L20. Reused exploratory cohort.',transform=axes[1].transAxes,fontsize=7.5,va='bottom')
  save(fig,f'{family.lower()}_layer_curve')
 # Four Native curves, same y-axis; every point comes from the sample size in its label.
 fig,ax=plt.subplots(figsize=(12,5.6),layout='constrained')
 for j,family in enumerate(['All-Text','Ponly','Donly','Both']):
  rr=familyselected.get(family,[])
  if not rr:continue
  x=np.array([int(r['layer']) for r in rr],float)+(j-1.5)*.10;y=np.array([f(r,'delta_pp') for r in rr]);lo=np.array([f(r,'CI95_low_pp') for r in rr]);hi=np.array([f(r,'CI95_high_pp') for r in rr])
  ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt=['o','s','^','D'][j],markersize=3.7,color=COLORS[family],elinewidth=.7,capsize=1.5,label=f'{family} (n={rr[0]["n"]})')
 ax.axhline(0,color='#444',ls='--',lw=1);ax.axvspan(10.5,20.5,color='#4E9A77',alpha=.1,label='L11-L20');ax.set(xlim=(-.8,31.8),xticks=range(32),xlabel='Intervened layer (0-based; small offsets for visibility)',ylabel='Paired delta vs Native Full (pp)',title='TextVQA: single-layer visual access ablations by query stage\nOrdinary pointwise 95% paired image-bootstrap CIs (20,000 draws)');ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16));save(fig,'native_stage_overlay')
 # Descriptive OCR heatmaps use each family's selected sample size, never reclassify.
 gr=rows(OUT/'ocr_group_accuracy.csv');categories=sorted({r['OCR_group'] for r in gr});fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
 vals=[]
 for family in ['All-Text','Ponly','Donly','Both']:
  rr=familyselected.get(family,[]);sample=rr[0]['sample_set'] if rr else '';matrix=np.full((len(categories),32),np.nan)
  for r in gr:
   if r['family']==family and r['sample_set']==sample:matrix[categories.index(r['OCR_group']),int(r['layer'])]=f(r,'delta_pp')
  vals.append(matrix)
 bound=max(1,max(float(np.nanmax(np.abs(a))) for a in vals if np.isfinite(a).any()))
 for ax,family,a in zip(axes,['All-Text','Ponly','Donly','Both'],vals):
  im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-bound,vmax=bound)
  rr=familyselected.get(family,[]);sample=rr[0]['sample_set'] if rr else '';n=rr[0]['n'] if rr else '0'
  counts={g:next((r['n'] for r in gr if r['family']==family and r['sample_set']==sample and r['OCR_group']==g),'0') for g in categories}
  ax.set_yticks(range(len(categories)),[g+' (n='+counts[g]+')' for g in categories],fontsize=8)
  ax.set_title(family+' | n='+str(n)+' | frozen prior OCR labels (descriptive)',loc='left',fontsize=10)
 axes[-1].set_xticks(range(32));axes[-1].set_xlabel('Intervened layer (0-based)');fig.colorbar(im,ax=axes,shrink=.8,label='Delta vs Native Full (pp)');save(fig,'ocr_group_layer_heatmaps')
 full=[r for r in stat if r['sample_set']=='Full300'];rank=[]
 for family in ['All-Text','Ponly','Donly','Both','Middle-minus','Addback']:
  rr=familyselected.get(family,[])
  if not rr:continue
  worst=min(rr,key=lambda r:f(r,'delta_pp'));best=max(rr,key=lambda r:f(r,'delta_pp'))
  rank.append(dict(family=family,n=worst['n'],worst_layer=worst['layer'],worst_tied_layers=[int(r['layer']) for r in rr if abs(f(r,'delta_pp')-f(worst,'delta_pp'))<1e-10],worst_acc=f(worst,'accuracy_pct'),worst_delta=f(worst,'delta_pp'),worst_Holm=f(worst,'p_Holm'),best_layer=best['layer'],best_tied_layers=[int(r['layer']) for r in rr if abs(f(r,'delta_pp')-f(best,'delta_pp'))<1e-10],best_acc=f(best,'accuracy_pct'),best_delta=f(best,'delta_pp'),Holm_significant_layers=[int(r['layer']) for r in rr if f(r,'p_Holm')<.05],loss_gt2pp_layers=[int(r['layer']) for r in rr if f(r,'delta_pp')< -2-1e-10]))
 rankby={r['family']:r for r in rank};donly={int(r['layer']):r for r in familyselected.get('Donly',[])};mid={int(r['layer']):r for r in familyselected.get('Middle-minus',[])}
 ponly={int(r['layer']):r for r in familyselected.get('Ponly',[])}
 phase_profile={}
 if len(ponly)==len(donly)==32:
  ptop=sorted(ponly,key=lambda k:(f(ponly[k],'delta_pp'),k))[:5];dtop=sorted(donly,key=lambda k:(f(donly[k],'delta_pp'),k))[:5]
  px=np.array([f(ponly[k],'delta_pp') for k in range(32)]);dx=np.array([f(donly[k],'delta_pp') for k in range(32)])
  correlation=float(np.corrcoef(px,dx)[0,1]) if px.std()>0 and dx.std()>0 else None
  phase_profile=dict(Ponly_five_most_negative_layers=ptop,Donly_five_most_negative_layers=dtop,overlap=sorted(set(ptop)&set(dtop)),curve_Pearson_descriptive=correlation,no_layer_independence_or_significance_claim=True)
 contrasts=[]
 for layer in MIDDLE:
  if layer in donly and layer in mid:
   a,b=donly[layer],mid[layer]
   if a['sample_set']!=b['sample_set'] or a['n']!=b['n']:continue
   contrasts.append(dict(layer=layer,Native_Donly_delta_pp=f(a,'delta_pp'),Middle_minus_delta_pp=f(b,'delta_pp'),Native_n=int(a['n']),Middle_n=int(b['n']),context_amplification_pp=f(b,'delta_pp')-f(a,'delta_pp'),Native_near_zero_abs_le1pp=abs(f(a,'delta_pp'))<=1,Middle_loss_gt2pp=f(b,'delta_pp')< -2-1e-10))
 from .analysis import csvwrite
 csvwrite(OUT/'native_vs_middle_leave_one_out.csv',contrasts)
 regions=[]
 for family in ['All-Text','Ponly','Donly','Both']:
  rr=familyselected.get(family,[])
  for region,lo,hi in [('Early L0-L10',0,10),('Middle L11-L20',11,20),('Late L21-L31',21,31)]:
   selected=[r for r in rr if lo<=int(r['layer'])<=hi]
   if selected:regions.append(dict(family=family,sample_set=selected[0]['sample_set'],n=int(selected[0]['n']),region=region,layer_count=len(selected),mean_layer_delta_pp=float(np.mean([f(r,'delta_pp') for r in selected])),worst_layer_delta_pp=min(f(r,'delta_pp') for r in selected),best_layer_delta_pp=max(f(r,'delta_pp') for r in selected),layers_loss_gt2pp=[int(r['layer']) for r in selected if f(r,'delta_pp')< -2-1e-10],interpretation='descriptive layer-profile summary; layers not independent samples; no region significance test'))
 csvwrite(OUT/'layer_region_profile.csv',regions)
 summary=dict(family_extrema=rank,phase_profile=phase_profile,region_profile=regions,mandatory_conditions_complete300=av['mandatory_conditions_complete300'],optional_conditions_complete300=av['optional_conditions_complete300'],figures=figures,Native300=next(r for r in base if r['sample_set']=='Full300' and r['condition']=='Native'),Middle300=next(r for r in base if r['sample_set']=='Full300' and r['condition']=='Middle'),near_zero_Native_large_Middle=[r for r in contrasts if r['Native_near_zero_abs_le1pp'] and r['Middle_loss_gt2pp']])
 write(OUT/'report_summary.json',summary)
 header='\n'.join(f'- {r["family"]}：最小Δ的层{r["worst_tied_layers"]}，ACC {r["worst_acc"]:.3f}%，Δ{r["worst_delta"]:+.3f}pp；最大Δ的层{r["best_tied_layers"]}，Δ{r["best_delta"]:+.3f}pp。Holm<0.05层：{r["Holm_significant_layers"]}。' for r in rank)
 native300=float(summary['Native300']['accuracy_pct']);middle300=float(summary['Middle300']['accuracy_pct']);mandatory=av['mandatory_conditions_complete300'];optional=av['optional_conditions_complete300']
 txt=f'''# TextVQA单层视觉读取消融：阶段差异与Middle留一

## 完成范围与直接发现

固定历史TextVQA300（300张不同图片），官方十答案留一共识软评分。新增{av['new_checkpoints']}份原子Checkpoint；主矩阵139个条件中{mandatory}个完成300题；可选单层Add-back中{optional}/32个完成300题。历史1200条Native/Middle/Decode-None/Both-None记录只读审计复用。没有训练、扩样、修改Prompt、筛选不利输出或物理删除KV。

生成计数边界：新增矩阵记录中{account['retained_smoke_in_matrix']}条来自通过门禁后保留的Smoke处理轨迹；常规正式新增{account['fresh_formal_outside_smoke']}条。双卡Smoke及校验实际执行{account['mandatory_dual_GPU_smoke_trajectories'] + account['optional_dual_GPU_smoke_trajectories']}条轨迹，其中未保留为新矩阵记录的重放不增加评分样本数。模型实际生成总轨迹{account['total_model_generation_trajectories']}条，详见`generation_accounting.json`；双卡相同3题Smoke是明确要求的校验重复，正式矩阵没有重复Question×Condition。

Native Full为{native300:.3f}%，Middle为{middle300:.3f}%。下列排序仅描述本次反复研究的探索样本，不能当作独立确认或据此宣称一组层可安全联合剪除。

{header}

## 冻结设计

执行源码先冻结为`f8f6f0550b6b0575ce6eb2f973f37b98c3367280`，继承历史交付`c43f0f075989318c6bc061e411d4d2e703f8e9b8`。模型LLaVA-1.5-7B，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/Eager，Transformers4.49.0，Greedy，`logits_to_keep=1`，max_new_tokens32，真实EOS和独立KV Cache。

All-Text仅在目标层关闭R、P−1、全部真实Decode→原始Image Key；Ponly仅P−1；Donly仅真实Decode；Both仅P−1+Decode。其他31层原生；R严格为图像之后至P−2。图像前系统文本保持原生。Middle留一从开放集合{MIDDLE}中逐次去掉一层，仅作用于Decode，完整Prefill和P−1原生。可选Add-back从Decode-None恢复一层，仅测试充分性。

命名注意：本轮`Ponly-Ll`/`Donly-Ll`表示从Native只关闭一个指定层的指定阶段，绝不是历史预算实验中“只开放k层、另一阶段全关闭”的Ponly-k/Donly-k。新旧条件不混用评分。

所有干预均在Softmax前把指定边置为`−inf`，保留图像Query、其余文本边、完整QKV/KV、MLP、Norm、Residual、RoPE。没有o_proj消融，未实现物理KV裁剪、减少实际计算或测量加速。

300张图片按Image SHA排序后交替分至物理GPU2/3，各150张。120题队列为SHA256(Question ID + `|` + Image SHA)最小120项；不读取Gold、OCR类别、历史正确性或模型分数。其GPU分布54/66，是全300的子集。120题已完成记录按相同源码、协议、输入哈希复用至300，未重新生成。

## 完整层曲线

所有图只画实测点，普通逐点95%CI；不平滑、不插值。绿色带标记L11–L20。上下两图分别为绝对软Accuracy及相对同样本基线的配对ΔAccuracy。样本量在每张图及CSV中注明；若某族仅有120题，绝不会标为300题。

''' + '\n\n'.join(f'![{name}](figures/{name}.png)\n\n[矢量PDF](figures/{name}.pdf)' for name in figures)+'''

## 全部实测层点与配对结果

普通CI与Holm是两种不同输出：CI未作多重校正；p值是20,000次图片整簇配对符号翻转、加一修正，然后对预声明族做Holm。Native四阶段共128个比较为一族，Middle11个为另一族；可选Add-back32个为独立探索族。120/300分别计算，不能把重复观察当成新确认。区间跨0不是无损或非劣效证据。

'''
 fields=[('layer','Layer'),('accuracy_pct','ACC%'),('delta_pp','Δpp'),('CI95_low_pp','CI低'),('CI95_high_pp','CI高'),('score_down','软分下降'),('score_up','软分上升'),('p_Holm','Holm p')]
 for family,rr in familyselected.items():txt+=f'\n### {family}（n={rr[0]["n"]}，对照{rr[0]["reference"]}）\n\n'+tab(rr,fields)+'\n'
 txt+='''
## 阶段差异与联合效应

`phase_interactions.csv`提供逐层Both−Ponly−Donly+Native的观察性交互对比以及All-Text−Both。后一对比是在同层P−1与Decode均禁图条件下追加R禁图的条件效应，不是独立R-only实验；不能将其直接解释为没有上下文条件的Prefill写入贡献。

Ponly修改P−1的更新，也可能改变其随后层的缓存状态，因此即使首个生成Token仍是相同空格，后续答案仍可能变化；它不只影响可见答案的第一个词。Donly保持完整Prefill和首位置Logits严格一致，测量的则是后续真实增量Query的视觉回看干预。首个生成Token可能只是空格，因此Donly首预测Logits严格不变，并不保证首个可见答案词不变；逐题结果另外保留首个有内容Subword。

`native_vs_middle_leave_one_out.csv`逐层比较Native背景Donly和Middle背景留一。前者只关闭一层、仍有31层Decode可读图；后者其余可读图层只剩10层，干预背景显著不同。单层必要性依赖上下文，不能从Native单层小影响直接推导多层联合删除安全。

'''
 contrasts_bad=summary['near_zero_Native_large_Middle']
 if contrasts_bad:txt+='本批存在Native单层|Δ|≤1pp、Middle留一损失>2pp的层：'+str([r['layer'] for r in contrasts_bad])+'。这直接提示层间替代或协作依赖干预背景，但不能从Accuracy单独区分具体机制。\n'
 else:txt+='按描述阈值Native单层|Δ|≤1pp且Middle留一损失>2pp，本批未观察到符合条件的层。阈值仅用于说明反例，未作为选层或显著性检验。\n'
 txt+='''
## 固定OCR分组与案例

`ocr_group_accuracy.csv`保留OCR-Direct230、OCR-Reasoning12、Non-OCR Visual18、Ambiguous40的冻结标签及全部层点。标签来自历史输入盲审AI分类，并非独立人工真值；小组结果只有描述性区间，不进行额外显著性或重新挑层。组间结果不应外推为总体机制比例。

`negative_cases.csv`含全部新增条件相对其基线的软分下降记录；同一道题可在多个条件重复出现。`positive_cases.csv`含全部软分上升记录。`per_question.csv`保留完整原文答案、Token IDs、Gold十答案、官方软分、首Token与首个有内容Subword变化、EOS/截断/空答案/重复及Checkpoint SHA。所有输出异常保留评分，没有因为不利而删题。第一个生成Token可能为空格，故独立报告首个有内容Token。

'''
 per=rows(OUT/'per_question.csv');new=[r for r in per if r['origin']=='new generation'];ordered=sorted(new,key=lambda r:(float(r['delta_score_vs_reference']),r['condition'],r['question_id']));chosen=[]
 for seq in [ordered,list(reversed(ordered))]:
  seen=set()
  for r in seq:
   if r['question_id'] in seen:continue
   seen.add(r['question_id']);chosen.append(r)
   if len(seen)>=4:break
 by={(r['question_id'],r['condition']):r for r in per}
 for r in chosen:
  ref=by[(r['question_id'],r['reference'])]
  txt+=f'- Q{r["question_id"]}，{r["condition"]}：{r["question"]}；对照答“{ref["answer"]}”（{float(ref["score"]):.3f}），干预答“{r["answer"]}”（{float(r["score"]):.3f}），Δ软分{float(r["delta_score_vs_reference"]):+.3f}。\n'
 txt+=f'''

## 正确性与资源

7项CPU测试通过，覆盖全部干预及基线的目标层、阶段、No-op与非目标连接。两卡各自使用同3题（39419、34726、39145）进行真实7B Smoke，Native/No-op全部有效Logits与HF严格`torch.equal`；历史Native、Middle、Decode-None完整Token及首向量逐位复现。Donly全部32层首向量严格等于Native，Ponly与Both同层首向量原始FP16 SHA相同。禁止的真实Attention概率严格0，非目标Mask完全相同，原生图像Query/其余Prefill保持；32层MLP/Norm/Block执行，绝对位置/RoPE/KV长度、独立Cache及父Cache不变、输入/视觉特征/权重版本、正常和异常Hook清理均验证。普通行和仅使用历史冻结ROW_ATOL做FP16归一化检查；该容差从未用于Logits相等或禁止概率为0的要求。

正式生成不保存全层Attention大矩阵，仅保留每层每阶段Mask数量、形状、位置及Cache审计；完整概率审计集中Smoke。全部{av['new_checkpoints']}新增Checkpoint和1200历史记录由CPU重算官方分数，最大差异{av['score_max_difference']}。`analysis_validation.json`、`validation.json`记录独立复核；`gpu_release_audit.json`记录进程结束后的真实资源状态。原先worktree稀疏检出初始化曾因Git info目录缺失失败，已在CPU准备阶段创建目录后修复，发生在源码冻结和任何GPU加载之前；不涉及科学协议、模型、数值或历史结果修改。

任务总墙钟最多6小时，自2026-11-05 13:17:30北京时间起算，GPU不晚于18:47:30停止启动新生成，最终截止19:17:30。恢复不重置时钟。实际时间与可选实验决策见worker_complete、execution_exit及progress日志。资源只用GPU2/3，外部GPU0/1任务未主动干预；不修改共享环境、不复制权重、不下载图片、不修改main或历史结果。

## 机制解释与最终回答

'''
 if 'All-Text' in rankby:
  a=rankby['All-Text'];p=rankby.get('Ponly');d=rankby.get('Donly');b=rankby.get('Both');m=rankby.get('Middle-minus')
  txt+=f'综合文本视觉访问最敏感的实测层为All-Text L{a["worst_layer"]}（Δ{a["worst_delta"]:+.3f}pp，n={a["n"]}）；P−1最敏感点为L{p["worst_layer"]}（{p["worst_delta"]:+.3f}pp），Decode为L{d["worst_layer"]}（{d["worst_delta"]:+.3f}pp）。这些排序是本批点估计，是否超出多重比较噪声应结合上表Holm。损失超过2pp的层分别为All-Text {a["loss_gt2pp_layers"]}、Ponly {p["loss_gt2pp_layers"]}、Donly {d["loss_gt2pp_layers"]}、Both {b["loss_gt2pp_layers"]}，应据这些完整分布判断是否集中中层，不能预设“中层最重要”。\n\n'
  for family0 in ['All-Text','Ponly','Donly','Both']:
   losses=rankby[family0]['loss_gt2pp_layers'];inside=[k for k in losses if 11<=k<=20];outside=[k for k in losses if not 11<=k<=20]
   txt+=f'{family0}点估计损失>2pp：L11–L20内{inside}，区间外{outside}。该阈值描述和排序均不替代Holm检验。\n\n'
  regionmaps={family0:{r['region']:r['mean_layer_delta_pp'] for r in regions if r['family']==family0} for family0 in ['All-Text','Ponly','Donly','Both']}
  mid_larger=all(v['Middle L11-L20']<min(v['Early L0-L10'],v['Late L21-L31']) for v in regionmaps.values())
  if mid_larger:txt+='本批点估计支持敏感性较集中于中层：四条曲线在L11–L20的平均单层Δ均比早段L0–L10和晚段L21–L31更负。但这是相关的层点之描述，不能证明中层唯一重要；各单层的多重校正结论仍以上表为准。\n\n'
  else:txt+='本批没有出现四条曲线都在中层平均最敏感的一致模式；应按各阶段的完整层分布解释，不能预设中层优势。\n\n'
  txt+='各固定层段内单层Δ的描述性均值如下（不是联合移除效果，也不将层视为独立样本）：\n\n'
  txt+=tab(regions,[('family','条件'),('region','层段'),('n','题数'),('mean_layer_delta_pp','层点平均Δpp'),('worst_layer_delta_pp','最小层点Δpp')])+'\n\n'
  if phase_profile:
   txt+=f'P−1损害最大的五层为{phase_profile["Ponly_five_most_negative_layers"]}，Decode为{phase_profile["Donly_five_most_negative_layers"]}，交集{phase_profile["overlap"]}。32个层点Δ曲线的描述性Pearson相关为{phase_profile["curve_Pearson_descriptive"]}；层间不是独立样本，因此不附加相关显著性或把它当机制同一性证明。这些完整阶段曲线比单个最坏层更能回答位置依赖是否一致。\n\n'
  if m:txt+=f'Middle留一最损害L{m["worst_layer"]}（Δ{m["worst_delta"]:+.3f}pp）。Middle用11层达到{middle300:.3f}%、距Native {middle300-native300:+.3f}pp，说明该冻结组合在本批保留了大部分回答能力；逐层留一揭示的是这个组合内部的条件必要性，不能凭此证明11层的唯一性，也不能把小单层影响累加预测联合剪枝。多层之间可能替代、协作并受生成路径改变影响。P−1与Decode的最损害层与全曲线可不同；即使点估计峰值相同，也不足以证明两阶段机制完全相同。\n\n'
  decode_none300=float(next(r for r in base if r['sample_set']=='Full300' and r['condition']=='Decode-None')['accuracy_pct'])
  if m and int(m['n'])==300:
   txt+=f'为什么11层能接近Full：在保持完整Prefill原生时，Middle相对Decode-None恢复{middle300-decode_none300:.3f}pp；留一的最大下降仅{-m["worst_delta"]:.3f}pp，且Middle家族的显著层为{m["Holm_significant_layers"]}。本批未见某个单层被移除就导致明显崩溃的证据；结果与多个层共同维持视觉回看、存在替代或协作相容，但不能据不显著证明每层冗余。\n\n'
  if int(d['n'])==300:
   txt+=f'单层必要性不足以预测联合剪枝：Native背景中，Donly最差单层损失{-d["worst_delta"]:.3f}pp；历史同协议全32层Decode禁图却从{native300:.3f}%降至{decode_none300:.3f}%（{decode_none300-native300:+.3f}pp）。两者是实测的不同干预，不能将单层的小效应简单外推为多层一起删除也安全。\n\n'
  phasefull=[r for r in rows(OUT/'phase_interactions.csv') if r['sample_set']=='Full300']
  if phasefull:
   relay=sorted(phasefull,key=lambda r:float(r['AllText_minus_Both_pp']))[:3]
   desc='；'.join(f'L{r["layer"]}: {float(r["AllText_minus_Both_pp"]):+.3f}pp，普通95%CI [{float(r["relay_context_CI95_low_pp"]):+.3f}, {float(r["relay_context_CI95_high_pp"]):+.3f}]' for r in relay)
   txt+='Prefill文本接收者的条件性线索：All-Text−Both最负的三个点为'+desc+'。这些是事后描述性对比，均是在同层P−1和Decode已禁图的背景下追加关闭R，不能当作Native背景的R-only效应，也未作新的显著性声明。\n\n'
  if 'Addback' in rankby:
   c=rankby['Addback']
   c_native=float(next(r for r in base if r['sample_set']==('Full300' if int(c['n'])==300 else 'Pilot120') and r['condition']=='Native')['accuracy_pct'])
   txt+=f'可选单层充分性（n={c["n"]}）：在完整Prefill原生、Decode仅一层读图时，最佳实测层为{c["best_tied_layers"]}，ACC {c["best_acc"]:.3f}%，相对同队列Decode-None {c["best_delta"]:+.3f}pp，距同队列Native {c["best_acc"]-c_native:+.3f}pp。完整32点及其配对CI、Add-back独立Holm家族见上表；这个最佳点排序不等于独立确认，也不能与Native单层移除的必要性混为一谈。\n\n'
 txt+='所有结论是反复使用TextVQA300上的探索性功能干预证据；单层消融不显著不意味着联合删除安全，Accuracy区间跨0不等于无损。到此停止，不根据曲线追加选层、训练或物理KV实验。\n\n## 复现命令\n\nGPU精确恢复只允许相同冻结源码、协议和输入哈希，并需重新核对授权卡资源；已完成记录不覆盖。最终已完成交付不应重复启动。\n\n```bash\ncd /data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation/vlm_research\n# 原始两卡命令（由launch.py同时启动；不是跨卡切分）\nCUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 python -m textvqa_single_layer_visual_ablation.run --worker 0\nCUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 python -m textvqa_single_layer_visual_ablation.run --worker 1\n# CPU统计和图表\nCUDA_VISIBLE_DEVICES="" python -m textvqa_single_layer_visual_ablation.analysis\nCUDA_VISIBLE_DEVICES="" python -m textvqa_single_layer_visual_ablation.report\n```\n\nPython固定为`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`。启动时设置PYTHONPATH为当前vlm_research目录，CUBLAS_WORKSPACE_CONFIG=:4096:8，OPENBLAS_NUM_THREADS=1，OMP_NUM_THREADS=1。数据、写入掩码、源码、权重与输入哈希详见protocol/source/model/shards和Checkpoint索引。\n'
 # The original launcher starts both resident processes together; foreground
 # serial invocation of the two commands would wait forever at the smoke gate.
 first='CUDA_VISIBLE_DEVICES=GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 python -m textvqa_single_layer_visual_ablation.run --worker 0\nCUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 python -m textvqa_single_layer_visual_ablation.run --worker 1'
 txt=txt.replace(first,'python -m textvqa_single_layer_visual_ablation.launch\n# launch.py同时创建worker0和worker1，各自绑定协议中的唯一GPU UUID。')
 (OUT/'report_zh.md').write_text(txt)
 write(OUT/'figure_manifest.json',dict(figures=[dict(name=n,png_sha256=sha(F/(n+'.png')),pdf_sha256=sha(F/(n+'.pdf'))) for n in figures],measured_points_only=True,ordinary_pointwise_CIs=True,pdf_vector_matplotlib=True))
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
