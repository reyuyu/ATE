import pathlib,sys,csv,json,datetime,collections
root=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget/vlm_research');sys.path.insert(0,str(root))
from textvqa_reaccess.common import *
def main():
 summary=list(csv.DictReader((OUT/'accuracy_summary.csv').open()));pairs=list(csv.DictReader((OUT/'paired_comparisons.csv').open()));curve=list(csv.DictReader((OUT/'budget_curve.csv').open()));targets=list(csv.DictReader((OUT/'descriptive_targets.csv').open()));cases=list(csv.DictReader((OUT/'diagnostic_cases.csv').open()))
 first={r['condition']:r for r in summary if r['scope']=='300'};second={r['condition']:r for r in summary if r['scope']=='120'}
 def comparison(family,a,b):
  r=next((r for r in pairs if r['family']==family and r['left']==a and r['right']==b),None)
  if r:return f"{float(r['delta_pp']):+.3f}pp [{float(r['CI95_low_pp']):+.3f}, {float(r['CI95_high_pp']):+.3f}]，Holm p={float(r['Holm_p_within_family']):.5f}"
  r=next(r for r in pairs if r['family']==family and r['left']==b and r['right']==a)
  return f"{-float(r['delta_pp']):+.3f}pp [{-float(r['CI95_high_pp']):+.3f}, {-float(r['CI95_low_pp']):+.3f}]，Holm p={float(r['Holm_p_within_family']):.5f}"
 lines=['','## 对十个研究问题的直接回答','',f"1. 仅P−1全32层（F）为{float(first['F']['accuracy_pct']):.3f}%，相对B：{comparison('stage1','F','B')}。",'',f"2. 仅Decode全32层（G）为{float(first['G']['accuracy_pct']):.3f}%，相对B：{comparison('stage1','G','B')}。",'', '3. Both实际预算曲线（k: Accuracy%）为'+', '.join(f"{r['k']}: {float(r['accuracy_pct']):.3f}" for r in curve)+'。','']
 lines+=['4–5. 以下均为观察到的离散预算点标准，不插值、不证明非劣效。']
 for r in targets:lines+=['',f"{r['target']}：最小观察k={r['minimum_observed_k']}；稀疏k<32达到：{r['sparse_point_target_achieved']}，最小稀疏k={r['minimum_observed_sparse_k'] or '未达到'}。"]
 lines+=['',f"6. 全32层P−1-only相对Decode-only（F−G）：{comparison('stage1','F','G')}。下表进一步在同一120题、同一S_k上比较阶段，不从各自点估计直接作全局重要性排名。",'', '| k | Ponly−Donly pp [普通95%CI] / Holm p |','|---:|---|']
 for k in [2,4,8,16,32]:
  a='C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k);b='D' if k==2 else 'G' if k==32 else 'Donly-'+str(k);lines.append(f"| {k} | {comparison('stage2',a,b)} |")
 lines+=['', '7. 随机层对照在同一120题比较，固定嵌套集合相对三种子逐题均值：']
 for k in [8,16]:
  rr=[float(second[f'Random-{k}-s{j+1}']['accuracy_pct']) for j in range(3)];lines+=['',f"k={k}：{comparison('stage2','Both-'+str(k),'RandomMean-'+str(k))}；随机种子Accuracy范围{min(rr):.3f}–{max(rr):.3f}%。该结果只覆盖三个冻结随机集合，不等于最优层选择机制。"]
 declines=[(a['k'],b['k'],float(b['accuracy_pct'])-float(a['accuracy_pct'])) for a,b in zip(curve,curve[1:]) if float(b['accuracy_pct'])<float(a['accuracy_pct'])-1e-12];harm=[r for r in cases if r['category']=='higher_budget_harms'];unique=len({r['question_id'] for r in harm})
 lines+=['', '8. 总体预算点出现下降：'+('；'.join(f'{a}→{b}: {d:+.3f}pp' for a,b,d in declines) if declines else '本轮八个固定嵌套预算的总体点估计没有下降')+f'。但更高预算造成逐题Score下降的相邻预算记录有{len(harm)}条、涉及{unique}道不同问题，不能把总体单调误读为逐题安全。','',f"9. 后续Decode确实可以改变能力：G−B为{comparison('stage1','G','B')}。G的字面首Token原始Logits与B逐题严格相同；后续轨迹和完整答案差异见诊断CSV。首个Tokenizer位置可能只是空白，不能把“后续Decode”与“第二个语义词以后”混淆。",'', '10. 本轮只建立在固定300题上的质量—功能性回看权限关系。是否少量层足够，应同时看恢复比例、相对Native Full的绝对差距和不确定区间；未完成新的独立数据验证。随机结果只在120题成立，不能直接替代300题预算曲线；不能从本批结果自动挑层追加生成。','']
 # Explicitly retain all new abnormal outputs by scope and original condition.
 lines+=['## 新增条件的异常输出','', '| 范围 | 条件 | 截断 | 重复3gram | 空答案 | 相对B完整Token序列变化 |','|---|---|---:|---:|---:|---:|']
 for r in summary:
  if r['condition'] in NEW and ((r['scope']=='300' and DESIGN['conditions'][r['condition']]['scope']==300) or (r['scope']=='120' and DESIGN['conditions'][r['condition']]['scope']==120)):lines.append(f"| {r['scope']} | {r['condition']} | {r['truncated']} | {r['repeated']} | {r['empty']} | {r['token_ids_changed_vs_B']} |")
 runs=[read(OUT/'workers'/str(j)/'gpu_run_complete.json') for j in [0,1]];start=min(datetime.datetime.fromisoformat(r['UTC']) for r in read(OUT/'launch_receipt.json'));end=max(datetime.datetime.fromisoformat(r['UTC']) for r in runs);lines+=['', '## 运行与释放','',f"双卡GPU阶段墙钟时间约{(end-start).total_seconds()/60:.2f}分钟（包含加载、Smoke、阶段等待与审计开销；不是纯Kernel计时）。每卡一份模型，各自1770个正式新增检查点；新矩阵共3540个，历史复用1500个，规范逐题表共5040条。Smoke中兼容且属于本卡分片和正式范围的生成审计后复用，不重复计算。"]
 for r in runs:lines+=['',f"Worker{r['worker']}：GPU UUID `{r['GPU_UUID']}`；峰值allocated {r['peak_allocated_bytes']/1024**3:.2f}GiB，reserved {r['peak_reserved_bytes']/1024**3:.2f}GiB。"]
 lines+=['', '释放状态和当前显存/进程快照以gpu_release.json为准；资源快照不构成未来可用性承诺。没有终止其他任务，也没有加载第三张卡。完成交付后停止。']
 p=OUT/'report_zh.md';p.write_text(p.read_text()+'\n'.join(lines).rstrip()+'\n')
 print('INTERPRETATION_COMPLETE',len(harm),unique)
if __name__=='__main__':main()
