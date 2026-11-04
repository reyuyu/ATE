"""Append concrete external-validation interpretation without changing frozen rules."""
import pathlib,sys,csv,collections,numpy as np,datetime,json
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation');sys.path.insert(0,str(W/'vlm_research'))
from receiver_cross_dataset.common import *
lines=['','## 最终判断：跨数据集迁移、阈值与工程意义',''];checks=[]
for dataset in ['vqa_v2','textvqa']:
 d=OUT/dataset;summary=list(csv.DictReader((d/'accuracy_summary.csv').open()));pairs=list(csv.DictReader((d/'paired_comparisons.csv').open()));tr=list(csv.DictReader((d/'trigger_analysis.csv').open()));n=len(tr);length=np.array([int(r['Q_tokens']) for r in tr]);cnt=collections.Counter(r['first_word_group'] for r in tr)
 qword=sum(r['what_which']=='True' for r in tr);long15=sum(r['length_ge15']=='True' for r in tr);long18=sum(r['length_ge18']=='True' for r in tr)
 only15=sum(r['what_which']=='False' and r['length_ge15']=='True' for r in tr);only18=sum(r['what_which']=='False' and r['length_ge18']=='True' for r in tr)
 disagree=sum(r['trigger15']!=r['trigger18'] for r in tr)
 lines += [f"**{dataset}结构迁移。** what/which占{qword}/{n}（{100*qword/n:.2f}%）。Question Token长度均值{length.mean():.2f}、中位数{np.median(length):.1f}、四分位数{np.quantile(length,.25):.1f}–{np.quantile(length,.75):.1f}、范围{length.min()}–{length.max()}。仅由长度触发、而非what/which的题数为T15 {only15}、T18 {only18}；两阈值接收集合只在{disagree}题不同。",'']
 for t in [15,18]:
  q=next(r for r in pairs if r['left']==f'Threshold-{t}' and r['right']=='Question-Only');rnd=next(r for r in pairs if r['left']==f'Threshold-{t}' and r['right']=='Budget-Matched Random');allr=next(r for r in pairs if r['left']==f'Threshold-{t}' and r['right']=='All-Writer')
  positive=lambda r:float(r['delta_pp'])>0 and float(r['Holm_p_6within_dataset'])<.05
  checks.append(dict(dataset=dataset,threshold=t,Q_gain_Holm=positive(q),Random_gain_Holm=positive(rnd),delta_Q_pp=float(q['delta_pp']),delta_Random_pp=float(rnd['delta_pp']),delta_All_pp=float(allr['delta_pp'])))
  label='同时优于Question-Only与随机触发，且均通过预声明六比较Holm校正' if positive(q) and positive(rnd) else '尚未同时取得超越Question-Only与随机触发的校正后证据'
  lines += [f"Threshold-{t}：{label}。相对Q为{float(q['delta_pp']):+.3f}pp，相对随机为{float(rnd['delta_pp']):+.3f}pp，相对All为{float(allr['delta_pp']):+.3f}pp。T18与随机的预算不保证相同，因为随机只匹配T15。",'']
 if dataset=='textvqa':lines += ['TextVQA原标注没有VQAv2的官方yes/no、number、other字段；表中统一标为textvqa，另按真实首词和触发类型分组，未臆造官方答案类型。','']
if all(x['Q_gain_Holm'] and x['Random_gain_Holm'] for x in checks):
 conclusion='这两个预先冻结的Question-Conditional Assistant规则在两项外部样本上均取得了超越Question-Only及随机触发的证据，支持分配策略的跨数据集迁移；不能简单解释为只对GQA有效。'
else:
 conclusion='外部迁移证据需要区分数据集和控制条件：并非所有数据集/阈值都同时通过相对Question-Only及随机的校正后检验，因而不能把两条规则概括为已验证的普适语义机制。具体正负结果均完整保留。'
lines += [conclusion,'','**15是否最优或过拟合？** 本轮没有搜索阈值。两条阈值在新数据上仅有少数接收集合差异；即便Accuracy接近或相同，也不能证明15是唯一最优阈值，更不能用本轮数据重新选择阈值。表中所有CIs为点位探索性区间；没有检验非劣效。','', '**模板与因果边界。** 干预确认的是开放既有接收位置会改变模型输出，以及固定策略在这些问题上的配对效果。原生ASSISTANT四Subword和其余Prompt完全保留；没有改变模板、换模型或操纵问题首词，故不能证明what/which本身导致必要性，也不能宣称跨模板、跨模型成立。随机对照只是一组预先固定种子，且精确长度分层限制了可置换范围。','', '**Physical Visual Handoff。** 若后续算法需要功能性Receiver规则，这两条冻结规则可按本报告实际外部证据保留为候选；本轮结果不足以锁定最优策略，更没有证明密集Attention已减少FLOPs、显存或延迟。全部原生QKV、MLP、Residual和完整KV更新均执行。工程收益及是否保留视觉状态必须另行验证，本轮到此停止。','']
run=read(OUT/'gpu_run_complete.json');start=read(OUT/'resource_start.json');elapsed=(datetime.datetime.fromisoformat(run['UTC'])-datetime.datetime.fromisoformat(start['UTC'])).total_seconds()
unique=sum(read(OUT/d/'completion.json')['unique_generations'] for d in ['vqa_v2','textvqa']);lines += [f'**执行规模和时间。** 500+300题共4000条逻辑条件记录，精确去重后正式生成{unique}条，未复用任何GQA答案；Smoke另计。GPU工作进程从启动资源记录至正式验证结束约{elapsed/60:.2f}分钟（含加载、Smoke、两阶段正式生成和核验），数据下载/CPU准备另计。GPU已释放，详见gpu_release.json。','']
report=OUT/'report_zh.md';text=report.read_text();marker='\n## 最终判断：跨数据集迁移、阈值与工程意义'
if marker in text:text=text.split(marker)[0]
report.write_text(text+'\n'.join(lines)+'\n')
write(OUT/'cross_dataset/final_interpretation.json',dict(UTC=utc(),checks=checks,conclusion=conclusion,GPU_worker_minutes=elapsed/60,formal_unique_generations=unique))
print(json.dumps(checks),flush=True)
