from pathlib import Path
import json,csv,collections,statistics,math,subprocess,hashlib
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def read(p):return json.loads(Path(p).read_text())
def csvrows(name):
 p=A/name
 return list(csv.DictReader(p.open())) if p.exists() and p.stat().st_size else []
def mdtable(headers,rows):
 return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in r)+'|' for r in rows])
def f(x):return '—' if x is None or x=='' else f'{float(x):.3f}'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summ=csvrows('optimized_efficiency_32tokens.csv');pairs=csvrows('paired_comparisons.csv');nums=csvrows('numerical_summary.csv')
selected={k:read(A/k/'selected_stages.json') for k in ['Qwen','InternVL','LLaVA'] if (A/k/'selected_stages.json').exists()}
completion=read(A/'coordinator_status.json')
formal=sum(int(r['requests']) for r in summ)
# Always label preliminary output until final coordinator completion and independent validation.
finished=completion['status']=='COMPLETE_ALL_PHASES'
title='跨模型物理优化与效率复测：'+('完成验证' if finished else '阶段性记录，尚未完成')
out=['# '+title,'','本轮只测真实发生变化的路径、对应冻结实现及同 Backbone Native；没有重跑全量质量 Benchmark、训练、层搜索、其他外部基线或 GitHub 推送。普通 LLaVA SVH/SVH-R 源码与执行路径未改，未为统一表格新增其 GPU 请求。','',
 f'已记录正式效率请求：{formal}。正式协议为原 GQA15 + TextVQA Paper-OCR15，32输出Token、1 Prefill +31真实Decode、Batch1、原精度/Eager、两个独立Session。全部请求固定 GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6。','']
# Concise interpretation of both sessions, without pooling separate campaigns.
brief=[]
for kind,method,stage in [('Qwen','SVH','E'),('Qwen','SVH-R','C'),('InternVL','SVH','B'),('LLaVA','FastV+SVH','B'),('LLaVA','SparseVLM+SVH','B'),('LLaVA','V2Drop+SVH','B'),('LLaVA','ApET+SVH','B')]:
 sub=sorted([r for r in pairs if r['kind']==kind and r['method']==method and r['stage']==stage and r['reference']=='same_method_frozen'],key=lambda r:r['session'])
 if len(sub)==2:
  brief.append([kind,method,stage]+[' / '.join(f(float(r[m+'_improvement_pct_mean'])) for r in sub) for m in ['TTFT_ms','TPOT_ms','E2E_ms']])
out+=['## 主要结论：两轮配对改善','',mdtable(['Backbone','Method','版本','TTFT改善% S1/S2','TPOT改善% S1/S2','E2E改善% S1/S2'],brief),'',
'Qwen SVH-R 与 LLaVA ApET+SVH 显示稳定的主要 E2E 改善；Qwen SVH 最后E版也在独立两轮中重复获得收益。FastV/V²Drop融合的E2E改善不足1%，不据此宣称显著整体加速。SparseVLM融合改善较小。',
'InternVL SVH B的正式E2E改善约0.572%和−0.002%，没有稳定的整体收益，本轮不建议据此替换默认原版。其C2/E虽然数值通过，但筛选E2E退化约0.533%，拒绝采用。InternVL SVH-R的B/C未显示有用收益，未强制进行正式复测。',
'ApET+SVH第二轮TTFT退化约3.598%，尽管Decode与E2E改善明显，仍完整报告该局部负结果。',
'上表每个S1/S2属于该版本自己的同题配对Session；E补充与早期B/C是不同campaign，不把它们的Native绝对均值混合。所有版本和慢样本均保留。','']
rows=[]
for r in summ:
 rows.append([r['kind'],r['session'],r['method'],r['stage'],r['requests'],f(r['KV_Live_MiB_mean']),f(r['TTFT_ms_mean']),f(r['TPOT_ms_mean']),f(r['E2E_ms_mean']),f(r['E2E_ms_median'])])
out+=['## 正式效率（每行30题算术均值）','',mdtable(['Backbone','Session','Method','阶段','n','KV Live MiB','TTFT ms','TPOT ms/token','E2E@32 ms','E2E中位数'],rows),'','中位数、Peak Allocated/Reserved、真实逐层Live/Storage/Capacity及全部逐题数据见CSV，不以峰值总显存替代KV。','']
rows=[]
for r in pairs:
 if r['reference']!='same_method_frozen':continue
 rows.append([r['kind'],r['session'],r['method'],r['stage'],r['pairs'],f(r['TTFT_ms_improvement_pct_mean']),f(r['TPOT_ms_improvement_pct_mean']),f(r['E2E_ms_improvement_pct_mean']),f(r['KV_Live_MiB_improvement_pct_mean'])])
out+=['## 同Session、同输入配对改善','',mdtable(['Backbone','Session','Method','优化阶段','配对n','TTFT改善%','TPOT改善%','E2E改善%','KV改善%'],rows),'','改善百分比为逐题(1−优化/冻结原版)的算术平均，不能与均值比值混淆。正值代表改善；负值与慢样本不删除。两Session均保留，不选最快一轮。相对本轮Native的对应数据在paired_comparisons.csv。','']
out+=['## 共同问题与模型差异','',
'跨模型重复的是控制路径与存储管理成本：重复索引/Mask、调试标量回传、位置Gather，以及InternVL的逐Token KV cat。融合路径还有排名/聚类账本与调试同步；同名API的成本和语义不能直接等同。',
'Qwen特有问题包括mRoPE/DeepStack约束、原数值修复链的固定Lt descriptor与补零、SVH-R晚层辅助视觉KV构造。B/C没有改变必要构造、模型权重、访问层集合或原始位置。Qwen的热路径isin与反复布局断言可以缓存/移到审计；必要视觉构造仍计时。',
'LLaVA SVH/SVH-R原本已使用Compact Attention、请求级Mask与预分配KV，无证据支持改写。四个融合版本保留官方排名、Token Recycling、FPS/基底替换及冻结剪枝预算；没有把融合配置冒充独立论文工作点。','']
rows=[]
for kind,ms in selected.items():
 for method,entry in ms.items():
  rows.append([kind,method,entry['stage'] or 'NOT_ADOPTED',entry['status'],json.dumps(entry['decisions'],ensure_ascii=False)])
for kind in ['Qwen','InternVL']:
 p=A/kind/'refine_selected_stages.json'
 if p.exists():
  for method,entry in read(p).items():rows.append([kind+' / E补充',method,entry['stage'] or 'NOT_ADOPTED',entry['status'],json.dumps(entry['decisions'],ensure_ascii=False)])
out+=['## 阶段选择与负结果','',mdtable(['Backbone','Method','E实际指向','筛选状态','全部筛选决策'],rows),'',
'阶段A/B/C的普通筛选耗时见ablation_efficiency.csv，三种长度输入、各3次并交错执行。E为冻结筛选规则下的阶段别名，不是重新调参数；C相对B的小差值不能单独当作稳定收益。',
'没有收益或回退的步骤保留，不强制采用。未通过数值门槛的D不进入正式计时，不报告其仪器扰动耗时为有效性能。','']
d=[r for r in nums if r['stage']=='D']
if d:
 out+=['D的数值失败/候选记录：','',mdtable(['Backbone','阶段','样本','状态','Max|logit差|','Logits相对L2','Hidden相对L2','参考上下文Top-1序列一致'],[[r['kind'],r['phase'],r['key'],r['status'],f(r['max_abs']),f(r['logits_relative_L2_max']),f(r['hidden_relative_L2_max']),r['tokens_exact']] for r in d]),'']
for kind in ['Qwen','InternVL','LLaVA']:
 for campaign in ['screen','refine_screen']:
  p=A/kind/campaign/'candidate_eligibility.json'
  if p.exists():
   fail=read(p).get('failures',{})
   if fail:out+=['候选拒绝 '+kind+' / '+campaign+': '+json.dumps({k:{a:b for a,b in v.items() if a!='traceback'} for k,v in fail.items()},ensure_ascii=False),'']
out+=['E补充版本：Qwen SVH缓存同一步dead-column索引/Mask，原补零保留；InternVL SVH C2修复被有效长度检查拒绝的原C实现，使用独立包。原失败源码与日志保留。E在GPU前另冻结门槛，只有相对B确认额外收益才做正式复测。E与原B/C分开列出，不将不同轮次均值直接作配对比较。','']
out+=['## 数值、质量与论文使用限制','',
'B/C只涉及索引、Mask复用或物理缓存管理，按GPU前冻结要求检查原30题全部32步有效Logits及Token严格一致；原始A还对齐历史冻结Token记录。Native/No-op和真实Audit Mode重新启用检查。逐层数值与缓存生命周期见numerical_*、kv_*及各阶段原始记录。',
'本轮不是全量Accuracy复评。对B/C，已匹配输入上的完整输出没有回归；不能由30题宣称所有任务绝对无质量回退。未修改Attention算术且数值回归通过的版本可作为对应冻结算法的独立工程实现效率记录；仍须保留原质量协议与配置来源。此前融合质量来源未验证的部分不会因效率优化自动获得质量可比性。',
'回归参照是冻结的Physical原版；本轮不撤销、不掩盖历史Physical Numerical Waiver。D诊断采用参考序列teacher forcing，记录的是参考上下文的逐步Top-1变化；数值失败后不额外声称已完成自由生成质量测试。',
'D修改Attention数值执行路径，数值失败版本不能继承历史Accuracy。即使某候选小样本答案不变，也仅能证明该回归集合；没有独立质量证据时，不将其效率与历史全量Accuracy拼成同一论文工作点。',
'这些计时仅适用于本3090、原30输入及固定32输出协议，不替代正常EOS质量实验、不代表不同分辨率/长度的泛化速度。','']
warnings=read(A/'performance_warnings.json')['warnings'] if (A/'performance_warnings.json').exists() else []
drift=[x for x in warnings if x['type']=='NATIVE_DRIFT_GT5PCT']
out+=['## 漂移与计时审计','',f'已发现Native探针>5%告警 {len(drift)} 条；全部保留并列于performance_warnings.json。计时不删除慢请求、不重设基线、不修改GPU功耗/频率或系统配置。','',
'Qwen调度顺序差异已记录：全部30输入精确回归先完成，实际Audit Mode在Session1之后、Session2之前运行。Session1只有在Audit Mode PASS后纳入最终验证，未为选择更快成绩重跑。',
'计时沿用冻结边界，包含方法专用初始化、Vision、注入、Prefill、31步Decode与Context恢复；H2D/预处理/写盘和最终缓存元数据检查/对象释放不在边界内，所有条件一致。Profiler属于额外审计请求，不能将其墙钟耗时纳入正式数据。Kernel数和同步数为固定中等输入的诊断，不能简单相加推断墙钟收益。','']
prof=csvrows('profile_stage_counts.csv')
if prof:
 out+=['## 固定中等输入Profiler实测','',mdtable(['Backbone','Campaign','Method','阶段','Kernel数','CUDA同步API数','isin','cat','index_select','copy','CPU local_scalar'],[[r['kind'],r['campaign'],r['method'],r['stage'],r['kernel_count'],r['cuda_sync_api_count'],r['isin_count'],r['cat_count'],r['index_select_count'],r['copy_count'],r['local_scalar_dense_count']] for r in prof]),'']
out+=['## 文件与资源状态','',
'服务器持久化交付目录：/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization/',
'本地SSD完整工作区：/home/featurize/ate-experiments/svh-cross-opt-20261113/',
'源码快照、实际修改diff、模型Revision回执、独立适配包、命令、SQLite检查点、逐题Token与输入SHA、数值诊断、监控和完整Profiler均保留。SHA256SUMS与evidence_manifest.json说明文件完整性。无模型权重或图片复制到结果目录；没有推送GitHub。','',
'最终释放状态以resource_release.json为准。未完成独立验证及持久盘SHA复核前，不标记最终PASS。']
(A/'final_report_zh.md').write_text('\n'.join(out)+'\n')
# Numerical report, including first divergence evidence.
by=collections.defaultdict(list)
for r in nums:
 if r['phase'] in ['screen','refine_screen']:by[r['kind']+' / '+r['phase'],r['method'],r['stage']].append(r)
n=['# 数值与输出回归','', '门槛在正式GPU前冻结于protocol.json。B/C要求有效Logits逐位一致，D采用固定BF16容差；没有放宽失败门槛。','',
mdtable(['Backbone','Method','阶段','样本数','PASS数','精确Logits数','精确Token数','最大Logits差'],[[*k,len(rs),sum(r['status']=='PASS' for r in rs),sum(r['logits_torch_equal']=='True' for r in rs),sum(r['tokens_exact']=='True' for r in rs),f(max(float(r['max_abs']) for r in rs))] for k,rs in sorted(by.items())]),'']
for kind in ['Qwen','InternVL','LLaVA']:
 p=A/kind/'audit/audit_mode_validation.json'
 if p.exists():n+=['Audit '+kind+': '+json.dumps(read(p),ensure_ascii=False),'']
 p=A/kind/'diagnosis/numerical_validation.json'
 if p.exists():
  for rec in read(p):
   diffs=[x for x in rec.get('layerwise_differences',[]) if x['relative_L2']>0]
   if diffs:n+=['首个记录分歧 '+kind+': '+json.dumps(diffs[0],ensure_ascii=False),'']
n+=['完整32步Token和早期/中间/末步Hidden比较，及真实缓存指针/容量，均为独立请求审计，不混入正式耗时。B/C匹配30题输出回归不等于重跑全量质量Benchmark。D若失败，正式效率与质量绑定均NOT_RUN。']
(A/'numerical_equivalence_report.md').write_text('\n'.join(n)+'\n')
print(json.dumps(dict(report_status='FINAL_CANDIDATE' if finished else 'PROVISIONAL',formal=formal,summary_rows=len(summ),drift_warnings=len(drift))))
