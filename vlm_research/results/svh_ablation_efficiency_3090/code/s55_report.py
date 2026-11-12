"""Create Chinese factual report only after both experiment validators pass."""
from s55_common import *
import csv,statistics,collections,datetime
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'
def csvrows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def main():
 q=read(O/'ablation/validation.json');v=read(O/'generation_length/validation.json');drift=read(O/'generation_length/calibration.json')
 assert q['status']=='PASS' and v['execution_correctness']=='PASS' and v['formal_requests']==720
 verify_sources();now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 summaries=q['summaries'];bymethod=collections.defaultdict(dict)
 for x in summaries:bymethod[x['condition']][x['benchmark']]=x
 label=dict(Native='Native Full',wo_selective='SVH w/o Selective Transfer',wo_bypass='SVH w/o Attention Bypass',SVH='SVH (Full)')
 out=['# ATE Section 5.5：Component Ablation 与生成长度敏感性实测报告','',f'生成时间：{now}。执行与正确性：PASS；效率整体状态：**{v["status"]}**。',
  '',f'当前GPU：RTX3090，UUID `{UUID}`。模型 `{M}`，Revision `{REV}`。Python3.11.8 / torch2.6.0+cu124 / transformers4.49.0；FP16、Eager、Batch1、Greedy；单GPU串行执行。原算法、旧Manifest、此前18条件结果及论文LaTeX未修改；未Push或创建PR。',
  '', '## 1. Component Ablation：Accuracy','',
  'ScienceQA-IMG 2017道正式图像题；POPE沿用正式8910题版本（Random2910、Popular3000、Adversarial3000），未补齐到9000题。正式vicuna_v1 Prompt、pad-to-square CLIP336 Processor、模型EOS2停止，SQA cap1024、POPE cap128，未强制32输出。',
  '',table(['Configuration','SQA-IMG Accuracy (%)','POPE Overall Accuracy (%)'],[[label[m],f'{bymethod[m]["ScienceQA-IMG"]["accuracy_percent"]:.4f}',f'{bymethod[m]["POPE"]["accuracy_percent"]:.4f}'] for m in ['Native','wo_selective','wo_bypass','SVH']]),
  '', 'Full SVH复用同Checkpoint、Processor、Prompt、全部输入/图片/Gold、生成参数、冻结执行源码与评分协议的10927条历史Physical-C3预测；CPU逐条重评分及独立解析检查通过。当前GPU上两项Benchmark的物理/数值Smoke与历史SVH输出一致。Native此前无完整匹配同协议对照，本次与两个消融新增32781条自然EOS预测。全表43708条逐题预测齐全；不会将历史15题Smoke或公开论文分数当作完整Native。',
  '',table(['Condition','Benchmark','N','Correct','Delta vs Full SVH (pp)','Parse failed','Empty','Truncated'],[[label[x['condition']],x['benchmark'],x['questions'],x['correct'],f'{x["delta_vs_full_SVH_pp"]:+.4f}',x['parse_failures'],x['empty'],x['truncated']] for x in summaries]),
  '', '### POPE子集','',table(['Condition','Subset','N','Accuracy (%)'],[[label[x['condition']],x['strategy'],x['questions'],f'{x["accuracy_percent"]:.4f}'] for x in q['pope_subsets']]),
  '', '### 消融语义与正确性证据','',
  '- **w/o Selective Transfer**：Prefill Relays在全部L0–31读取真实图像K/V，视觉隐藏状态全32层保留，不能在L20提前退出；答案P−1与Decode仍不可直接访问图像，L2–5答案Attention仍物理跳过，持久化KV仍移除视觉Token。额外保留后12层视觉Prefill计算是该消融必然产生的成本。本任务没有为它另做效率测试。',
  '- **w/o Attention Bypass**：L0–19的Transfer、进入L20前真实移除视觉隐藏状态与全部层视觉KV移除不变；L2–5恢复正常Q/K/V/O Attention和真实文本KV；没有保留“空KV元数据”而仅改方法名。',
  '- 两项仅通过独立子类配置组件，decoder forward、Attention算子、模型权重、视觉编码器和Greedy循环复用冻结代码。CPU掩码测试验证目标组件和非目标掩码；GPU检查原始位置/RoPE、逐层QKV调用、隐藏行数、真实KV、图像概率零及同语义短Dense参考数值等价。Dense只用于Smoke参考，正式SVH/消融均为物理实现。Native/No-op输出与全部Smoke logits逐位一致。',
  '', '### Accuracy解释','']
 for bench in ('ScienceQA-IMG','POPE'):
  x=bymethod['wo_selective'][bench];y=bymethod['wo_bypass'][bench]
  out.append(f'{bench}：关闭Selective Transfer相对Full SVH变化{x["delta_vs_full_SVH_pp"]:+.4f}个百分点；关闭Attention Bypass变化{y["delta_vs_full_SVH_pp"]:+.4f}个百分点。这是指定两个Benchmark及冻结协议下的消融结果；不预设两个组件都提高Accuracy。')
 out+=['', '具体而言，Full SVH相对Native在SQA少答对11题，在POPE多答对32题；关闭Selective Transfer相对Full SVH在SQA少6题、POPE仅多1题。关闭Attention Bypass在SQA多8题、POPE少72题。因此，这组消融支持组件对质量存在任务相关的影响，不能写成两个组件均单调提高Accuracy，也不能凭POPE的一题差距宣称优势。SQA的11/12条解析失败均保留并计错；没有进一步逐题因果证据时，不能将全部Accuracy差距归因于输出格式。']
 out+=['','## 2. Generation Length Sensitivity','',
  '仅Native、SVH、SVH-R；原GQA15+TextVQA15的30题ID、图像、Prompt/OCR、Gold及三个输入Tensor哈希完全一致。原32-Token3090旧Session存在15.09%漂移且GPU UUID不同，因此本次重新实测这三个必要条件的32 Token，与16/64/128共同构成当前GPU的配对长度曲线，未混入4090数据。',
  '', '每个Session独立进程/模型加载，正式360请求，两轮共720。输入内交错长度与方法，并以相同确定性轮换计划运行；所有请求真实执行N个Token，1次Prefill和N−1次Decode，忽略EOS。原4090运行器只在独立副本中泛化长度与依赖常量，反向转换AST与原版完全一致；32Token原版/新版实际输出与首logits逐位一致。所有输出长度均完成GPU Smoke与真实KV检查。',
  '', '计时从GPU Tensor就绪且同步后、控制器/位置/掩码初始化前开始，到第N个Token产生、上下文清理和CUDA同步后结束。包含Vision、Projector、Prefill、Greedy、Decode及必要请求开销；排除加载、CPU图片/Processor、H2D、日志与离线KV审计。TTFT为首Token壁钟时间；TPOT=(E2E−TTFT)/(N−1)；CUDA Event Decode TPOT独立记录。KV为第N输出Token后的实际live K/V Tensor字节，此时包含N−1个Decode输入；不把GPU显存当KV。',
  '', '### 两轮均值表','', '以下为两个独立Session均值的平均；“±”是两个Session均值的样本标准差，不是60次独立重复的置信区间。Speedup先在各Session相对同轮Native计算，再取均值。', '']
 means=v['summary'];out.append(table(['Tokens','Method','TTFT ms','TPOT ms/token','E2E ms','Within-session E2E speedup'],[[x['output_tokens'],x['method'],f'{x["TTFT_ms"]:.2f} ± {x["TTFT_ms_session_mean_std"]:.2f}',f'{x["TPOT_ms"]:.3f} ± {x["TPOT_ms_session_mean_std"]:.3f}',f'{x["E2E_ms"]:.2f} ± {x["E2E_ms_session_mean_std"]:.2f}',f'{x["E2E_speedup_mean_within_session"]:.4f}×'] for x in means]))
 raw=csvrows(O/'generation_length/latency_summary.csv')
 out+=['','### Session独立结果','',table(['Session','Tokens','Method','TTFT ms','TPOT ms/token','CUDA Event TPOT','E2E ms','KV MiB','E2E reduction (%)'],[[x['session'],x['output_tokens'],x['method'],f'{float(x["TTFT_ms_mean"]):.3f}',f'{float(x["TPOT_ms_mean"]):.3f}',f'{float(x["CUDA_Event_Decode_TPOT_ms_mean"]):.3f}',f'{float(x["E2E_ms_mean"]):.3f}',f'{float(x["KV_MiB_mean"]):.4f}',f'{float(x["E2E_ms_reduction_percent"]):+.3f}'] for x in raw]),
  '', '### 性能稳定性与重复性','',
  '每轮固定40次Native完整32Token预热，再对全部4长度×3方法各2次完整预热；两固定Native探针各10次建立独立中位数基线。每5题（每60正式请求）进行双探针各3次校准，共6窗口/轮；性能漂移只WARN，不重置基线、不重跑、不丢弃异常请求。5Hz NVML GPU遥测、1Hz CPU遥测及其他计算进程检查均保存。',
  '',f'全部24个“Session×窗口×探针”检查中，{drift["warn_checks"]}个超过2%门限；最大绝对E2E漂移{drift["max_abs_E2E_drift"]*100:.3f}%，最大绝对TPOT漂移{drift["max_abs_TPOT_drift"]*100:.3f}%。状态为{v["status"]}；执行/正确性通过与性能稳定性分别报告。',
  '',table(['Session','Window','Probe','E2E drift (%)','TPOT drift (%)','Flag'],[[x['session'],x['window'],x['key'],f'{x["relative_drift"]["E2E_ms"]*100:+.3f}',f'{x["relative_drift"]["TPOT_ms"]*100:+.3f}','WARN' if x['warn'] else 'within2%'] for x in drift['checks']]),
  '',table(['Tokens','Method','Session1 E2E','Session2 E2E','Cross-session change (%)'],[[x['output_tokens'],x['method'],f'{x["E2E_ms_session1"]:.3f}',f'{x["E2E_ms_session2"]:.3f}',f'{x["E2E_ms_session2_vs_session1_percent"]:+.3f}'] for x in means]),
  '', '### 配对优势与解释','']
 telemetry=read(O/'generation_length/GPU_telemetry_summary.json')
 warning_counts=[]
 for session in (1,2):
  counts=collections.Counter(x['type'] for x in read(O/f'generation_length/session{session}/performance_warnings.json'))
  warning_counts.extend([[session,k,n] for k,n in sorted(counts.items())])
 out[-2:-2]=['### 遥测与警告类别','',table(['Session','GPU samples','SM clock min/max MHz','Temperature min/max C','Mean power W'],[[x['session'],x['samples'],f'{x["SM_MHz"]["min"]}/{x["SM_MHz"]["max"]}',f'{x["temperature"]["min"]}/{x["temperature"]["max"]}',f'{x["power_mW"]["mean"]/1000:.2f}'] for x in telemetry]),'',table(['Session','Warning category','Count'],warning_counts) if warning_counts else '没有记录到上述性能警告。','', '以上遥测范围覆盖Smoke、预热、基线及正式运行，不能直接解释为正式请求间的时钟波动；逐请求对应状态在per_request.csv中。请求内早/晚Decode差异超过5%只作诊断标记，可能还包含序列长度增长、首步和运行时开销差异，不能单凭该标记认定GPU降频。CPU调度、P-State与Throttle原始字段保留，未做缺乏证据的根因断言。','']
 paired=csvrows(O/'generation_length/paired_statistics.csv')
 out.append(table(['Session','Tokens','Method','Decode faster /30','E2E faster /30','E2E paired Native−method mean ms','Exploratory95% CI ms'],[[x['session'],x['length'],x['method'],next(z['method_better_inputs'] for z in paired if z['session']==x['session'] and z['length']==x['length'] and z['method']==x['method'] and z['metric']=='TPOT_ms'),x['method_better_inputs'],f'{float(x["native_minus_method_mean"]):.3f}',f'[{float(x["exploratory_CI95_low"]):.3f}, {float(x["exploratory_CI95_high"]):.3f}]'] for x in paired if x['metric']=='E2E_ms' and x['method']!='Native']))
 out+=['', '配对统计按同题、同长度、同Session计算；所有方法的逐题延迟和变化完整保留。区间为30个配对输入的探索性t区间，存在系统状态/请求相关性，未做多重检验校正；尤其约1–2%的差异不能直接宣称显著加速。只有两个Session，不能据此概括所有3090实例。']
 for x in means:
  if x['method']!='Native':out.append(f'{x["output_tokens"]}Token {x["method"]}：同轮Native的平均E2E降低{x["E2E_reduction_percent_mean_within_session"]:+.3f}%，Speedup {x["E2E_speedup_mean_within_session"]:.4f}×。是否支持长度下的端到端收益以该实测值和两轮配对区间为准，未由32Token外推。')
 for method in ('SVH','SVH-R'):
  effect=[x for x in raw if x['method']==method]
  intervals=[x for x in paired if x['method']==method and x['metric']=='E2E_ms']
  decodes=[x for x in paired if x['method']==method and x['metric']=='TPOT_ms']
  out.append(f'{method}的8个“长度×Session”条件中，{sum(float(x["E2E_ms_reduction_percent"])>0 for x in effect)}/8的E2E点估计低于同轮Native；{sum(float(x["exploratory_CI95_low"])>0 for x in intervals)}/8的探索性配对区间下界大于0。Decode逐题占优数量的范围为{min(int(x["method_better_inputs"]) for x in decodes)}–{max(int(x["method_better_inputs"]) for x in decodes)}/30。上述是当前实例的观察，性能WARN与仅两个Session的限制仍然成立，不能据此声称所有硬件状态均稳定复现。')
 out+=['', '### 对Section 5.5的支持范围','',
  '本次在16、32、64、128 Token均观察到SVH和SVH-R相对同Session Native的端到端收益，Decode TPOT也较低。因此，这组实测支持“固定输出长度改变后仍有推理收益”。但E2E降低比例随长度增加而减小：短输出中的TTFT收益占比更高，Decode节省并未使相对E2E收益随长度单调增大，不能写成“输出越长，加速比例越大”。SVH和SVH-R的平均TPOT非常接近，二者的小差异不构成可靠速度排名证据。',
  f'Native四种长度的Session2相对Session1 E2E增幅为{min(x["E2E_ms_session2_vs_session1_percent"] for x in means if x["method"]=="Native"):.2f}%–{max(x["E2E_ms_session2_vs_session1_percent"] for x in means if x["method"]=="Native"):.2f}%。因此绝对延迟的跨轮漂移不可忽略；同轮配对收益与跨轮绝对稳定性是两个不同结论。全部结果按性能WARN交付，不可标为稳定PASS。',
  '质量方面，Full SVH在SQA略低于Native、在POPE略高于Native；两个消融的变化方向也不同。这组结果反映质量与效率的折中，不支持“所有任务质量均不降”或“两个组件均提高Accuracy”。']
 out+=['','## 3. 结果、图与复现','',
  '- 质量：`ablation/summary.csv`、`pope_subsets.csv`、`sqa_results.csv`、`pope_results.csv`、`predictions.jsonl`、`checkpoint.sqlite`、`validation.json`。',
  '- 效率：`generation_length/per_request.csv`（720请求，带时间对齐GPU状态）、`latency_summary.csv`（逐Session完整均值/中位数/标准差）、`latency_two_sessions.csv`、`paired_statistics.csv`、`calibration.json`、两个Session各自SQLite、校准/预热/Smoke/5HzGPU与CPU日志。',
  '- 图：`generation_length/latency_vs_tokens.pdf`与同名PNG；可编辑绘图源码`code/s55_plot.py`。误差线显示两个Session均值的范围，非置信区间；完整数值来自latency_two_sessions.csv。图必须在真实数据和CPU验证完成后绘制。',
  '- 可复现源码：`code/`、`protocol.json`、`source_manifest.json`（测量前冻结）、`artifact_manifest.json`（最终文件SHA）、`generation_length/runtime_length_only.diff`及AST审计。',
  '', '在资源已恢复且同名结果目录不存在的新实例中，将此任务code目录放到同一项目位置，核对GPU UUID和模型回执并在新独立入口中更新实例身份后运行以下命令；已完成结果禁止直接覆盖。源程序对已有协议/Checkpoint拒绝重跑，不会继续旧Session或挑选最快请求。',
  '', '```bash',f'source {REST}/activate.sh',f'cd {O}',f'{P} -u -B code/s55_prepare.py',f'{P} -u -B code/s55_ablation_run.py',f'{P} -u -B code/s55_analyze.py --quality-only',f'{P} -u -B code/s55_length_run.py --session 1',f'{P} -u -B code/s55_length_run.py --session 2',f'{P} -u -B code/s55_analyze.py --length-only',f'{P} -u -B code/s55_report.py','```',
  '', '冻结GPU UUID与恢复回执是有意保留的实例防错门禁；跨实例复现必须在新目录留下映射修订证据，不能谎称新UUID与旧GPU相同。模型、数据和Python环境在本地ext4盘，work只存配置、代码和实验输出。',
  '', '本报告不改论文Table/Figure，不保证结果符合预期。图和结论只使用本次真实测量；历史FullSVH质量复用来源明确，旧4090效率/旧3090漂移结果没有混入新正式统计。']
 (O/'report.md').write_text('\n'.join(out)+'\n')
 save(O/'validation.json',dict(status=v['status'],quality=q['status'],efficiency_execution=v['execution_correctness'],quality_records=43708,efficiency_formal_requests=720,source_integrity='PASS',paper_unchanged=True,no_push=True,report_created_ns=time.time_ns()))
 print('CHINESE_REPORT_CREATED',O/'report.md',flush=True)
if __name__=='__main__':main()
