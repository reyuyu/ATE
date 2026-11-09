"""Report only actual new-GPU results; unfinished formal fields stay blank."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,subprocess
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
W=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(name):
 p=O/name
 if not p.exists():return []
 with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def f(v,n=3):return '—' if v is None or v=='' else f'{float(v):.{n}f}'
def md(header,rows):return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows]
def main():
 progress=read(O/'progress.json');validation=read(O/'validation.json');protocol=read(O/'protocol.json');hist=table('historical_accuracy_summary.csv');hcomp=table('historical_paired_comparisons.csv');summary=table('accuracy_summary.csv');comp=table('paired_comparisons.csv');eff=table('efficiency_summary.csv');ecomp=table('efficiency_paired_comparisons.csv')
 seven=read(O/'seven_completion.json') if (O/'seven_completion.json').exists() else {};efin=(O/'efficiency_completion.json').exists();hfin=(O/'historical_completion.json').exists();aud=read(O/'independent_scoring_audit.json') if (O/'independent_scoring_audit.json').exists() else {};post=(O/'postprocessing_manifest.json').exists()
 done=hfin and seven.get('status')=='COMPLETE' and efin and aud.get('status')=='PASS';status='COMPLETE_EXPLORATORY_NUMERICAL_EQUIVALENCE_FAILED' if done else 'PARTIAL'
 hdelta=next((r for r in hcomp if r['condition']=='Physical-Original' and r['reference']=='Native'),None);speed=next((r for r in ecomp if r['regime']=='fixed32' and r['condition']=='Physical-Original'),None)
 official_harms=[r for r in comp if r['benchmark'] not in ['MME','OCRBench-v1'] and 'ATE_PAPER_PROTOCOL'!=r['protocol_label']];official_harms.sort(key=lambda r:float(r['delta']))
 lines=['# Qwen3-VL 原始 Physical C3：新 RTX 3090 评测','',f'状态：**{status}**。更新时间 UTC：{datetime.now(timezone.utc).isoformat()}。','']
 opening=[]
 if hdelta:opening.append(f"本轮历史 GQA1000 中，Physical-Original − Native 为 {f(hdelta['delta'])} pp（图片簇配对95%CI [{f(hdelta['CI95_low'])}, {f(hdelta['CI95_high'])}]）。这是历史1000题复现，不是完整官方GQA成绩。")
 else:opening.append('历史GQA1000尚未完成，不填写正式复现准确率差值。')
 if speed:opening.append(f"固定32 Token下，真实KV Live节省 {f(speed['KV_live_saving_percent'])}%，Capacity节省 {f(speed['KV_capacity_saving_percent'])}%；新卡本轮Native/C3的E2E@32加速为 {f(speed['speedup'])}×（95%CI [{f(speed['CI95_low'])}, {f(speed['CI95_high'])}]）。")
 else:opening.append('固定32 Token的48题效率阶段尚未完成，KV节省和E2E@32加速待测，不使用旧卡1.174×填补。')
 if official_harms:opening.append('已完成可比质量集合中差值最低的是 '+', '.join(f"{r['benchmark']} {f(r['delta'])} pp" for r in official_harms[:3])+'；不同单位的MME/OCRBench另表列示。')
 else:opening.append('七项正式评测尚未全部完成，最受损能力待完整数据审计。')
 lines+=opening+['','## 模型、输入与执行版本','',f"实际GPU：NVIDIA GeForce RTX3090 24GB；UUID `{protocol['gpu_UUID']}`。模型 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，16个资产与固定Revision及历史权重SHA完全一致。BF16/Eager、Transformers4.57.6、PyTorch2.6.0+cu124、batch1、greedy、logits_to_keep=1。",'',f'工作树：`{W}`；独立分支 `codex/qwen3vl-c3-3090-official-eval`。执行基线Commit `30d3e7a0c9318e71a5951aca9a05b6b78108db72`，新增入口由source_manifest.json逐文件绑定。冻结C3核心未改；没有Repaired、训练、层搜索、量化或Attention后端变化。','', '历史1000题/918图像簇的Question ID、Gold、图像SHA和全部输入Tensor绑定与冻结Manifest完全一致；原Manifest SHA256 `bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88`。历史图像按固定MMEB旧Revision恢复，918张JPEG全部精确匹配原SHA。七项数据来自已校验备份，仅对实际读取图片按需核对SHA。','', '## 物理与数值验收','',f"结构检查：**{validation['structural_status']}**；严格数值门禁：**{validation['numerical_status']}**。max_abs≤0.5、mean_abs≤0.02、Top1与完整Token相同的原门槛未变。",'', '最短、典型、最长输入均验证Native/no-op有效Logits严格相同、手动Native与HF generate严格相同、审计开关不改变Physical输出。L22前视觉Hidden退出，P−1/Decode的L1–L6 QKV/QK/AV/o_proj实际跳算；36层Norm/MLP/Residual仍执行，DeepStack三路注入正常，mRoPE绝对位置保留，禁用连接概率为0，Compact KV及请求隔离/Hook清理通过。', '']
 lines+=md(['Question ID','输入Token','视觉Token','最大max_abs','最大mean_abs','完整输出Token相同','严格门禁'],[[s['question_id'],s['input_length'],s['image_tokens'],f(max(d['max_abs'] for d in s['logit_differences'])),f(max(d['mean_abs'] for d in s['logit_differences'])),s['complete_tokens_equal'],s['numerical_status']] for s in validation['smoke']])
 lines+=['','因此本轮保留原始实现的负面数值结果，全部质量和效率结论均标为探索性，不能称为Dense/Physical数值等价。','', '## 历史 GQA1000 严格复现','']
 lines+=md(['条件','完成题数','Accuracy %','空输出','截断','旧参考 %'],[[r['condition'],r['completed']+'/1000',f(r['formal_score']),r['empty'],r['truncated'],{'Native':'69.4','Dense-F':'58.8','Physical-Original':'59.0'}[r['condition']]] for r in hist])
 lines+=['']+md(['条件 − 参考','差值 pp','图片簇95%CI','分数下降/上升','首Token改变'],[[r['condition']+' − '+r['reference'],f(r['delta']),'['+f(r['CI95_low'])+', '+f(r['CI95_high'])+']',r['score_decreases']+'/'+r['score_increases'],r['first_token_changed']] for r in hcomp])
 lines+=['','## 七项质量评测','', '协议差异与固定作者版本见official_protocol_audit_zh.md。官方/ATE存在不同输入时独立生成、独立计分。TextVQA明确保留Paper-OCR；ScienceQA明确为2017道IMG子集；MMBench为1176个完整Circular组的exact_matching，不付费调用裁判；OCRBench原始总分0–1000，MME总分0–2800。','']
 lines+=md(['Benchmark / 协议','条件','完成/应有','正式分数','单位','状态'],[[r['benchmark']+' / '+r['protocol_label'],r['condition'],r['completed']+'/'+r['expected'],f(r['formal_score']),r['metric'],r['status']] for r in summary])
 lines+=['']+md(['Benchmark / 协议','C3 − Native','图片簇95%CI','单位'],[[r['benchmark']+' / '+r['protocol_label'],f(r['delta']),'['+f(r['CI95_low'])+', '+f(r['CI95_high'])+']',r['metric']] for r in comp])
 lines+=['','逐题完整答案、Token IDs、Gold、解析、初始/独立复算分数及空输出/截断均保留。paired_cases.csv.gz给出每题损害/纠正及答案、首Token变化，stratified_results.csv保留问题类型/子任务；重复yes/no或选项答案只作描述，不作为删除样本的理由。七项阶段每个OOM会保留失败题、输入长度和协议不变记录，停止该视图，不降低图像分辨率、不更换后端。','', 'MME保留 `landmark/03d5e3bfc958be38.jpg` 已知单图SHA版本差异及其两题，不替换或删除。因此本轮Native/C3使用相同恢复图片可配对，但不能声称与最早旧实例所有MME图片逐字节相同。','', '## 新卡真实效率','', '48题使用原冻结的48个不同图像SHA，Native、Dense-F、Physical-Original在同一新GPU循环轮换。每输入/条件/模式先2次预热、再5次测量，取各输入重复中位数；主加速为中位数之和的比值。自然EOS、固定16和固定32分别保存。固定32真实执行1次Prefill+31次Decode，遇EOS仍继续。','', '统一E2E边界包括H2D、Execution/Cache上下文创建、Vision/Projector、语言Prefill、真实Decode、greedy与EOS及上下文清理；不含图片磁盘读取、Processor、模型加载或计时后的文本解码。事件时间单列Vision、Prefill与Decode，KV Live/Capacity按实际Tensor元素/存储字节统计，与分配器峰值显存分开。','']
 lines+=md(['模式 / 条件','输入数','KV Live MiB','KV Capacity MiB','E2E ms','TTFT wall ms','TPOT ms','Vision ms','语言Prefill ms','Decode ms','峰值Allocated MiB','Reserved MiB'],[[r['regime']+' / '+r['condition'],r['inputs'],f(r['KV_Live_MiB']),f(r['KV_Capacity_MiB']),f(r['e2e_wall_ms']),f(r['ttft_wall_ms']),f(r['decode_gpu_ms_per_token']),f(r['vision_ms']),f(r['prefill_language_ms']),f(r['decode_gpu_ms']),f(r['Peak_Allocated_MiB']),f(r['Peak_Reserved_MiB'])] for r in eff])
 lines+=['']+md(['模式 / 条件','Native/条件 E2E加速','95%CI','KV Live节省 %','Capacity节省 %'],[[r['regime']+' / '+r['condition'],f(r['speedup'])+'×','['+f(r['CI95_low'])+', '+f(r['CI95_high'])+']',f(r['KV_live_saving_percent']),f(r['KV_capacity_saving_percent'])] for r in ecomp])
 old=O.parent/'qwen3vl_c3v2_exploratory_quality_speed/benchmark_summary.csv'
 if old.exists():
  with old.open() as ff:oldrows=list(csv.DictReader(ff))
  for regime in ['natural','fixed16']:
   new=next((r for r in eff if r['condition']=='Native' and r['regime']==regime and r['inputs']=='48'),None);previous=next((r for r in oldrows if r['condition']=='Native' and r['regime']==regime),None)
   if new and previous:lines+=['',f"Native {regime}：新卡 {f(new['e2e_wall_ms'])} ms vs 历史 {f(previous['e2e_mean_ms'])} ms，描述性漂移 {f((float(new['e2e_wall_ms'])/float(previous['e2e_mean_ms'])-1)*100)}%。历史硬件/驱动/运行状态不同，不将该漂移视为算法加速，不用于本轮C3比值。"]
 lines+=['','## 独立评分、异常及局限','', 'CPU原始标注审计验证GQA全12578题、ScienceQA全部2017整数Gold与Hint/Choices、TextVQA全部5000组十答案、MME1187双题组、MMBench1176完整旋转组、POPE三策略以及OCRBench1000条官方顺序。32条ScienceQA Hint只有冻结Paper模板的连续空白压缩，信息完整，未改输入。','', '新增ScienceQA初始评分适配器曾错误地将解析字母直接与整数Gold比较。已在生成开始前冻结独立离线v2转换（字母→0-based索引），通过专项测试；原始生成、Gold、初始分数全部保留，正式分数只用已独立复核的修正结果，不重跑GPU。逐条修正清单见scienceqa_offline_score_corrections.csv。', '', '第一次GPU进程在第96题写入非ASCII答案时遭遇ASCII文件编码错误。已保存的285条预测按同一模型/源码/输入/协议哈希复用；仅通过启动环境PYTHONUTF8/PYTHONIOENCODING恢复，未改冻结推理源码。未保存完整记录的请求需要重做，完整错误保留在execution.log及execution_interruption.json。', '', '统计采用20000次Image SHA256整簇配对Bootstrap，固定种子20261106，问题加权；MME重算任务Acc+AccPlus，MMBench按完整Circular组，TextVQA只报告软分数变化，不冒充二值翻转。不显著不能证明等价。未完成集合不填写正式成绩；MMBench若需要不可用官方裁判，保留exact_matching分数及AWAITING_OFFICIAL_SCORING状态。', '', '模型作者公开vLLM采样及长输出设置不同于本轮授权的BF16/Eager Greedy，因此本轮不宣称复现Qwen模型榜单。原始Physical算法的严格数值一致性失败限制其等价性主张；质量损失与效率收益必须同时保留。','', '## 保存与资源状态','']
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 lines += [f'当前GPU：`{gpu}`。',f'计算进程：`{apps or "无"}`。','', '全部预测和必要大文件留在本独立持久盘目录；可恢复检查点绑定源码、模型、协议、输入及图片SHA。轻量代码和报告只保存到独立本地分支，未Push、未改main、未创建PR。Windows备份只有真实传输并校验后才在backup_receipt.json中标注完成。','',f"最新执行状态：`{progress['status']}` / `{progress.get('phase','GPU stages finished')}`。本报告不自动启动其他实验，也不关闭服务器。"]
 (O/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (O/'final_status.json').write_text(json.dumps(dict(status=status,GPU_stages_complete=hfin and bool(seven) and efin,all_seven_complete=seven.get('status')=='COMPLETE',numerical_equivalence=validation['numerical_status'],strict_thresholds_unchanged=True,GitHub_push=False,GPU_processes=apps,at_UTC=datetime.now(timezone.utc).isoformat()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(status,flush=True)
if __name__=='__main__':main()
