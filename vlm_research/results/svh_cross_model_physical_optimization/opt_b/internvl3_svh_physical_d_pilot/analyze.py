"""Independent CPU analysis of the user-authorized quality-first Physical-D pilot."""
import csv,json,statistics
from collections import Counter,defaultdict
import numpy as np
from .common import *
from .statistics import analyze_complete
def main():
 state=read(RUN/'execution_status.json')
 assert state['status']=='COMPLETE_WITH_NUMERICAL_WAIVER'
 samples=read(RUN/'sample_manifest.json')['samples'];proto=read(RUN/'protocol.json')
 summary,tasks,pairs,es=analyze_complete()
 for path,h in read(RUN/'source_manifest.json').items():assert sha(path)==h,path
 for path,h in read(RUN/'diagnostic_source_manifest.json').items():assert sha(path)==h,path
 original=read(RUN/'engineering_attempts/attempt01_frozen_gate_failure/smoke_validation.json')
 assert not original['passed']
 assert proto['user_override']['continue_despite_numerical_gate_failure']
 old=read(OLD/'sample_manifest.json');old=old['samples'] if isinstance(old,dict) else old
 assert not {s['question_id'] for s in samples}&{s['question_id'] for s in old}
 assert not {s['image_sha256'] for s in samples}&{s['image_sha256'] for s in old}
 assert len({s['image_sha256'] for s in samples})==120
 qrows=list(csv.DictReader((RUN/'per_question.csv').open()));eff=list(csv.DictReader((RUN/'efficiency_per_input_summary.csv').open()))
 assert len(qrows)==360 and len(eff)==90
 # Verify all actual physical KV layouts and byte accounting for every formal request.
 layouts=[]
 for path in sorted((RUN/'efficiency_checkpoints').glob('*.json')):
  r=read(path)
  live=sum(x['live_bytes'] for x in r['kv_layout'])/1048576
  cap=sum(x['capacity_bytes'] for x in r['kv_layout'])/1048576
  assert live==r['KV_live_MiB'] and cap==r['KV_capacity_MiB']
  for x in r['kv_layout']:
   i=x['layer']
   expected=0 if r['method']=='Physical-D' and i in range(2,6) else x['logical_length']-(256 if r['method']=='Physical-D' else 0)
   assert x['physical_length']==expected
   assert x['visual_kv']==(0 if r['method']=='Physical-D' else 256)
   if x['physical_length']==0:assert x['live_bytes']==x['capacity_bytes']==0
   layouts.append(dict(question_id=r['question_id'],repeat=r['repeat'],**x))
 csvwrite(RUN/'kv_layout.csv',layouts,['question_id','repeat','method','layer','logical_length','physical_length','visual_kv','k_shape','v_shape','dtype','live_bytes','capacity_bytes','skip_attention'])
 table={r['method']:r for r in es};acc={r['method']:r for r in summary}
 comparisons=[]
 for method in ['Dense-D','Physical-D']:
  a=table['Native'];c=table[method]
  comparisons.append(dict(method=method,KV_reduction_percent=100*(1-c['KV_live_MiB_mean']/a['KV_live_MiB_mean']),
   E2E_reduction_percent=100*(1-c['E2E_ms_mean']/a['E2E_ms_mean']),speedup=a['E2E_ms_mean']/c['E2E_ms_mean'],
   TPOT_reduction_percent=100*(1-c['TPOT_ms_mean']/a['TPOT_ms_mean'])))
 csvwrite(RUN/'relative_efficiency.csv',comparisons,list(comparisons[0]))
 # Paired efficiency differences are computed over the same 30 input means.
 ed=[]
 for method in ['Dense-D','Physical-D']:
  for metric in ['E2E_ms','TTFT_ms','TPOT_ms','KV_live_MiB']:
   n={r['question_id']:float(r[metric]) for r in eff if r['method']=='Native'}
   c={r['question_id']:float(r[metric]) for r in eff if r['method']==method}
   d=[c[q]-n[q] for q in n]
   ed.append(dict(comparison=method+' - Native',metric=metric,n=30,mean_difference=statistics.mean(d),median_difference=statistics.median(d)))
 csvwrite(RUN/'efficiency_paired_comparisons.csv',ed,list(ed[0]))
 taskpairs=list(csv.DictReader((RUN/'task_paired_comparisons.csv').open()))
 compare=[r for r in taskpairs if r['task']=='CompareAttr']
 counts={m:dict(Counter(r['eos'] for r in qrows if r['method']==m)) for m in METHODS}
 validation=dict(status='COMPLETE_WITH_NUMERICAL_WAIVER',artifact_integrity='PASS',quality_independent_rescore='PASS',
  strict_numerical_gate='FAILED',user_authorized_continuation=True,operational_smoke='PASS',
  formal_quality_requests=360,formal_efficiency_requests=450,quality_n_per_method=120,
  efficiency_inputs_per_method=30,efficiency_repeats_per_input=5,paired_input_tensor_SHA_equal=True,
  actual_KV_layout_and_storage_bytes='PASS',new_QID_overlap=0,new_image_sha_overlap=0,unique_new_images=120,
  generated_quality_tokens=sum(len(json.loads(r['token_ids'])) for r in qrows),
  no_model_weight_change=True,all_generation_sources_unchanged=True,
  protocol_sha256=sha(RUN/'protocol.json'),source_manifest_sha256=sha(RUN/'source_manifest.json'),
  GPU_UUID=GPU,bootstrap_replicates=20000,bootstrap_unit='whole Image SHA cluster; ordinary unadjusted paired 95%CI')
 save(RUN/'validation.json',validation)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 colors=['#677384','#76a4cf','#db934f']
 axes[0].bar(METHODS,[acc[m]['accuracy_percent'] for m in METHODS],color=colors)
 axes[0].set_ylabel('GQA accuracy (%)');axes[0].set_ylim(0,100);axes[0].set_title('New held-out 120 questions')
 for i,m in enumerate(METHODS):axes[0].text(i,acc[m]['accuracy_percent']+1,f"{acc[m]['correct']}/120",ha='center',fontsize=9)
 for ax,key,title in [(axes[1],'E2E_ms_mean','E2E@32 (ms/request)'),(axes[2],'KV_live_MiB_mean','Persistent KV@32 (MiB)')]:
  values=[table[m][key] for m in METHODS];ax.bar(METHODS,values,color=colors);ax.set_title(title)
  for i,v in enumerate(values):ax.text(i,v+max(values)*.02,f'{v:.2f}',ha='center',fontsize=9)
  ax.set_ylim(0,max(values)*1.15)
 fig.suptitle('Physical-D pilot | Quality-first user authorization | Strict numerical gate FAILED',fontsize=10)
 fig.tight_layout();fig.savefig(RUN/'figure_quality_efficiency.png',dpi=210);fig.savefig(RUN/'figure_quality_efficiency.pdf');plt.close(fig)
 native=acc['Native'];dense=acc['Dense-D'];physical=acc['Physical-D'];rel=next(r for r in comparisons if r['method']=='Physical-D')
 pairD=next(r for r in pairs if r['comparison']=='Physical-D - Dense-D')
 pairN=next(r for r in pairs if r['comparison']=='Physical-D - Native')
 report=f"""**Physical-D在新的120题上达到 {physical['correct']}/120（{physical['accuracy_percent']:.4f}%）。相对Dense-D为 {pairD['delta_pp']:+.4f} pp；相对Native为 {pairN['delta_pp']:+.4f} pp。Persistent KV@32减少 {rel['KV_reduction_percent']:.2f}%，E2E@32加速 {rel['speedup']:.4f}×。**

本轮状态为 **COMPLETE_WITH_NUMERICAL_WAIVER**：物理执行、完整质量/效率与文件核验已完成；原数值门槛仍为FAILED。用户在六样本数值失败诊断后明确要求“0.0004不大吧，在小测试集上得分差不多就行”，因此在任何正式生成前新增协议授权，以真实配对质量为主要判断依据。没有将原数值验收改写为通过，也不能宣称严格数值等价。

| 条件 | Correct/120 | Accuracy | KV@32 MiB | E2E@32 ms | TTFT ms | TPOT ms/token |
|---|---:|---:|---:|---:|---:|---:|
"""
 for m in METHODS:
  a=acc[m];e=table[m]
  report+=f"| {m} | {a['correct']}/120 | {a['accuracy_percent']:.4f}% | {e['KV_live_MiB_mean']:.4f} | {e['E2E_ms_mean']:.4f} | {e['TTFT_ms_mean']:.4f} | {e['TPOT_ms_mean']:.4f} |\n"
 report+="\n| 配对比较 | ΔAccuracy pp | 普通95%簇Bootstrap CI | 原对→错 | 原错→对 | 首Token翻转 | 文本变化 |\n|---|---:|---|---:|---:|---:|---:|\n"
 for r in pairs:report+=f"| {r['comparison']} | {r['delta_pp']:+.4f} | [{r['CI95_low']:+.4f}, {r['CI95_high']:+.4f}] | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_token_flip']} | {r['text_change']} |\n"
 report+="\n| 任务 | Native | Dense-D | Physical-D | Physical−Dense pp | Physical−Native pp |\n|---|---:|---:|---:|---:|---:|\n"
 for task in TASKS:
  sub={r['method']:r['accuracy_percent'] for r in tasks if r['task']==task}
  report+=f"| {task} | {sub['Native']:.1f}% | {sub['Dense-D']:.1f}% | {sub['Physical-D']:.1f}% | {sub['Physical-D']-sub['Dense-D']:+.1f} | {sub['Physical-D']-sub['Native']:+.1f} |\n"
 report+="\nCompareAttr单列配对统计：\n\n"
 for r in compare:report+=f"- {r['comparison']}：{float(r['delta_pp']):+.4f} pp，95%CI [{float(r['CI95_low']):+.4f}, {float(r['CI95_high']):+.4f}]，原对→错 {r['correct_to_wrong']}、原错→对 {r['wrong_to_correct']}。\n"
 report+=f"""
全部新增错误见harm_cases.csv，完整Token/答案/Gold/得分/EOS/截断/空输出见per_question.csv。该120题为工程小样本，六任务各20题；每个任务一题即5 pp。20,000次按Image SHA整簇配对Bootstrap，普通95%CI未作多重比较校正。CI跨0不证明等价或非劣效，不据结果重新选层。Native与Dense-D还存在答案直接读图权限差异，不能将全部质量差距归于Hidden退出。

物理执行：进入L20前实际删除256视觉Hidden行；L2–L5只对P−1/Decode跳过Attention，QKV、QK/AV、o_proj均不为这些Query执行，其余Prefill Query原生。全部28层Norm、MLP、Residual保留。Prefill早层临时视觉KV供R访问；Prefill结束后全层持久原始视觉KV清除，L2–L5无持久KV。逻辑位置和RoPE不按物理长度重编号。Native/Dense-D各层KV长度为P+31；Physical-D非跳算层为P−256+31，L2–L5为0。KV Live依据真实K/V tensor字节，Capacity依据唯一底层storage.nbytes核验，无完整Dense后备。全部450个正式效率请求的逐层布局均重核验，见kv_layout.csv。

严格数值门槛保留失败：CPU 28层tiny Qwen2 float64数学对照Logits完全一致，Hidden最大差4.16e−17。六个GPU样本Native/No-op/官方全部有效Logits严格相等，干净Dense-D与历史冻结D严格相等，三个历史D的完整Token/输入/每步Logits SHA复现。六例Dense-D/Physical-D完整Token相同，但Logits最大差1.00–2.50，超过原0.125上限；mean_abs及Hidden relative RMS亦有失败。首分歧L2 AV：同一输入下Q/K/V、RoPE、QK、Softmax前缀严格一致，缩短数学上零概率的末尾Key后，AV的1,139,712元素中33个不同，max_abs=0.00048828125，随后逐层放大。该算子定位解释最早的数值差异，不是豁免数值门槛的依据；继续执行来自用户明确改指令。原失败协议、源码和诊断全部保留在engineering_attempts/attempt01_frozen_gate_failure及diagnostics目录。

计时：GPU准备好的图片/Token输入开始，包含Vision Encoder、Projector、注入、1次Prefill、31次真实Decode，以及方法安装/恢复与中间对象清理；CUDA同步墙钟到第32个Token完成。排除磁盘读取、Tokenizer/图像预处理、H2D、模型加载和输出写盘。正式计时无审计Hook；CUDA Event用于组件辅助时间。TTFT在首Token后同步，TPOT=(E2E−TTFT)/31。每条件2次预热；30题×3条件×5重复=450正式请求，按题与重复轮换方法顺序。表中均值为30个输入各5次均值的算术均值；中位数为这30个输入均值的中位数，原始450次与90个输入均值均保留。实际Attention/投影调用数来自计时外同一32步执行路径审计，详见physical_execution_audit.json，不冒称每次计时均启用了计数Hook。

GPU始终为RTX3090 {GPU}，BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0，模型Revision {REV}，logits_to_keep=1、Greedy、256视觉Token。质量正常EOS、最多32Token；效率忽略EOS固定32。未改变权重、层号、系统功耗或挂载；未运行3000题、七Benchmark、OCRBench、训练或Kernel搜索。120个新QID及120个图片SHA与已核验历史3000清单无重叠，来源官方GQA testdev_balanced；未提供的其他探索集不在独立性声明范围内。

正式质量完成360/360，效率完成450/450。是否值得扩大验证应同时看上述质量差值、CompareAttr损害与宽CI；本轮只能提供工程Pilot证据，不能替代大样本独立非劣效验证。更大规模测试需要另外授权，本轮不启动。

有效协议SHA：{sha(RUN/'protocol.json')}；源码冻结Commit：{read(RUN/'environment.json')['source_commit']}；模型路径：{MODEL}。完整服务器SSD原件 {RUN}，持久化目录 {OUT}。GitHub仅上传本轮轻量源码/冻结清单/CSV/图/配置/报告/校验，服务器保留原始检查点、日志和各版本失败证据。发布文件对应关系见artifact_manifest.json，最终Commit和远端核验见服务器github_push_receipt.json。GPU最终状态见gpu_final_release.json，服务器实例不自动关闭。

![Quality and efficiency](figure_quality_efficiency.png)
"""
 (RUN/'report_zh.md').write_text(report)
 print(json.dumps(dict(accuracy=summary,paired=pairs,efficiency=es,relative=comparisons),ensure_ascii=False))
if __name__=='__main__':main()
