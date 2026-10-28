"""Offline, honest partial-run summary; never imputes missing GPU outcomes."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'results/attention_inheritance_15133706');out=parser.parse_args().output_dir
    initial=out/'initial_attempt_fp16_validation';blocked=out/'resource_blocked_attempt'
    if (out/'validation.json').exists() and json.loads((out/'validation.json').read_text()).get('status')=='completed':raise ValueError('Refuse to replace completed results with a partial report')
    initial_rows=json.loads((initial/'per_condition_results.json').read_text())
    if len(initial_rows)!=1 or initial_rows[0]['source_layer']!=23 or initial_rows[0]['lambda']!=0:raise ValueError('Unexpected partial checkpoint')
    full=initial_rows[0]['full_margin'];physical=initial_rows[0]['historical_physical_k24_margin']
    rows=[]
    for source in [23,24]:
        for strength in [0,.5,1]:
            match=next((r for r in initial_rows if r['source_layer']==source and r['lambda']==strength),None)
            rows.append(dict(question_id='15133706',source_layer=source,**{'lambda':strength},
                             status='gpu_verified_initial_validator' if match else 'not_completed',
                             margin=match['margin'] if match else None,gain_vs_full=match['gain_vs_full'] if match else None,
                             correct_logp=match['correct_logp'] if match else None,wrong_logp=match['wrong_logp'] if match else None,
                             predicted_candidate=match['predicted_candidate'] if match else None,
                             forward_seconds_two_candidates=match['forward_seconds_two_candidates'] if match else None,
                             peak_allocated_mib=match['peak_allocated_mib'] if match else None,
                             reason='' if match else 'Initial nonzero pass stopped on FP16 conditional-L1 validation; corrected runner blocked before load by new GPU process'))
    save(out/'per_condition_results.json',rows)
    with (out/'per_condition_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with np.load(initial/'prediction_logits.npz') as logits:
        exact=bool(np.array_equal(logits['native_full'],logits['L23_lambda0']))
    mapping=json.loads((initial/'input_mapping.json').read_text());ids=[c['answer_records'][0]['token_id'] for c in mapping['candidate_mapping']]
    with np.load(initial/'prediction_logits.npz') as f:
        assert exact and np.array_equal(f['native_full'][0],f['native_full'][1])
        measured_margin=float(f['native_full'][0,ids[1]])-float(f['native_full'][0,ids[0]])
    if measured_margin!=full:raise ValueError('Saved Full margin mismatch')
    with np.load(initial/'reference_distributions.npz') as ref,np.load(ROOT/'results/phase3_4_roi_sink_diagnostic/attention_rows/15133706_task.npz') as cache:
        reference_exact=all(np.array_equal(ref[f'layer{l}_raw'],cache['attention'][l,:,1,:].astype(np.float32)) for l in [23,24])
    audits=json.loads((initial/'audit_L23_lambda0.json').read_text());layers=[r for c in audits for r in c['layer_audits']]
    if len(layers)!=16 or not all(r['passed'] and r['lambda_zero_tensor_exact'] for r in layers):raise ValueError('Initial lambda0 audit failure')
    resource=json.loads((blocked/'gpu_resource.json').read_text())
    library=json.loads((initial/'experiment_config.json').read_text())['library']
    library_exact=sha(library['library_file'])==library['library_file_sha256']
    cpu_log=(out/'cpu_regression.txt').read_text()
    test_match=re.search(r'Ran (\d+) tests? in',cpu_log)
    if not test_match or not cpu_log.rstrip().endswith('OK'):raise ValueError('CPU test log missing success')
    test_count=int(test_match.group(1))
    validation=dict(status='incomplete_gpu_resources',experiment_completed=False,completed_condition_count=1,planned_condition_count=6,
                    cpu_regression_tests=test_count,cpu_regression_passed=True,lambda0_L23_saved_vocabulary_logits_exact=exact,
                    fresh_source_L23_L24_match_cache=reference_exact,third_party_library_unchanged=library_exact,
                    lambda0_local_audit_count=16,lambda0_max_row_sum_error=max(a['row_sum_max_abs_error'] for a in layers),
                    nonzero_lambda_gpu_outcomes_available=False,latest_resource=resource,
                    completed_lambda0_forward_seconds=initial_rows[0]['forward_seconds_two_candidates'],
                    completed_lambda0_peak_allocated_mib=initial_rows[0]['peak_allocated_mib'],
                    corrected_validator_gpu_tested=False,
                    artifacts_sha256={str(p.relative_to(out)):sha(p) for p in [initial/'prediction_logits.npz',initial/'reference_distributions.npz',initial/'audit_L23_lambda0.json',out/'per_condition_results.csv']},
                    final_code_sha256={p:sha(ROOT/p) for p in ['attention_inheritance.py','attention_inheritance_pilot.py','test_attention_inheritance.py','summarize_attention_inheritance.py']})
    save(out/'validation.json',validation)
    fig,ax=plt.subplots(figsize=(8.8,5.3))
    bars=ax.bar([0,1,2],[full,initial_rows[0]['margin'],physical],color=['#777777','#327da8','#528d5f'],width=.62)
    for b,value in zip(bars,[full,full,physical]):ax.annotate(f'{value:+.8f}',(b.get_x()+b.get_width()/2,value),xytext=(0,6 if value>0 else -16),textcoords='offset points',ha='center')
    ax.axhline(0,color='black',lw=.9)
    ax.set(xticks=[0,1,2],xticklabels=['Native Full\n(current verified)','L23 inheritance, lambda=0\n(current verified)','Physical K24\n(historical only)'],ylabel='logP(right) - logP(left) [nats]',ylim=(-.055,.065))
    ax.set_title('15133706 Girl / Boy: correct answer = right\nINCOMPLETE: no nonzero-lambda answer results',fontsize=12)
    ax.text(.5,.95,'L24-L31, P-1 -> Image only; L23/L24 x lambda 0.5/1 pending',transform=ax.transAxes,ha='center',fontsize=9)
    fig.text(.5,.015,'Do not infer improvement from missing conditions. Near-zero FP16 margins; no causal mechanism conclusion.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'margin_comparison.png',dpi=180);plt.close(fig)
    report=f'''# 15133706 单案例 Attention 继承：实验未完成

**状态：已实现并通过101项CPU回归；真实GPU仅完成L23、λ=0。非零λ的答案结果尚不存在。** 不能判断继承是否提高right Margin或使其转正。所有CSV空值均为未完成，不是零值、负面结果或推算结果。

问题：**Is the young girl to the left or to the right of the boy?** 标准答案right；候选顺序left/right保持历史不变。仅此一张图，不训练、不扩展样本。

## 已实际获得的分数

| 条件 | Margin (nats) | 来源 |
| --- | --- | --- |
| Full | {full:+.8f} | 本轮真实7B，精确匹配历史Full |
| Full L23→L24～L31，λ=0 | {full:+.8f} | 本轮两候选真实前向，整个P−1词表logits与Full逐元素相同 |
| Physical K24 | {physical:+.8f} | Phase3.2历史数据；本轮没有重跑Physical |
| Full L23，λ=0.5/1 | 未完成 | 第一次λ=.5在L25验证停止，没有输出候选分数 |
| Full L24，λ=0/0.5/1 | 未完成 | GPU资源阻塞，未完成条件 |

![只展示实测与明确标注的历史分数](margin_comparison.png)

Full的right/left logP为−18.5991971377/−18.5679471377，原始FP16 logits为8.2578125/8.2890625。正确候选的全词表概率约8.3651e−9；二候选相对概率约0.49219。Margin接近零，候选正确不等于开放生成正确。

## 干预实现与定位

使用既有load_model、Processor、问题/候选定位及完整候选评分。Prompt长度603；P−1=602；输入候选后长度604；视觉位置5～580，共576。left/right各一Token，其预测Query均为602，**不使用已经输入的答案位置603**。

固定Full L23或L24的每个Head视觉行，分别归一化成p_ref；新鲜Full参考与Phase3.4缓存逐元素相同。L24～L31每层保持Head对应关系，令当前层视觉行a、m=sum(a)，仅在Softmax后、乘当前层V之前写入：

`new_visual = m * ((1−λ) * a/m + λ * p_ref)`

其余Query、P−1非视觉权重与原始Mask不改，不做全行重新归一化。λ=0跳过写入。数值修改发生在现有FP32 Softmax转成FP16之后，替换值也保持FP16；模型权重、V、RoPE、位置、候选及Prompt不变。后层局部编辑前的权重可能因前层干预自然变化，不能宣称其等于Full。

通过模型实例私有forward/globals包装eager函数，无全局softmax猴子补丁，不改第三方源码；上下文finally恢复。头h跨深度直接对应，不假定它们具有相同语义功能。无KV Cache/Decode、无物理删除、无加速声明。

## 验证结果与首次停止

7项新增CPU测试覆盖λ0全logits/Attention精确、混合公式、因果早期位置、P@V计算、FP16量化、异常清理、实例隔离及非法输入；全工程共101项回归通过，见`cpu_regression.txt`。

真实GPU初次执行：Full分数和输入映射与历史完全一致；Full L23/L24参考与缓存完全一致；L23 λ0两个候选的P−1全部logits、Question最终hidden精确等于Full，16个层级审计全部通过；行和最大误差{validation['lambda0_max_row_sum_error']:.9g}，视觉Mass改变量0，非目标连接逐元素不变。

首个非零条件（L23 λ=.5）在L25停止：某Head视觉Mass约0.00209993，FP16条件分布L1误差0.00429884，超初始固定0.002阈值。行和误差0.00024229，最大Mass绝对误差7.81e−6，非目标连接精确。该条件**没有完整Margin**。完整失败、初始配置和已完成分数保存在`initial_attempt_fp16_validation/`。

调查表明576个FP16子正常数的绝对舍入，经极小Mass条件化会放大；固定L1阈值未覆盖此情形。干预公式与计算完全不改，新增逐元素等于公式FP16舍入值的bitwise检查，并以dtype/质量推导的界验证：`u=eps/2; d=tiny*eps; E=u*sum(abs(x))+576*d/2; L1<=2E/sum(round(x))+2e−6`。这是舍入误差和三角不等式的上界，不从答案或实测误差拟合。原0.002阈值保留为显式告警；原行和0.002、Mass atol2e−5+rtol0.001不变。**修正验证器只通过CPU，尚未经真实7B非零λ验证，不能称整个实现已通过GPU验收。**

## GPU资源阻塞和真实开销

第一次运行时GPU0约17GB空闲，Full及λ0成功。修正验证后重跑前，其他用户进程1536339新增约10104MiB；结合原进程7524MiB，GPU0空闲仅约{resource['free_mib']:.1f}MiB，低于既有16000MiB门槛。GPU1也被占用。资源检查在模型加载前停止，没有CUDA OOM、没有终止其他任务，没有启用自动轮询。见`resource_blocked_attempt/`。

已完成λ0两候选诊断耗时**{initial_rows[0]['forward_seconds_two_candidates']:.6f}秒**、峰值allocated **{initial_rows[0]['peak_allocated_mib']:.3f}MiB**、reserved **{initial_rows[0]['peak_reserved_mib']:.3f}MiB**。计时含本地审计开销且GPU共享，非稳定性能比较。首次环境/加载记录保存在初次尝试目录；当前仍为Python3.11.18、PyTorch2.6.0+cu124、Transformers4.49.0、CUDA12.4及原模型revision。安装库SHA256复核未改变。

## 观察与机制解释

实际仅证明λ0实现路径保持Full；**尚不能回答非零继承是否改善答案**。Full与历史Physical的符号差异本就很小。即便后续某条件转正，单个已挑选案例、FP16弱Margin和一个Query行的干预，也不能证明Sink有害、解释Physical全部效应或代表一般能力提升。Physical改变多位置的信息流和计算，不能与本干预等同。

## 恢复方式

GPU资源足够后，手动运行；保留已有尝试，使用新的独立目录：

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_inheritance_pilot.py --output-dir results/attention_inheritance_15133706/completed_retry
```

只会重跑此一个案例的Full与6条件；仍先检查资源并拒绝覆盖已有配置目录。若再次失败，保留failure.json及逐条件已完成结果，不更改候选/强度/层以追求转正。此摘要可纯CPU重建：`CUDA_VISIBLE_DEVICES='' python summarize_attention_inheritance.py`。本次到此停止，需GPU可用后才能完成用户要求的真实对比。
'''
    (out/'report.md').write_text(report)
    (out/'README.md').write_text('# 15133706 Attention 继承（未完成）\n\n**只有L23、λ=0完成真实GPU验证；其他条件暂无Margin。**\n\n[实验报告与恢复命令](report.md) · [逐条件CSV](per_condition_results.csv) · [验证状态](validation.json)\n\n![实际分数对比](margin_comparison.png)\n')
    print(json.dumps({k:v for k,v in validation.items() if k not in ['latest_resource','artifacts_sha256','final_code_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
