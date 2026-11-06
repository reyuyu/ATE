"""Report only the stopped backend comparison; no unmeasured quality claims."""
import csv
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from ..mme_seed_quality.analyze import write_csv


def main():
    cases=[read(OUT/f'smoke/backend_comparison_rank{i}.json') for i in (0,1)]
    assert all(c['status']=='FAILED_NUMERICAL_GATE' for c in cases)
    g=read(OUT/'protocol.json')['backend_numerical_gates'];logits=[];features=[];memory=[];operators=[];raw={}
    for c in cases:
        common=dict(rank=c['rank'],uuid=c['uuid'],key=c['key'])
        for d in c['all_same_prefix_logits']:
            logits.append(dict(**common,**d,mean_abs_threshold=g['native_same_prefix_logits_mean_abs'],
                max_abs_threshold=g['native_same_prefix_logits_max_abs'],
                mean_abs_pass=d['mean_abs']<=g['native_same_prefix_logits_mean_abs'],
                max_abs_pass=d['max_abs']<=g['native_same_prefix_logits_max_abs'],
                token_role='first answer token' if d['step']==0 else 'EOS prediction',
                same_complete_tokens=c['tokens_equal']))
        for d in c['features']:
            features.append(dict(**common,**d,feature_name=['pooled','DeepStack L8','DeepStack L16','DeepStack L24'][d['feature']],
                mean_abs_threshold=g['features_mean_abs'],max_abs_threshold=g['features_max_abs'],
                mean_abs_pass=d['mean_abs']<=g['features_mean_abs'],max_abs_pass=d['max_abs']<=g['features_max_abs']))
        for phase in c['phases']:
            b=phase['backend'];memory.append(dict(**common,vision_backend=b['vision_backend'],**phase['memory']))
            operators+= [dict(**common,vision_backend=b['vision_backend'],**d) for d in b['vision_layer_operators']]
        raw[c['key']]=c['raw_arrays_server_only']
        assert sha(raw[c['key']]['path'])==raw[c['key']]['sha256']
    write_csv(OUT/'backend_logits_comparison.csv',logits);write_csv(OUT/'backend_feature_comparison.csv',features)
    write_csv(OUT/'smoke_memory.csv',memory);write_csv(OUT/'actual_vision_backend.csv',operators)
    write_csv(OUT/'backend_comparison_answers.csv',[dict(key=c['key'],rank=c['rank'],
        eager_text=c['eager']['text'],sdpa_text=c['sdpa']['text'],eager_tokens=c['eager']['token_ids'],sdpa_tokens=c['sdpa']['token_ids'],
        tokens_equal=c['tokens_equal'],eager_eos=c['eager']['eos'],sdpa_eos=c['sdpa']['eos']) for c in cases])
    atomic(OUT/'raw_array_sha_index.json',raw)
    allfinite=all(p['backend']['vision_forwards'][0]['all_finite'] for c in cases for p in c['phases'])
    validation=dict(status='STOPPED_NUMERICAL_GATE',at=now(),cpu_tests_passed=5,
        authorization_received=True,backend_config={'vision_config':'sdpa','text_config':'eager'},
        backend_comparison_cases=2,short_native_full_requests=4,actual_cuda_flash_calls=len(operators),
        vision_blocks_per_sdpa_case=27,text_layers_eager=36,math_or_eager_fallback=False,
        feature_thresholds_passed=True,all_observed_values_finite=allfinite,oom_observed=False,
        complete_greedy_tokens_equal_both_cases=True,
        failed_logit_mean_abs_steps=sum(not d['mean_abs_pass'] for d in logits),
        total_logit_steps=len(logits),numeric_thresholds_changed=False,
        native_hf_noop_strict_validation='NOT_RUN',worst_input_memory_validation='NOT_RUN',
        physical_dense_semantic_and_numeric_validation='NOT_RUN',
        formal_mme_conditions_completed=0,formal_seed_conditions_completed=0,
        historical_physical_numeric_equivalence='Previously not passed; unchanged and not retested',
        original_physical_sha=sha(ROOT/'vlm_research/qwen3vl_c3v2_physical/engine.py'),
        source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'))
    atomic(OUT/'validation.json',validation);atomic(OUT/'gpu_smoke_validation.json',validation)
    atomic(OUT/'completion.json',dict(status='STOPPED',reason='Frozen Vision backend same-prefix Logits mean-absolute threshold exceeded',
        at=now(),expected_formal_conditions=33214,completed_formal_conditions=0,short_native_comparison_requests=4,
        worst_smoke_completed=False,full_benchmarks_completed=False,quality_differences=None,
        no_formal_inference_started=True,previous_results_unchanged=True))
    launch=read(OUT/'smoke_launch.json');processes=[]
    for w in launch['workers']:
        f=Path(f'/proc/{w["pid"]}/stat');state=f.read_text().split()[2] if f.exists() else 'ABSENT'
        assert state in ('ABSENT','Z'),f'Own GPU worker still live: {w}'
        processes.append(dict(**w,process_state=state,exited=True))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    assert all(w['uuid'] not in apps for w in launch['workers'])
    atomic(OUT/'gpu_release_audit.json',dict(status='RELEASED',at=now(),workers=processes,
        gpu_snapshot=gpu,compute_processes=apps,own_gpu_workers_live=0,remaining_disk_bytes=shutil.disk_usage('/data').free))
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    names=[f"{d['key']}\n{'first' if d['step']==0 else 'EOS'}" for d in logits]
    for ax,metric,threshold,label in zip(axes,['mean_abs','max_abs'],[0.05,1.0],['Mean absolute logit difference','Max absolute logit difference']):
        ax.bar(range(len(logits)),[d[metric] for d in logits],color=['#c64c43' if d[metric]>threshold else '#3c8c88' for d in logits])
        ax.axhline(threshold,color='black',ls='--',lw=1,label=f'Frozen threshold = {threshold}')
        ax.set_xticks(range(len(logits)),names,fontsize=8);ax.set_ylabel(label);ax.legend(fontsize=8)
    fig.suptitle('Vision Eager vs SDPA: same-prefix Native logits (Text Eager)')
    for ext in ('png','pdf'):fig.savefig(figdir/f'backend_logits_gate.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    x=np.arange(4);width=.36
    for j,c in enumerate(cases):ax.bar(x+(j-.5)*width,[d['max_abs'] for d in c['features']],width,label=c['key'])
    ax.axhline(.5,color='black',ls='--',lw=1,label='Frozen max-abs threshold = 0.5')
    ax.set_xticks(x,['Pooled','DeepStack L8','DeepStack L16','DeepStack L24']);ax.set_ylabel('Max absolute feature difference')
    ax.set_title('Vision feature differences: both short inputs pass');ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(figdir/f'backend_feature_difference.{ext}',dpi=220)
    plt.close(fig)
    table='\n'.join(f"| {d['key']} | {d['token_role']} | {d['mean_abs']:.6f} | {d['max_abs']:.6f} | {'通过' if d['mean_abs_pass'] else '未通过'} |" for d in logits)
    report=f'''# Qwen3-VL：Vision SDPA / Text Eager 续测结果

**最终状态：STOPPED。** 本轮在两张已授权空闲RTX3090上完成两道短输入的Vision Eager/SDPA Native完整请求比较。两题所有27个Vision Block均确认使用实际CUDA Flash SDPA，Text保持36层Eager；两种后端生成Token完全一致且自然EOS结束。但4个有效Logits步骤的平均绝对差均超过事先冻结的0.05阈值，因此停止，未启动MME/SEED正式评估，也未运行最坏输入或Physical C3请求。不能据此报告C3的质量损失，不能宣称最坏输入OOM已解决。

## 固定条件与CPU审计

基于 `213d4e3be766444e1ecd9a90ac479836f7756e9b`，使用独立分支 `codex/qwen3vl-mme-seed-2048-vision-sdpa`。Qwen3-VL-8B-Instruct固定revision为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16、PyTorch 2.6.0+cu124、Transformers 4.57.6、batch1、Greedy、自然EOS、最多32个Token、`logits_to_keep=1`。原2048语言侧视觉Token上限、Processor、Prompt、模型文件及输入Tensor绑定保持不变。

完整复用前轮6811图片路径的真实Processor审计及16607题输入绑定，逐字节SHA与基线Commit一致；重新核验原始图片和模型文件SHA。MME2374问/1187路径、SEED14233条原始记录/5624路径，保留重复Question ID及整图分片。5项CPU测试通过，包括官方模态后端接口及只读SDPA观察器逐位一致的CPU测试。旧LLaVA结果、旧PARTIAL状态和main均未修改。

## 预声明数值门槛

GPU运行前冻结：Vision pooled与三个DeepStack特征Mean Abs≤0.02、Max Abs≤0.5；Native相同前缀的全部有效Logits Mean Abs≤0.05、Max Abs≤1.0。跨后端完整生成Token是否一致单独观察，不要求逐位Logits相等。同后端Native/HF/No-op仍须严格 `torch.equal`；该项因前置门禁失败尚未运行。上述门槛没有根据结果调整。

## 两张卡的实际短输入结果

| 问题键 | 有效步骤 | Logits Mean Abs | Logits Max Abs | Mean Abs门禁 |
|---|---|---:|---:|---|
{table}

`MME__00798`：264个ViT Patch、66个语言侧视觉Token；两种后端均输出 `Yes`，完整Token `[9454, 151645]`。`MME__02358`：260个Patch、65个视觉Token；均输出 `No`，Token `[2753, 151645]`。首Token和EOS预测均相同，不存在空答案或截断。这两道输入依据输入长度/视觉Token数事先选择，不能当作正式Benchmark样本分数。

全部8份视觉特征的误差通过既定阈值，Mean Abs最大0.002857、Max Abs最大0.154297；输出形状、Patch数、语言侧视觉Token数、三个DeepStack索引 `[8,16,24]` 及输入Tensor绑定保持不变。特征小误差在语言模型中形成了上述Logits差异。这里只确认观测，不能据此推断大样本Accuracy损失。

实际调用 `aten._scaled_dot_product_flash_attention.default` 共54次，每道SDPA请求27次；Text未调用SDPA，Math/cuDNN回退被官方上下文禁用。没有计时、Profiler、FLOPs测量或Kernel搜索。

短输入两卡Peak Reserved均为16.4414GiB，余量7.1469GiB；没有OOM或NaN/Inf。此结果只覆盖260/264 Patch，**不覆盖原OOM的8160/8192 Patch**。

## 初始审计器问题与修正记录

第一次尝试被新写的只读观察器误拦：`inference_mode` 暴露外层 `aten.scaled_dot_product_attention.default`，观察器尚未看到内部CUDA算子就拒绝了它。这是本轮审计器错误，不是模型或C3语义失败，也未发生OOM。完整日志、源码SHA及失败现场保存在 `attempts/attempt1_observer_composite_wrapper_failure/`。

修正仅让观察器进入PyTorch已注册的官方Composite分解以观察真实融合算子，不添加Attention算术或自定义Kernel，不改变后端选择和任何阈值。CPU观察前后结果严格相等；修正版本在GPU运行前重新测试、冻结并提交。之后真实GPU数值比较触发当前停止状态。审计器修正不能视为模型数值修复。

## Physical C3与正式质量评估状态

历史Physical-Original引擎保持原字节SHA `{validation['original_physical_sha']}`；保留L0–L21 R写入、L22前物理视觉Hidden退出、P−1/Decode禁图、L1–L6答案Attention物理跳算及原紧凑KV。没有加载Repaired、cuBLASLt或Zero-Key版本。

历史Physical/Dense F的严格门槛Mean Abs≤0.02、Max Abs≤0.5及Token一致没有被修改，历史不等价记录仍有效。本轮没有执行Physical请求，因而没有新的Mask/mRoPE/Cache/退出/跳算验收结论，不能将配置或CPU检查冒充GPU语义验收。

| 项目 | 实际状态 |
|---|---|
| 两卡短输入Native跨后端比较 | 已运行，数值门禁未通过 |
| 真实Vision Flash SDPA / Text Eager | 短输入已确认 |
| 最坏输入完整Native/Physical请求与显存门禁 | 未运行 |
| 同后端Native/HF/No-op严格一致 | 未运行 |
| Physical/Dense F语义及数值检查 | 未运行 |
| MME Native / Physical完整得分 | 未运行，正式Checkpoint 0/4748 |
| SEED Native / Physical完整ACC | 未运行，正式Checkpoint 0/28466 |
| 图片簇配对20000次Bootstrap和质量图 | 无完整预测，不计算 |

MME与SEED的CSV保留覆盖状态，不填入预测分数或零分。SEED仍冻结为本地HF完整候选平均条件负LogP排序、最终ACC，不冒充官方公布分数。没有转用自由生成或改变候选协议。

## 文件与资源释放

逐步骤Logits与特征误差分别见 `backend_logits_comparison.csv`、`backend_feature_comparison.csv`；真实运算符、显存及答案见 `actual_vision_backend.csv`、`smoke_memory.csv`、`backend_comparison_answers.csv`。图表PNG/PDF位于 `figures/`，只画真实观测的数值差异，不伪造质量图。

完整特征与Logits数组保存在服务器 `/data/lk/ATE-resources/experiments/qwen3vl_mme_seed_2048_vision_sdpa/`，路径和SHA见 `raw_array_sha_index.json`；这些数组不在GitHub。原始停止日志与失败现场均保留。

两张卡的本轮Worker均已退出，当前各27MiB，无计算进程；详见 `gpu_release_audit.json`。未启动正式评估、未追加GPU实验，也没有降低分辨率、改变精度或放宽数值门槛。

**本轮最终回答：** 目前只能确认短输入能够通过官方Vision Flash SDPA完成Native请求，且这两题的生成答案保持一致；无法确认最坏输入OOM是否解决。由于预声明跨后端Logits门禁失败，MME和SEED的Native/Physical完整质量差均尚未测得，不能回答C3损失多少质量，也不能判断退化是否集中于OCR或空间关系。
'''
    (OUT/'report_zh.md').write_text(report,encoding='utf-8')
    print('STOPPED REPORT AND RELEASE AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
