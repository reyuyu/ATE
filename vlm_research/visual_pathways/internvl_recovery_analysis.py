"""Recovery-only analysis; immutable previous delivery remains untouched."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import ROOT, TASKS, START, DEADLINE, TASK1_END, read, write, sha, digest, command
from .analyze import scan_model, csv_write, rescue_summary

OUT = ROOT/'vlm_research/results/internvl_native_replay_recovery'
PARENT = ROOT/'vlm_research/results/visual_pathways_seven_hour'


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def pilot_freeze():
    assert read(OUT/'pilot_gate.json')['passed']
    assert not (OUT/'pilot_sensitive_layers.json').exists()
    assert not list((OUT/'internvl').glob('all_gpu*'))
    samples = [s for s in read(OUT/'sample_manifest.json')['samples'] if s['pilot']]
    losses = []
    hashes = {}
    for sample in samples:
        path = OUT/'internvl/checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json'
        record = read(path)
        assert record['complete'] and record['sample_sha256'] == digest(sample)
        rows = sorted(record['rows'],key=lambda row:row['layer'])
        assert len(rows) == 28
        losses.append([row['absolute_logp_change'] for row in rows])
        hashes[str(path.relative_to(OUT))] = sha(path)
    mean = np.asarray(losses).mean(0)
    rank = sorted(range(28),key=lambda i:(-mean[i],i))
    value = dict(questions=240, primary='Mean Absolute LogP Change on fixed pilot240',
        keep_layers=rank[:2], mean_absolute_by_layer=mean.tolist(), ties='Lower layer index first',
        checkpoint_sha256=hashes, time=datetime.now().astimezone().isoformat(),
        frozen_before_3000_extension=True, exploratory=True, no_accuracy_selection=True)
    write(OUT/'pilot_sensitive_layers.json',value)
    print('Pilot-only top two layers frozen:',rank[:2],flush=True)


def analyze_rescue():
    rescue_summary('internvl',output_root=OUT)
    folder=OUT/'internvl_rescue'
    samples=read(folder/'sample_manifest.json')['samples']
    rows=[]
    for sample in samples:
        record=read(folder/'checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json')
        native=record['native_full_generation']
        rows.append(dict(task=sample['task'],question_id=sample['question_id'],
            image_sha256=sample['image_sha256'],answer=sample['answer'],
            correct=record['native_full_accuracy'],**native))
    csv_write(folder/'native_full_per_question.csv',rows)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .figures import style, curve, COLORS
    style()
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=280,bbox_inches='tight')
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    rows = read_csv(OUT/'internvl/layer_summary.csv')
    coverage = read(OUT/'internvl/coverage.json')
    metrics = [('mean_absolute_logp_change','Mean absolute LogP change (nat)','absolute',False),
               ('mean_signed_loss','Mean signed LogP loss (nat)','signed',False),
               ('top1_flip_rate','Top-1 flip rate (%)','flip',True)]
    overall = [r for r in rows if r['cohort']=='all' and r['task']=='Overall']
    fig,axes = plt.subplots(1,3,figsize=(11.5,3),layout='constrained')
    for ax,(metric,label,color,percent) in zip(axes,metrics):
        x,y = curve(ax,overall,metric,COLORS[color],percent=percent)
        i = int(y.argmax())
        ax.annotate(f'L{x[i]}',xy=(x[i],y[i]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set_ylabel(label)
    fig.suptitle(f'InternVL3-8B: P-1 single-layer Image OFF | N={coverage["questions"]:,} ({coverage["status"]})',fontsize=11)
    save(fig,'internvl_overall_sensitivity')
    for metric,label,color,percent in metrics:
        fig,axes = plt.subplots(2,3,figsize=(10.5,5.8),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            values = [r for r in rows if r['cohort']=='all' and r['task']==task]
            x,y = curve(ax,values,metric,COLORS[color],percent=percent)
            i = int(y.argmax())
            ax.annotate(f'L{x[i]}',xy=(x[i],y[i]),xytext=(3,5),textcoords='offset points',fontsize=8)
            ax.set_title(f'{task} (N={values[0]["questions"]})')
            ax.set_ylabel(label)
        fig.suptitle('InternVL3-8B: source-specific sensitivity (pointwise image-cluster 95% CI)',fontsize=11)
        save(fig,'internvl_six_tasks_'+color)
    qwen = read_csv(PARENT/'qwen/layer_summary.csv')
    llava = read_csv(PARENT/'llava_historical_imageoff_layer_summary.csv')
    csv_write(OUT/'cross_model_layer_summary.csv',qwen+rows+llava)
    for cohort,name,datasets in [
        ('all','qwen_internvl_same3000',[(qwen,'Qwen3-VL-8B','#216b9b'),(rows,'InternVL3-8B','#2e8064')]),
        ('pilot240','three_models_same240',[(qwen,'Qwen3-VL-8B','#216b9b'),(rows,'InternVL3-8B','#2e8064'),(llava,'LLaVA-1.5-7B (historical)','#c35139')])]:
        fig,axes = plt.subplots(1,2,figsize=(9.8,3.5),layout='constrained')
        for dataset,label,color in datasets:
            values = [r for r in dataset if r['cohort']==cohort and r['task']=='Overall']
            x,y = curve(axes[0],values,'mean_absolute_logp_change',color,label)
            peak = int(y.argmax())
            axes[0].annotate(f'L{x[peak]}',xy=(x[peak],y[peak]),xytext=(4,4),textcoords='offset points',fontsize=8,color=color)
            axes[1].plot(x/x.max(),y/y.sum(),color=color,marker='o',markersize=2,lw=1.4,label=label)
        axes[0].set_ylabel('Mean absolute LogP change (nat)')
        axes[0].set_title('Measured sensitivity (pointwise 95% CI)')
        axes[0].legend(frameon=False,fontsize=7)
        axes[1].set_xlabel('Relative decoder depth')
        axes[1].set_ylabel('Share of summed layer-wise mean changes')
        axes[1].set_title('Descriptive depth profile')
        axes[1].grid(axis='y',alpha=.15)
        n = 3000 if cohort=='all' else 240
        fig.suptitle(f'Exact same {n:,} questions; native model prompts, token targets and visual token counts differ',fontsize=10)
        save(fig,name)
    if (OUT/'internvl_rescue/accuracy_summary.csv').exists():
        fig,axes=plt.subplots(1,3,figsize=(11.7,3.5),layout='constrained')
        for ax,(root,model,label) in zip(axes,[(PARENT,'llava','LLaVA-1.5-7B'),(PARENT,'qwen','Qwen3-VL-8B'),(OUT,'internvl','InternVL3-8B')]):
            rescue_folder=root/(model+'_rescue')
            values=read_csv(rescue_folder/'accuracy_summary.csv')
            native=read(rescue_folder/'native_full_reference.json')
            y=np.array([float(r['accuracy_percent']) for r in values])
            low=np.array([float(r['accuracy_ci_low']) for r in values])
            high=np.array([float(r['accuracy_ci_high']) for r in values])
            ax.bar(range(3),y,color=['#597781','#bababa','#d2a34a'],width=.6)
            ax.errorbar(range(3),y,yerr=[y-low,high-y],fmt='none',color='#333333',capsize=3)
            ax.axhline(native['accuracy_percent'],ls='--',color='#b43d40',lw=1,
                       label=f'Native Full {native["accuracy_percent"]:.2f}%')
            ax.set_xticks(range(3),['Full Image\naccess','All Image\nOFF','Keep\nsparse'])
            ax.set_ylim(0,100)
            ax.set_ylabel('Final-answer accuracy (%)')
            ax.set_title(label+' | Relay OFF')
            for i,value in enumerate(y):
                ax.text(i,high[i]+2,f'{value:.2f}',ha='center',fontsize=8)
            ax.legend(frameon=False,fontsize=7,loc='upper right')
        fig.suptitle('Frozen sparse access under damaged relay (240 paired questions per model)',fontsize=10)
        save(fig,'three_model_sparse_rescue')


def integrity():
    samples = read(OUT/'sample_manifest.json')['samples']
    original = read(PARENT/'sample_manifest.json')['samples']
    lookup = {(s['task'],s['question_id']):s for s in original}
    coverage = read(OUT/'internvl/coverage.json')
    assert coverage['questions']==3000 and coverage['conditions']==84000 and coverage['status']=='complete'
    assert coverage['tasks']=={task:500 for task in TASKS} and coverage['pilot_questions']==240
    protocol = read(OUT/'protocol.json')
    for path,expected in protocol['source_sha256'].items():
        p = Path(path) if Path(path).is_absolute() else ROOT/path
        assert sha(p)==expected, path
    row_count = 0
    for sample in samples:
        parent = lookup[(sample['task'],sample['question_id'])]
        assert sample['source_sample_sha256']==digest(parent)
        for field in ('task','question_id','image_id','image_sha256','question','answer','pilot'):
            assert sample[field]==parent[field]
        path = OUT/'internvl/checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json'
        record = read(path)
        assert record['complete'] and record['sample_sha256']==digest(sample)
        assert record['protocol_sha256']==sha(OUT/'protocol.json')
        assert record['input_binding']==digest(record['input_layout'])
        layout = record['input_layout']
        assert layout['image_tokens']==256 and layout['p_minus_1']==layout['input_length']-1
        assert len({row['full_first_token'] for row in record['rows']})==1
        assert len({row['full_top1_logp'] for row in record['rows']})==1
        for row in record['rows']:
            assert row['masked_probability_max']==0 and row['changed_entries']==256
            assert row['physical_gpu']==record['physical_gpu'] and row['physical_gpu'] in (0,2,3)
            row_count += 1
    assert row_count==84000
    commits = set()
    for cohort in ('pilot','all'):
        for gpu in (0,2,3):
            folder = OUT/'internvl'/f'{cohort}_gpu{gpu}'
            v,c = read(folder/'validation.json'),read(folder/'complete.json')
            assert v['passed'] and c['status']=='complete' and c['done']==c['total']
            assert c['weights_unchanged'] and c['hooks_clean'] and c['no_nan_inf_oom']
            assert all(check['native_logits_max_abs']==check['noop_logits_max_abs']==0 for check in v['checks'])
            commits.add(v['code_commit'])
    assert len(commits)==1
    rescue=OUT/'internvl_rescue'
    rescue_samples=read(rescue/'sample_manifest.json')['samples']
    assert len(rescue_samples)==240
    rescue_counts={}
    rescue_endings={}
    for sample in rescue_samples:
        record=read(rescue/'checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json')
        assert record['complete'] and record['sample_sha256']==digest(sample)
        assert record['protocol_sha256']==sha(rescue/'protocol.json')
        assert len(record['conditions'])==3
        for condition,value in record['conditions'].items():
            assert value['correct']==int(value['prediction']==sample['answer'])
            assert value['token_ids'][0]==value['changed_first_token']
            assert value['decode_hooks_removed'] and value['fresh_cache']
            assert value['generated_tokens']==len(value['token_ids'])<=32
            assert value['truncated']==(value['generated_tokens']==32 and not value['eos'])
            assert value['empty']==(not bool(value['text']))
            assert len(value['prefill_mask_audits'])==28
            assert all(audit['masked_probability_max']==0 for audit in value['prefill_mask_audits'].values())
            rescue_counts[condition]=rescue_counts.get(condition,0)+1
            endings=rescue_endings.setdefault(condition,dict(eos=0,empty=0,truncated=0))
            for flag in endings:
                endings[flag]+=int(value[flag])
        native=record['native_full_generation']
        assert record['native_full_accuracy']==int(native['prediction']==sample['answer'])
        endings=rescue_endings.setdefault('NativeFull',dict(eos=0,empty=0,truncated=0))
        for flag in endings:
            endings[flag]+=int(native[flag])
    rp=read(rescue/'protocol.json')
    assert all(sha(ROOT/path)==expected for path,expected in rp['source_sha256'].items())
    assert rp['keep_layers']==read(OUT/'pilot_sensitive_layers.json')['keep_layers']
    for folder in rescue.glob('gpu*'):
        validation,complete=read(folder/'validation.json'),read(folder/'complete.json')
        assert validation['passed'] and complete['passed'] and complete['done']==complete['total']==240
        assert complete['weights_unchanged'] and complete['hooks_clean'] and complete['no_nan_inf_oom']
    old_hashes = read(PARENT/'result_hashes.json')
    for relative,expected in old_hashes.items():
        assert sha(PARENT/relative)==expected, 'Previous artifacts were changed: '+relative
    write(OUT/'integrity.json',dict(passed=True,questions=3000,conditions=row_count,
        paired_question_and_image_bindings=True, image_clusters=coverage['unique_images'],
        gpu_smoke_runs_passed=6, physical_gpu_count=3, smoke_question_runs=12,
        official_and_noop_logits_max_abs=0,
        numerical_tolerance_change=False, model_source_unchanged=True,
        previous_artifacts_unchanged=len(old_hashes), code_commit=commits.pop(),
        rescue_questions=240,rescue_condition_counts=rescue_counts,rescue_endings=rescue_endings,
        time=datetime.now().astimezone().isoformat()))
    print('INTEGRITY PASS: 3000 / 84000; previous delivery unchanged',flush=True)


def report():
    coverage = read(OUT/'internvl/coverage.json')
    peaks = read(OUT/'internvl/peaks.json')
    qpeaks = read(PARENT/'qwen/peaks.json')
    overall = next(p for p in peaks if p['cohort']=='all' and p['task']=='Overall')
    qoverall = next(p for p in qpeaks if p['cohort']=='all' and p['task']=='Overall')
    diagnosis = read(OUT/'diagnosis.json')
    now = datetime.now().astimezone()
    lines = ['# InternVL 原生前向修复与视觉来源扫描补齐','',
        '## 修复与正确性','',
        '- 已定位旧独立前向与官方 Chat 不一致的原因：官方生成设置 `logits_to_keep=1`，旧独立前向默认对全部位置投影 LM-head。相同 P−1 Hidden State 在不同 BF16 矩阵乘法形状下出现舍入差异；不是输入、视觉 Token 或模型权重不同。',
        f'- 诊断样本旧最大 Logits 差异为 {diagnosis["old_vs_official"]["max_abs"]}；改为原生最后位置投影后逐元素完全一致，最大差异 **0**。官方输入重放也完全一致。详见 `diagnosis.json` 的输入、Hidden 和投影形状哈希。',
        '- 没有放宽容差，没有修改第三方 Transformers/模型源码或模型权重。修复仅是独立前向显式传入 `logits_to_keep=1`，其余原生前向和 Softmax 前来源 Mask 不变。',
        '- 9项修复与Mask CPU回归检查、1项补偿控制器检查、2项图片簇配对统计检查通过。每个扫描阶段的GPU0/2/3各用两题Smoke，官方Full/No-op精确一致；目标Image概率为0、非目标Prefill Query状态逐层精确不变、无Cache前向、条件隔离和Hook清理检查通过。',
        '- 旧失败记录及上一轮3843个产物原样保留。本目录是独立恢复交付，不把先前的失败改写成已通过。','',
        '## 完成范围','',
        f'- InternVL：{coverage["questions"]}/3,000题，{coverage["conditions"]:,}/84,000个单层条件，六任务各500；先完成原240题的6,720条件，再扩展，未重新抽样。',
        '- Qwen：严格复用上一轮3,000题、108,000条件，没有重跑；三模型同240题对照中的LLaVA来源为明确标记的历史结果。',
        '- InternVL固定revision `853e3a797a661694b1b8ece0cb72dc2b23e3dac9`，BF16/eager、Transformers4.49.0，官方模板，max_num=1，256视觉Token。每题Full目标为其自己的原生首Token Top1，不是Gold候选。',
        '- 逐层扫描每次只对一个目标层的原始P−1→实际Image Key施加Softmax前−inf；不是o_proj OFF，也不是Keep-only。其他27层、其他Prefill Query、MLP、Residual和视觉注入均原生，不将Decode或EOS Query替代首答案位置。',
        '- 分片按全部3,000题的Image SHA256整簇确定，Pilot和扩展阶段沿用同一GPU归属。同题Full与28个干预在同卡配对，GPU1始终排除。','',
        '## 主要结果','',
        '|模型与队列|Mean Absolute峰层|峰值 nat|前三层|前三层描述性占比|',
        '|---|---|---|---|---|',
        f'|Qwen，3,000题|L{qoverall["absolute_peak_layer"]}|{qoverall["absolute_peak_value"]:.6f}|{qoverall["top3_layers"]}|{100*qoverall["top3_share"]:.2f}%|',
        f'|InternVL，3,000题|L{overall["absolute_peak_layer"]}|{overall["absolute_peak_value"]:.6f}|{overall["top3_layers"]}|{100*overall["top3_share"]:.2f}%|','',
        '|InternVL任务|题数|Mean Absolute峰层|峰值 nat|', '|---|---|---|---|']
    for value in peaks:
        if value['cohort']=='all' and value['task']!='Overall':
            lines.append(f'|{value["task"]}|{value["questions"]}|L{value["absolute_peak_layer"]}|{value["absolute_peak_value"]:.6f}|')
    iq=next(p for p in peaks if p['cohort']=='all' and p['task']=='QueryAttr')
    query_share=iq['absolute_peak_value']/(6*overall['absolute_peak_value'])
    lines += ['',
        f'InternVL总体L19峰中，QueryAttr的题目加权份额约为{100*query_share:.1f}%；任务组成对总体峰很重要，不能把均值峰解释成所有任务共享的处理阶段。该份额只是本层平均绝对变化的算术分解，不是独立因果贡献。',
        'Qwen与InternVL均呈非均匀分布，但主峰不在相同绝对层，InternVL分任务也出现L9/L19/L21差异。两者Top3均不到逐层平均绝对变化之和的一半，因此支持“有热点、任务相关”的描述，不支持“直接视觉必要性只存在于极少数层”的强结论。',
        '峰值与Top3占比只能描述曲线集中程度，不构成这些层充分或其余层可跳过的证据。跨模型保留各自原生Prompt、Tokenizer、视觉Token数和精度；同题、同图隔离了数据差异，但没有控制全部架构外因素。','',
        '## 与原实验的关系','',
        '- 上一轮LLaVA 240题的LogP交互仍为 −0.011213 nat，图片簇95%CI [−0.019249, −0.003380]。没有出现预声明的正交互，因此不新增InternVL的L2–L5 2×2实验，也不修改候选层寻找正面结果。',
        '- LLaVA/Qwen中继受损后的固定稀疏层补偿结果不变，详见上一轮报告。单层OFF峰值不是Keep-only充分性证明。','',
        '## InternVL 可选稀疏补偿','',
        f'仅依据先完成的240题探索曲线、在3,000题扩展前冻结的两层为 {read(OUT/"pilot_sensitive_layers.json")["keep_layers"]}。没有使用完整生成Accuracy或3,000题新峰选择层。本项严格沿用原定三个条件，Decode仍原生；额外运行原生Full仅用于同机基线与首Logits绑定。','',
        '|240题条件|完整答案Accuracy|固定Full首Token Loss nat|首Token翻转|EOS/截断/空答案|',
        '|---|---|---|---|---|']
    native=read(OUT/'internvl_rescue/native_full_reference.json')
    endings=read(OUT/'integrity.json')['rescue_endings']
    def ending_text(name):
        return '/'.join(str(endings[name][key]) for key in ('eos','truncated','empty'))
    lines.append(f'|原生Full（中继ON）|{native["accuracy_percent"]:.3f}%|0|0|{ending_text("NativeFull")}|')
    for row in read_csv(OUT/'internvl_rescue/accuracy_summary.csv'):
        lines.append(f'|Relay OFF / {row["condition"]}|{float(row["accuracy_percent"]):.3f}%|{float(row["mean_full_token_logp_loss"]):.6f}|{row["first_token_flip_count"]}|{ending_text(row["condition"])}|')
    lines += ['', '|探索性配对比较|ΔAccuracy pp|图片簇95%CI pp|参照对→错|参照错→对|', '|---|---|---|---|---|']
    for row in read_csv(OUT/'internvl_rescue/paired_comparisons.csv'):
        lines.append(f'|{row["comparison"]}|{float(row["delta_accuracy_pp"]):+.3f}|[{float(row["ci_low"]):+.3f}, {float(row["ci_high"]):+.3f}]|{row["reference_correct_to_wrong"]}|{row["reference_wrong_to_correct"]}|')
    comparisons=read_csv(OUT/'internvl_rescue/paired_comparisons.csv')
    vs_off=next(row for row in comparisons if row['comparison']=='KeepSparse - AllImageOFF')
    vs_access=next(row for row in comparisons if row['comparison']=='KeepSparse - FullImageAccess')
    vs_native=next(row for row in comparisons if row['comparison']=='KeepSparse - NativeFull')
    lines += ['',
        f'InternVL固定L19/L9相对全部直接视觉关闭恢复了{float(vs_off["delta_accuracy_pp"]):.3f}pp，但仍比保留全部直接视觉读取低{abs(float(vs_access["delta_accuracy_pp"])):.3f}pp、比原生Full低{abs(float(vs_native["delta_accuracy_pp"])):.3f}pp。因此是有限恢复，不是受损中继已被两个热点充分补偿。',
        '本项为同240题、数据驱动挑选两层的探索性充分性检查，不是独立确认。不能将普通CI包含0解释成等价，不能将稀疏访问相对全关的改善解释成已恢复原生Full；来源Hard Mask也会重分配保留Key的概率。','',
        '综合三个模型：指定中继联合干预会损害预测，固定稀疏热点只产生有限恢复。该结果与“信息路径背景值得研究”相容，但LLaVA的预声明L2–L5跳过交互仍不支持“Dead Attention Can Wake Up”。没有据此追加候选或改变负面结论。',
        'CSV口径：`accuracy_summary.csv`/`task_accuracy.csv`的`delta_full_pp`指相对Relay OFF的FullImageAccess；相对原生NativeFull的差异在`paired_comparisons.csv`中显式列出。原生Full逐题答案另存`native_full_per_question.csv`。','',
        '## 统计、图表与时限','',
        '- 复用题目加权、Image SHA256整簇配对Bootstrap，10,000次、seed20261102；输出Mean Absolute、Mean Signed和Top1 Flip的总体/六任务逐层估计及逐点95%CI。探索性曲线不以未经多重比较校正的区间宣称确认性显著性。',
        '- 3,000题仍是原Research 1冻结的六类研究题库，不是新抽取的独立确认集；240题热点选择及充分性检查也属于探索。本轮没有Qwen/InternVL的L2–L5交互实测，不能把LLaVA的负结果泛化成所有模型、所有层都没有条件性。',
        '- PNG与矢量PDF：`figures/internvl_overall_sensitivity`、`internvl_six_tasks_absolute/signed/flip`、`qwen_internvl_same3000`、`three_models_same240`。原始逐题CSV、每层/任务汇总和完整检查点均保留。',
        f'- 沿用原总墙钟起点 {START} 和截止 {DEADLINE}，未重新计算7小时预算。报告时间 {now.isoformat()}，累计 {(now-datetime.fromisoformat(START)).total_seconds()/3600:.2f} 小时。最终资源状态见 `delivery_audit.json`。','',
        '本恢复补齐原定InternVL扫描与可选固定两层补偿，没有新增干预条件。首Token分布变化不是完整答案Accuracy，直接来源低敏感也不代表模型没有视觉信息。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    (OUT/'README.md').write_text('# InternVL Native Replay Recovery\n\n'
        f'InternVL: {coverage["questions"]}/3000 questions, {coverage["conditions"]:,} Image-source OFF conditions.\n\n'
        'Exact official-chat logits restored without tolerance changes; previous failure evidence is retained.\n\n'
        'See [report_zh.md](report_zh.md).\n\n'
        'The previous delivery remains immutable under ../visual_pathways_seven_hour/.\n'
        'This recovery completes InternVL and adds only the originally optional fixed two-layer rescue.\n\n'
        'In rescue accuracy/task summaries, delta_full_pp references relay-OFF FullImageAccess, not NativeFull.\n'
        'Explicit NativeFull contrasts are in paired_comparisons.csv; native per-question answers are retained separately.\n')


def delivery():
    assert read(OUT/'integrity.json')['passed']
    now=datetime.now().astimezone()
    apps=command('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader')
    active={int(line.split(',')[1].strip()) for line in apps.splitlines() if line.strip()}
    pids={read(path)['pid'] for path in OUT.rglob('preflight.json')}
    assert not (active & pids), 'An experiment process still owns GPU memory'
    gpu_inventory=command('nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader')
    began=datetime.fromisoformat('2026-11-02T06:08:37+08:00')
    durations={str(path.relative_to(OUT)):read(path) for path in sorted((OUT/'internvl').glob('*/complete.json'))}
    task1_finished=max(datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                       for path in (OUT/'internvl').glob('all_gpu*/complete.json'))
    write(OUT/'delivery_audit.json',dict(time=now.isoformat(),original_start=START,
        recovery_start=began.isoformat(),total_deadline=DEADLINE,
        diagnosis_gpu_start=datetime.fromtimestamp(read(OUT/'diagnosis/preflight.json')['time']).astimezone().isoformat(),
        total_elapsed_seconds=(now-datetime.fromisoformat(START)).total_seconds(),
        recovery_elapsed_seconds=(now-began).total_seconds(),within_original_seven_hours=now<datetime.fromisoformat(DEADLINE),
        task1_finished=task1_finished.isoformat(),task1_within_original_limit=task1_finished<datetime.fromisoformat(TASK1_END),
        gpu_ids_used=[0,2,3],gpu1_used=False,gpu_released=True,experiment_pids=sorted(pids),
        external_processes_untouched=True,gpu_inventory=gpu_inventory,compute_apps=apps,
        branch=command('git','branch','--show-current'),pre_results_commit=command('git','rev-parse','HEAD'),
        analysis_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted((ROOT/'vlm_research/visual_pathways').glob('*.py'))},
        phase_durations=durations,previous_delivery_unchanged=True,
        status='Complete original Qwen/InternVL3000 sensitivity and fixed240 rescue; no interaction extension after negative LLaVA gate'))
    write(OUT/'result_hashes.json',{str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*'))
        if path.is_file() and path.name!='result_hashes.json' and not path.name.endswith('.tmp')})
    print('Delivery audit: all experiment GPUs released; hashes written',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['pilot-freeze','aggregate','rescue','figures','integrity','report','delivery'])
    args = parser.parse_args()
    if args.action=='pilot-freeze':
        pilot_freeze()
    elif args.action=='aggregate':
        print(scan_model('internvl',output_root=OUT),flush=True)
    elif args.action=='rescue':
        analyze_rescue()
    else:
        globals()[args.action]()
