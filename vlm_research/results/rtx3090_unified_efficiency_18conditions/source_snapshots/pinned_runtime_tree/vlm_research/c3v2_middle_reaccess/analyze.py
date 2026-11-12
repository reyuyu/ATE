"""CPU-only analysis of complete, hash-verified quality and timing checkpoints."""
import collections
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import *
from .run import verify_freeze
from ..fastp2_bypass_writer.common import hardware


def analyze():
    freeze=verify_freeze()
    samples=read(OUT/'sample_manifest.json'); partition=read(OUT/'partition.json')
    assert all(read(OUT/f'worker{r}/completion.json')['status']=='COMPLETE' for r in range(2))
    assert read(OUT/'driver_exit.json')['returncodes']==[0,0]
    resource=hardware()
    owned=set(read(OUT/'driver_pids.json')['workers'])
    active={int(line.split(',')[1].strip()) for line in resource['processes'].splitlines() if line.strip()}
    assert not owned.intersection(active),'Task worker still has a compute process'
    releases=[read(OUT/f'worker{r}/release.json') for r in range(2)]
    assert all(not Path('/proc',str(pid)).exists() for pid in owned),'Worker process has not exited'
    atomic(OUT/'gpu_release_audit.json',dict(status='RELEASED',at=now(),workers=releases,
        worker_processes_exited=True,task_pids_absent_from_nvidia_smi=True,
        note='Pre-exit allocator snapshots retain32MiB. Driver confirmed both processes exited; nvidia-smi is the final release check.',**resource))
    keys=sorted(samples); matrix=np.zeros((500,4)); records={}; rows=[]
    question_shards=collections.defaultdict(set)
    checkpoint_hashes={}
    for i,key in enumerate(keys):
        s=samples[key]
        for j,label in enumerate(LABELS):
            path=quality_path(key,label); v=load(path); g=v['generation']
            assert v['sample_sha256']==digest(s) and v['sources']==freeze['sources']
            assert v['score']==score(s,g['text']) and g['vision_calls']==1
            assert g['kv']['retained_image_layers']==list(KEEP[label])
            assert len(g['token_ids'])<=32 and g['first_top1_token']==g['token_ids'][0]
            n=len(s['input_mapping']['prompt_ids']); text_n=n-576
            image_positions=set(s['input_mapping']['image'])
            text_positions=[q for q in range(n) if q not in image_positions]
            for kvrow in g['kv']['layers']:
                layer=kvrow['layer']; absent=label!='Native' and layer in (2,3,4,5)
                prefill_n=n if label=='Native' or layer in KEEP[label] else text_n
                expected=0 if absent else prefill_n+len(g['token_ids'])-1
                assert kvrow['used_tokens']==expected
                expected_capacity=expected if label=='Native' else (0 if absent else prefill_n+31)
                assert kvrow['capacity_tokens']==expected_capacity
                if 'original_positions_sha256' in kvrow:
                    positions=[] if absent else (list(range(n)) if layer in KEEP[label] else text_positions.copy())
                    if not absent: positions+=list(range(n,n+len(g['token_ids'])-1))
                    assert digest(positions)==kvrow['original_positions_sha256']
            rank=partition['image_shard'][s['image_sha256']]
            assert v['rank']==rank
            question_shards[s['image_sha256']].add(rank)
            if label=='C3-V2':
                assert v['history']['exact_tokens'] and v['history']['exact_first_logits']
            matrix[i,j]=v['score']; records[key,label]=v
            checkpoint_hashes[str(path.relative_to(OUT))]=sha(path)
            rows.append(dict(key=key,question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                rank=rank,question=s['question'],gold_answers=s['answer'],condition=label,
                answer=g['text'],score=v['score'],token_ids=g['token_ids'],first_top1_token=g['first_top1_token'],
                eos=g['eos'],truncated=g['truncated'],empty=g['empty'],repeated=g['repeated'],
                content_tokens=g['content_tokens'],decode_steps=g['decode_steps'],
                first_logits_sha256=g['first_logits_sha256'],input_sha256=digest(v['input_binding']),
                retained_image_layers=g['kv']['retained_image_layers']))
    assert len(rows)==2000 and all(len(v)==1 for v in question_shards.values()) and len(question_shards)==309
    write_csv(OUT/'per_question.csv',rows)
    cluster_names=sorted(question_shards); idx={h:i for i,h in enumerate(cluster_names)}
    membership=np.array([idx[samples[k]['image_sha256']] for k in keys])
    sizes=np.bincount(membership)
    sums=np.zeros((309,4)); np.add.at(sums,membership,matrix)
    rng=np.random.default_rng(SEED)
    weights=rng.multinomial(309,np.full(309,1/309),size=20000)
    boot=(weights@sums)/(weights@sizes)[:,None]*100
    means=matrix.mean(0)*100
    summaries=[]
    for j,label in enumerate(LABELS):
        own=[v['generation'] for (k,l),v in records.items() if l==label]
        lo,hi=np.percentile(boot[:,j],[2.5,97.5])
        summaries.append(dict(condition=label,n=500,image_clusters=309,accuracy=means[j],ci_low=lo,ci_high=hi,
            delta_native_pp=means[j]-means[0],delta_c3_pp=means[j]-means[1],
            eos=sum(g['eos'] for g in own),truncated=sum(g['truncated'] for g in own),
            empty=sum(g['empty'] for g in own),repeated=sum(g['repeated'] for g in own)))
    write_csv(OUT/'accuracy_summary.csv',summaries)
    comparisons=[]; cases=[]
    for j,b in ((2,1),(3,1),(2,0),(3,0),(3,2),(1,0)):
        delta=matrix[:,j]-matrix[:,b]; low,high=np.percentile(boot[:,j]-boot[:,b],[2.5,97.5])
        flips=sum(records[k,LABELS[j]]['generation']['first_top1_token']!=records[k,LABELS[b]]['generation']['first_top1_token'] for k in keys)
        changed=sum(records[k,LABELS[j]]['generation']['token_ids']!=records[k,LABELS[b]]['generation']['token_ids'] for k in keys)
        comparisons.append(dict(comparison=LABELS[j]+' - '+LABELS[b],delta_accuracy_pp=100*delta.mean(),
            ci_low=low,ci_high=high,score_increased=int((delta>1e-12).sum()),score_decreased=int((delta< -1e-12).sum()),
            score_equal=int((np.abs(delta)<1e-12).sum()),full_score_to_lower=int(((matrix[:,b]>=1-1e-12)&(delta< -1e-12)).sum()),
            lower_to_full_score=int(((matrix[:,j]>=1-1e-12)&(delta>1e-12)).sum()),
            first_token_flips=flips,generated_sequence_changes=changed,bootstrap_replicates=20000))
        for i,k in enumerate(keys):
            if abs(delta[i])<1e-12: continue
            cases.append(dict(comparison=LABELS[j]+' - '+LABELS[b],key=k,question=samples[k]['question'],
                image_sha256=samples[k]['image_sha256'],gold_answers=samples[k]['answer'],delta_score=delta[i],
                baseline_answer=records[k,LABELS[b]]['generation']['text'],
                intervention_answer=records[k,LABELS[j]]['generation']['text'],
                baseline_score=matrix[i,b],intervention_score=matrix[i,j]))
    write_csv(OUT/'paired_comparisons.csv',comparisons)
    write_csv(OUT/'negative_and_rescue_cases.csv',cases)
    perf=[]; perf_records={}
    for key in partition['performance48']:
        for mode in ('natural_eos','fixed16'):
            for repeat in range(-3,5):
                for label in LABELS:
                    path=performance_path(key,mode,repeat,label); v=load(path); g=v['generation']
                    assert v['sources']==freeze['sources'] and v['sample_sha256']==digest(samples[key])
                    assert v['rank']==partition['image_shard'][samples[key]['image_sha256']]
                    assert g['vision_calls']==1 and g['kv']['retained_image_layers']==list(KEEP[label])
                    if mode=='natural_eos': assert g['token_ids']==records[key,label]['generation']['token_ids']
                    else: assert len(g['token_ids'])==16
                    checkpoint_hashes[str(path.relative_to(OUT))]=sha(path)
                    if repeat<0: continue
                    kv=g['kv']; mib=2**20
                    row=dict(key=key,condition=label,mode=mode,repeat=repeat,rank=v['rank'],
                        prefill_ms=g['prefill_ms'],language_prefill_ms=g['language_prefill_ms'],vision_ms=g['vision_ms'],
                        ttft_ms=g['ttft_ms'],e2e_ms=g['e2e_ms'],decode_ms=g['decode_ms'],decode_steps=g['decode_steps'],
                        content_tokens=g['content_tokens'],generated_tokens=len(g['token_ids']),eos=g['eos'],
                        live_kv_mib=kv['live_bytes']/mib,capacity_kv_mib=kv['storage_bytes']/mib,
                        image_live_mib=kv['image_live_bytes']/mib,text_live_mib=kv['text_live_bytes']/mib,
                        image_capacity_mib=kv['image_capacity_bytes']/mib,text_capacity_mib=kv['text_capacity_bytes']/mib,
                        peak_allocated_mib=g['peak_allocated']/mib,peak_reserved_mib=g['peak_reserved']/mib,
                        retained_image_layers=kv['retained_image_layers'])
                    perf.append(row); perf_records[key,mode,repeat,label]=row
    assert len(perf)==48*2*5*4
    write_csv(OUT/'latency_raw.csv',perf)
    metrics=[k for k,v in perf[0].items() if isinstance(v,(int,float)) and k not in ('rank','repeat','eos')]
    timing=[]
    for mode in ('natural_eos','fixed16'):
        for label in LABELS:
            own=[r for r in perf if r['mode']==mode and r['condition']==label]
            r=dict(condition=label,mode=mode,questions=48,repeats=5,requests=len(own),
                **{m:float(np.mean([q[m] for q in own])) for m in metrics})
            r['decode_ms_per_token']=sum(q['decode_ms'] for q in own)/sum(q['decode_steps'] for q in own)
            r['retained_image_layers']=list(KEEP[label])
            same=[k for k in partition['performance48'] if records[k,label]['generation']['token_ids']==records[k,'Native']['generation']['token_ids']]
            r['native_identical_sequence_questions']=len(same)
            subset=[q for q in own if q['key'] in same]
            reference=[q for q in perf if q['mode']==mode and q['condition']=='Native' and q['key'] in same]
            r['identical_sequence_speedup']=float(np.mean([q['e2e_ms'] for q in reference])/np.mean([q['e2e_ms'] for q in subset])) if same else None
            timing.append(r)
    for r in timing:
        native=next(q for q in timing if q['mode']==r['mode'] and q['condition']=='Native')
        c3=next(q for q in timing if q['mode']==r['mode'] and q['condition']=='C3-V2')
        r['e2e_speedup_native']=native['e2e_ms']/r['e2e_ms']
        r['e2e_overhead_c3_pct']=(r['e2e_ms']/c3['e2e_ms']-1)*100
        r['prefill_overhead_c3_pct']=(r['prefill_ms']/c3['prefill_ms']-1)*100
    write_csv(OUT/'latency_summary.csv',timing)
    kvrows=[]
    for label in LABELS:
        example=records[keys[0],label]
        for row in example['generation']['kv']['layers']:
            kvrows.append(dict(condition=label,example_key=keys[0],**row))
    write_csv(OUT/'physical_kv_layer_audit.csv',kvrows)
    make_plot(summaries,timing)
    atomic(OUT/'result_hashes.json',checkpoint_hashes)
    audit=dict(status='PASS',at=now(),quality_records=2000,performance_records=1920,warmup_records=1152,
        questions=500,image_clusters=309,performance_images=48,duplicate_shards=False,
        all_checkpoints_hash_verified=True,source_freeze_verified=True,C3_history_tokens_logits_exact=500,
        quality_KV_layer_records_verified=64000,physical_lengths_capacities_original_position_hashes_verified=True,
        native_and_noop_strict=True,CPU_tests=12,GPU_smoke_cases_per_card=3,
        physical_image_KV_layers={k:list(v) for k,v in KEEP.items()},
        dense_reference='Validation only; excluded from performance',
        bootstrap='20000 image-SHA cluster paired resamples, equal question weighting, seed20261105',
        interpretation='Exploratory repeated historical500; no independent non-inferiority claim',
        model_weights_sha256=freeze['weights'],sources=freeze['sources'])
    atomic(OUT/'validation.json',audit)
    atomic(OUT/'completion.json',dict(status='COMPLETE',at=now(),quality_generations=2000,
        performance_formal_requests=1920,performance_warmup_requests=1152,code_commit=freeze['code_commit'],
        checkpoint_hash_manifest_sha256=sha(OUT/'result_hashes.json')))
    report(summaries,comparisons,timing)
    print(json.dumps(dict(accuracy=summaries,comparisons=comparisons,latency=timing),ensure_ascii=False,indent=2))


def make_plot(summary,timing):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    colors=['#51565c','#008c8c','#cc763a','#b84768']
    names=['Native','C3-V2','Middle9','Middle10']
    x=np.arange(4); y=np.array([r['accuracy'] for r in summary])
    errors=np.array([[r['accuracy']-r['ci_low'] for r in summary],[r['ci_high']-r['accuracy'] for r in summary]])
    axes[0].bar(x,y,color=colors,width=.65)
    axes[0].errorbar(x,y,yerr=errors,fmt='none',color='black',capsize=3,lw=1)
    axes[0].set(xticks=x,xticklabels=names,ylabel='TextVQA soft accuracy (%)',title='Quality: frozen 500 questions')
    for i,v in enumerate(y): axes[0].text(i,summary[i]['ci_high']+.5,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    for mode,marker in (('natural_eos','o'),('fixed16','s')):
        pts=[next(r for r in timing if r['condition']==c and r['mode']==mode) for c in LABELS]
        for i,r in enumerate(pts):
            axes[1].scatter(r['e2e_speedup_native'],y[i],color=colors[i],marker=marker,s=48,
                label=names[i] if mode=='natural_eos' else None)
            axes[1].annotate(names[i] if mode=='natural_eos' else '',(r['e2e_speedup_native'],y[i]),
                xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].axvline(1,color='#777777',ls='--',lw=.8)
    axes[1].set(xlabel='Actual E2E speedup vs Native',ylabel='500-question soft accuracy (%)',
        title='Speed: 48 paired inputs, 5 repeats')
    axes[1].text(.02,.02,'Circle: natural EOS | Square: fixed 16-token timing\nQuality always uses natural-EOS answers',transform=axes[1].transAxes,fontsize=7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    for extension in ('png','pdf'): fig.savefig(OUT/('accuracy_speedup.'+extension),dpi=240,bbox_inches='tight')
    plt.close(fig)


def report(summary,comparisons,timing):
    table=['|条件|Accuracy|ΔNative pp|ΔC3 pp|','|---|---:|---:|---:|']
    table += [f"|{r['condition']}|{r['accuracy']:.2f}%|{r['delta_native_pp']:+.2f}|{r['delta_c3_pp']:+.2f}|" for r in summary]
    paired=['|配对比较|差值 pp|图片簇95%CI|软分上升/下降|','|---|---:|---|---:|']
    paired += [f"|{r['comparison']}|{r['delta_accuracy_pp']:+.2f}|[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]|{r['score_increased']}/{r['score_decreased']}|" for r in comparisons]
    perf=['|条件/模式|Prefill ms|TTFT ms|E2E ms|Decode ms/token|KV Live/Capacity MiB|Peak Allocated MiB|加速比|相对C3开销|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    perf += [f"|{r['condition']} / {r['mode']}|{r['prefill_ms']:.2f}|{r['ttft_ms']:.2f}|{r['e2e_ms']:.2f}|{r['decode_ms_per_token']:.2f}|{r['live_kv_mib']:.2f}/{r['capacity_kv_mib']:.2f}|{r['peak_allocated_mib']:.1f}|{r['e2e_speedup_native']:.3f}x|{r['e2e_overhead_c3_pct']:+.2f}%|" for r in timing]
    answers=[]
    for label in LABELS[2:]:
        s=next(r for r in summary if r['condition']==label)
        t=next(r for r in timing if r['condition']==label and r['mode']=='natural_eos')
        f=next(r for r in timing if r['condition']==label and r['mode']=='fixed16')
        answers.append(f"{label}相对C3恢复{s['delta_c3_pp']:+.2f}pp，与Native差{s['delta_native_pp']:+.2f}pp；自然EOS E2E相对C3开销{t['e2e_overhead_c3_pct']:+.2f}%，相对Native为{t['e2e_speedup_native']:.3f}x；固定16 Token为{f['e2e_speedup_native']:.3f}x。")
    resource=read(OUT/'gpu_release_audit.json') if (OUT/'gpu_release_audit.json').exists() else {'status':'See worker release records'}
    length_note='自然EOS平均生成Token数（含EOS）：'+ '，'.join(
        f"{r['condition']} {r['generated_tokens']:.2f}" for r in timing if r['mode']=='natural_eos')+'。'
    tm={(r['condition'],r['mode']):r for r in timing}
    length_note+='Middle10自然EOS略快于C3，但输出也更短，不能据此称其执行路径更省计算。'
    for label in LABELS[2:]:
        fixed=tm[label,'fixed16']; natural=tm[label,'natural_eos']
        length_note+=f"{label}固定16 Token相对C3 E2E开销{fixed['e2e_overhead_c3_pct']:+.2f}%，相对Native加速{fixed['e2e_speedup_native']:.3f}x；同完整自然序列子集为{natural['identical_sequence_speedup']:.3f}x（{natural['native_identical_sequence_questions']}题）。"
    length_note+='各条件的同序列子集不同，不可直接把子集加速比横向比较。'
    text='\n'.join(['# C3-V2 + Middle9/Middle10视觉回看','',*answers,'',
        '![Accuracy and actual E2E speedup](accuracy_speedup.png)','',
        '## 完整500题质量','',*table,'',*paired,'',
        '共500题、309个图片SHA簇，官方十答案共识软评分；分数上升/下降不等同二值正确/错误。四条件均重新生成；500题C3与冻结历史的完整Token和首Token Logits哈希全部一致。',
        '95%CI来自20,000次按图片SHA整簇配对Bootstrap，问题等权；本轮是重复使用历史样本的探索分析，CI跨0不能解释成等效或无损。','',
        '## 真实物理性能','',*perf,'',
        '测速48张不同图片在读取输出前按输入长度四分位与哈希冻结，两卡各24张；同题全部条件在同一卡。每模式每条件3次预热、5次交错重复，冷KV且每请求重算Vision。表格为等输入平均，Decode为总Decode时间/总Decode步数；速度比为Native均值/候选均值。',
        'Prefill是CUDA事件记录的Vision+Language首次前向；TTFT是请求开始至首Token就绪的墙钟时间。E2E包括控制器/Mask/KV准备及自然EOS，排除评分、Token转字符串和输出审计。固定16 Token使用自身预测前缀，忽略EOS仅用于长度控制，不计入质量分。',
        'Peak Allocated是重置统计后的实际峰值（含模型）；Reserved受PyTorch分配器跨请求复用影响，不能当作单独KV容量。重复标记是三Token子串重复的启发式，不自动认定为无效答案。',
        '全部计时均来自实际请求；Dense参考不参与速度排名。Peak Reserved、同完整序列子集和各层Image/Text KV容量见latency_summary.csv及physical_kv_layer_audit.csv。','',
        length_note,'',
        '## 物理路径与正确性','',
        'Native保持32层原生Image KV；C3无持久Image KV。Middle9仅在L11–L19持久保留原Image KV，进入L20前删除视觉Hidden；Middle10仅保留L11–L20的Image KV，进入L21前删除视觉Hidden。其余层缓存仅含文本。L2–L5物理省掉P−1/Decode Attention，MLP/Norm/Residual完整保留，不分配未来无用KV。',
        'R→Image始终只在L0–L19开放；Middle10的L20仅允许P−1/Decode读图，R仍被Softmax前-inf屏蔽。所有逻辑位置与RoPE保持原始编号，KV原地追加，无历史KV复制或拼接，无跨条件复用。',
        '12项CPU测试、每卡3题GPU Smoke通过：Native/No-op与原C3同形状H1 Logits严格torch.equal；与Dense功能参考按历史固定FP16门槛比较，并要求完整Greedy Token一致。P−1和真实Decode的选定层实际Image Attention均为正，其他层为0；Middle10 L20 R→Image概率严格为0；完整MLP、物理行数、独立Cache及Hook清理通过。',
        '阈值未放宽：logit_atol=0.10、rtol=0.003、Hidden relative RMSE≤0.01、cosine distance≤1e-4。详细逐项结果见worker*/smoke.json。','',
        '开发期CPU测试曾发现审计函数试图读取已被物理省略的P−1 Attention行，已修正为显式识别该行不存在；正式冻结前12项回归全部通过，没有改变计算语义或数值阈值。','',
        '首次CPU汇总时，资源检查错误地要求进程退出前分配器统计必须为0，因两卡各剩32MiB而中止。保留原始失败日志；资源释放改用实际进程退出、任务PID不在nvidia-smi及退出后显存状态判定，GPU结果与数值门槛未修改、未重跑。','',
        '## 异常与复现','',
        '所有EOS、截断、空答案、重复和负面样本完整保留，见accuracy_summary.csv、per_question.csv及negative_and_rescue_cases.csv。资源状态：'+resource['status']+'；任务进程已退出，详细显存和PID记录见gpu_release_audit.json。',
        '源码冻结及模型/数据/依赖绑定见execution_freeze.json、protocol.json与worker*/environment.json；逐记录SHA见result_hashes.json。','',
        'GitHub使用raw_checkpoints.tar.gz原样归档全部5,072份质量/计时断点（2,000质量＋1,920正式计时＋1,152预热），避免数百万行重复JSON影响审阅。CSV保持直接可读；归档逐文件SHA与result_hashes.json一致，打包审计见packaging.json。本地断点保留不变。','',
        '```bash','/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.c3v2_middle_reaccess.tests',
        'tar -xzf vlm_research/results/c3v2_middle_visual_reaccess/raw_checkpoints.tar.gz -C vlm_research/results/c3v2_middle_visual_reaccess',
        '/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.c3v2_middle_reaccess.run --driver',
        '/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.c3v2_middle_reaccess.analyze','```','',
        '已有冻结与Checkpoint不得覆盖；同一源码可安全断点续跑。完成本轮即停止，不进行其他条件搜索。',''])
    (OUT/'report_zh.md').write_text(text)


if __name__=='__main__': analyze()
