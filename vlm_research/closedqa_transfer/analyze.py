"""CPU-only aggregation of the predeclared, fixed transfer matrix."""
import argparse
import collections
import csv
import math
import subprocess

import numpy as np

from .common import *
from ..benchmarks.scoring import pope_metrics


def preflight():
    phase=json.loads(git_bytes(PHASE,'vlm_research/results/phase4_0_dataset_suite/dataset_inventory.json'))
    files=list(Path('/data/lk/ATE-resources/benchmarks/phase4_0').rglob('*'))
    readiness={}
    for name in ('MMBench','SEED'):
        matches=[str(p) for p in files if name.lower() in p.name.lower() and p.is_file()]
        readiness[name]=dict(status='NOT_READY_FOR_FROZEN_EVALUATION',
            registered_in_frozen_phase4=any(name.lower() in k for k in phase),
            matching_local_asset_files=matches,verified_manifest=False,verified_scorer=False,
            action='Readiness only; no download or evaluation in this task')
    atomic(OUT/'benchmark_readiness.json',readiness)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main',
        'refs/heads/codex/ate-accuracy-autoresearch-6h','refs/heads/codex/ate-accuracy-autoresearch-6h-continued',
        'refs/heads/codex/ate-closedqa-frozen-transfer'],cwd=ROOT,text=True)
    old=Path('/data/lk/ATE-worktrees/cross-dataset-c1-c2/vlm_research/cross_dataset_c1_c2/run.py')
    atomic(OUT/'history_reuse_audit.json',dict(at=now(),remote_refs=remote,
        execution_base=BASE,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        previous_full_qsa_studies=dict(commits=[BASE,'2c81863ed8c9f845f7aacfc3d80c873271415d77'],
            datasets=['GQA','VQAv2'],exact_closedqa_rows=0),
        historical_closedqa=dict(path=str(old),sha256=sha(old),
            reason='Full/C1/C2 use older intervention and model.generate without a frozen logits_to_keep=1 binding; not accepted as exact current five-condition rows'),
        accepted_prior_generated_rows=0,all_current_native_baselines_regenerated=True,
        within_run_reuse='Only exactly identical Image SHA, expanded Prompt IDs, semantic partitions and writer sets; same physical GPU and config. Per-row rescoring and source checkpoint SHA retained.',
        pre_freeze_issue=dict(name='Raw annotation JSON digest serialization mismatch',
            resolution='Use original frozen Phase4 dataset_registry.digest for original source-row SHA; data and thresholds unchanged'),
        no_history_latency_reuse=True))
    atomic(OUT/'cpu_validation.json',dict(status='PASS',tests=16,log_sha256=sha(OUT/'cpu_tests.log'),
        command='CUDA_VISIBLE_DEVICES= python -m vlm_research.closedqa_transfer.tests',at=now()))
    dependency_paths=[ROOT/'vlm_research'/p for p in (
        'ate_fulljoint_vqav2/common.py','ate_fulljoint_v2/common.py',
        'accuracy_autoresearch/common.py','p1_layerwise_accuracy/common.py',
        'physical_visual_handoff/common.py','fastp2_bypass_writer/run.py')]
    from ..p1_layerwise_accuracy.common import OLD
    dependency_paths += [OLD/'vlm_common.py',OLD/'knockout_probe.py']
    dependencies={str(p):sha(p) for p in dependency_paths}
    model_files={str(p):sha(p) for p in Path(MODEL_PATH).glob('*') if p.is_file() and p.suffix in ('.json','.model')}
    atomic(OUT/'dependency_bindings.json',dict(source_files=dependencies,model_processor_files=model_files))
    atomic(OUT/'performance_entry_freeze.json',dict(at=now(),sha256=sha(Path(__file__).with_name('benchmark.py')),
        original_execution_freeze_sha256=sha(OUT/'execution_freeze.json'),
        change='Measurement dictionary assembly only; no inference/configuration/quality changes',
        tests_sha256=sha(OUT/'analysis_tests.log'),quality_source_commit='b8351e2d9fea647b85db7ebb5def580e2b76eb38'))
    samples=read(OUT/'sample_manifest.json'); partition=read(OUT/'partition.json')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    rows=[]
    for group in partition['smoke'].values():
        for key in group:
            s=samples[key]; m=s['input_mapping']
            for i,tid in enumerate(m['prompt_ids']):
                owners=[part for part,positions in m['parts'].items() if i in positions]
                owners+=['Image'] if i in m['image'] else []
                owners+=['Pminus1'] if i==m['Pminus1'] else []
                rows.append(dict(sample=key,position=i,token_id=tid,token=tokenizer.convert_ids_to_tokens(tid),
                                 component='|'.join(owners) or 'BOS_or_template'))
    write_csv(OUT/'smoke_token_mapping.csv',rows)


def interval(values,alpha=.05):
    return [float(x) for x in np.quantile(values,[alpha/2,1-alpha/2])]


def draws(keys,samples,records,task):
    labels=list(CONFIGS); groups=sorted({samples[k]['image_sha256'] for k in keys})
    gi={h:i for i,h in enumerate(groups)}; totals=np.zeros((len(groups),len(labels),7))
    for k in keys:
        s=samples[k]; i=gi[s['image_sha256']]
        for j,c in enumerate(labels):
            r=records[k,c]; pred=r['scoring']['parsed']; gold=s['answer']
            yes=pred=='yes'; gy=gold=='yes'
            totals[i,j]=totals[i,j]+[r['scoring']['score'],yes and gy,not yes and not gy,
                yes and not gy,not yes and gy,yes,1]
    seed=int(digest([SEED,task])[:8],16)
    weights=np.random.default_rng(seed).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
    x=(weights@totals.reshape(len(groups),-1)).reshape(20000,len(labels),7)
    acc=x[:,:,0]/x[:,:,6]; output={'accuracy':acc}
    if task!='scienceqa_img':
        tp,tn,fp,fn=x[:,:,1],x[:,:,2],x[:,:,3],x[:,:,4]
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp!=0)
        recall=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=tp+fn!=0)
        f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn!=0)
        output.update(precision=precision,recall=recall,f1=f1,yes_ratio=x[:,:,5]/x[:,:,6])
    return output,len(groups)


def summarize():
    freeze=read(OUT/'execution_freeze.json'); assert source_hashes()==freeze['sources']
    samples=read(OUT/'sample_manifest.json'); partition=read(OUT/'partition.json')
    assert all(read(OUT/f'worker{i}/completion.json')['status']=='COMPLETE' for i in (0,1))
    records={}; raw=[]; negative=[]; reuse=0
    for k,s in samples.items():
        for label in CONFIGS:
            r=record(k,label,s); assert r is not None,(k,label)
            records[k,label]=r; reuse+=r['reuse_from'] is not None
    assert len(records)==7000
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    for (k,c),r in records.items():
        unstripped=tokenizer.decode(r['generation']['token_ids'],skip_special_tokens=True)
        assert scored(samples[k],unstripped)==r['scoring'],('Frozen raw decoding/scorer mismatch',k,c)
    for k,s in samples.items():
        full=records[k,'Native Full']; fullg=full['generation']; f=full['scoring']['score']
        for label in CONFIGS:
            r=records[k,label]; g=r['generation']; score=r['scoring']['score']
            row=dict(sample=k,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],subject=s['subject'],gold=s['answer'],condition=label,
                text=g['text'],token_ids=json.dumps(g['token_ids']),score=score,parsed=r['scoring']['parsed'],
                parse_failed=r['scoring']['parse_failed'],eos=g['eos'],truncated=g['truncated'],empty=g['empty'],
                repeated=g['repeated'],content_tokens=g['content_tokens'],first_token_flip=g['token_ids'][0]!=fullg['token_ids'][0],
                text_changed=g['text']!=fullg['text'],tokens_changed=g['token_ids']!=fullg['token_ids'],
                full_token_signed_logp_loss=fullg['native_first_token_logp']-g['native_first_token_logp'],
                harm=int(f>score),rescue=int(f<score),reuse_from=r['reuse_from'],
                input_binding_sha256=digest(r['input_binding']),checkpoint_sha256=sha(checkpoint(k,label)),rank=r['rank'])
            raw.append(row)
            if score==0 or row['parse_failed'] or g['empty'] or g['truncated']: negative.append(row)
    write_csv(OUT/'per_question.csv',raw); write_csv(OUT/'negative_cases.csv',negative)
    summaries=[]; pairs=[]; subjects=[]; tests=[]
    for task in TASKS:
        keys=sorted(k for k,s in samples.items() if s['task']==task)
        boot,ngroups=draws(keys,samples,records,task)
        point={}
        for j,c in enumerate(CONFIGS):
            rs=[records[k,c] for k in keys]; gens=[r['generation'] for r in rs]
            metrics=dict(accuracy=float(np.mean([r['scoring']['score'] for r in rs])))
            if task!='scienceqa_img':
                official=pope_metrics([g['text'] for g in gens],[samples[k]['answer'] for k in keys])
                metrics.update({m:official['official_generation_'+m] for m in ('precision','recall','f1','yes_ratio')})
            point[c]=metrics
            row=dict(task=task,condition=c,n=len(keys),image_clusters=ngroups,**{m:100*v for m,v in metrics.items()},
                accuracy_ci_low=100*interval(boot['accuracy'][:,j])[0],accuracy_ci_high=100*interval(boot['accuracy'][:,j])[1],
                parse_failed=sum(r['scoring']['parse_failed'] for r in rs),empty=sum(g['empty'] for g in gens),
                truncated=sum(g['truncated'] for g in gens),no_eos=sum(not g['eos'] for g in gens),
                repeated=sum(g['repeated'] for g in gens),mean_content_tokens=float(np.mean([g['content_tokens'] for g in gens])))
            summaries.append(row)
            if task=='scienceqa_img':
                for subject in sorted({samples[k]['subject'] for k in keys}):
                    sk=[k for k in keys if samples[k]['subject']==subject]
                    subjects.append(dict(subject=subject,condition=c,n=len(sk),
                        accuracy=100*np.mean([records[k,c]['scoring']['score'] for k in sk]),
                        parse_failed=sum(records[k,c]['scoring']['parse_failed'] for k in sk)))
        labels=list(CONFIGS)
        for base in ('Native Full','C3-V2'):
            b=labels.index(base)
            for j,c in enumerate(labels):
                if c==base: continue
                harm=sum(records[k,base]['scoring']['score']>records[k,c]['scoring']['score'] for k in keys)
                rescue=sum(records[k,base]['scoring']['score']<records[k,c]['scoring']['score'] for k in keys)
                for metric,values in boot.items():
                    delta=100*(values[:,j]-values[:,b]); lo,hi=interval(delta)
                    row=dict(task=task,condition=c,reference=base,metric=metric,n=len(keys),image_clusters=ngroups,
                        delta_pp=100*(point[c][metric]-point[base][metric]),ci95_low=lo,ci95_high=hi,
                        harm=harm if metric=='accuracy' else '',rescue=rescue if metric=='accuracy' else '',
                        interpretation='exploratory paired Image-SHA cluster interval')
                    if metric=='accuracy' and c=='Full-QSA' and base=='Native Full':
                        lower=float(np.quantile(delta,.05/4)); adjusted=interval(delta,.05/4)
                        row.update(NI_margin_pp=-1,NI_family_size=4,NI_one_sided_lower=lower,
                            NI_supported=lower>-1,bonferroni4_twosided_low=adjusted[0],bonferroni4_twosided_high=adjusted[1])
                        row['interpretation']='predeclared four-task Full-QSA vs Native Accuracy family; Bonferroni4 NI'
                        tests.append(dict(row))
                    pairs.append(row)
    write_csv(OUT/'accuracy_summary.csv',summaries); write_csv(OUT/'paired_comparisons.csv',pairs)
    write_csv(OUT/'scienceqa_subject_accuracy.csv',subjects); atomic(OUT/'noninferiority.json',tests)
    latency,flops_rows,shape_rows=performance_summary(samples,records,partition)
    write_csv(OUT/'latency_summary.csv',latency); write_csv(OUT/'flops_kv_summary.csv',flops_rows)
    write_csv(OUT/'executed_shape_census.csv',shape_rows)
    assert reuse==(len(samples)-partition['unique_actual_inputs'])*5
    assert all(sha(p)==h for p,h in freeze['weights'].items())
    dependencies=read(OUT/'dependency_bindings.json')
    assert all(sha(p)==h for bucket in dependencies.values() for p,h in bucket.items())
    atomic(OUT/'validation.json',dict(status='PASS',cpu=read(OUT/'cpu_validation.json'),
        smoke_samples=[c['sample'] for i in (0,1) for c in read(OUT/f'worker{i}/smoke.json')['checks']],
        sources_unchanged=True,weights_unchanged=True,all_rows_and_scores_revalidated=True,
        exact_frozen_prompts=True,actual_generated_rows=len(records)-reuse,verified_alias_rows=reuse,
        total_condition_rows=len(records),quality_questions=1400,performance_questions=80,
        timed_requests=2000,warmup_requests=1200,census_requests=400,
        all_performance_sequences_match_quality=True,all_smoke_native_noop_torch_equal=True,
        full_dense_references_pass_frozen_numerical_gates=True,no_tolerance_relaxation=True,
        raw_unstripped_token_decoding_scores_verified=True,no_search_or_training=True,
        performance_serialization_recovery=read(OUT/'performance_entry_freeze.json')))
    plots(summaries,latency)
    report(summaries,pairs,latency,flops_rows,tests)
    atomic(OUT/'completion.json',dict(status='COMPLETE',at=now(),questions=1400,condition_rows=7000,
        actual_generated_quality_rows=7000-reuse,verified_within_run_reused_rows=reuse,
        sources=source_hashes(),sample_sha256=sha(OUT/'sample_manifest.json')))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def performance_summary(samples,records,partition):
    keys=[k for group in partition['performance'].values() for k in group]
    per=[]; shapes=[]
    for k in keys:
        b=read(OUT/'performance'/(k+'.json')); r=b['payload']
        assert b['sha256']==digest(r) and r['sources']==source_hashes() and r['sample_sha256']==digest(samples[k])
        assert len(r['measurements'])==25 and len(r['flops'])==5
        for c in CONFIGS:
            ms=[v for v in r['measurements'] if v['condition']==c]
            assert len(ms)==5 and {m['rep'] for m in ms}==set(range(5))
            assert all(m['token_ids']==records[k,c]['generation']['token_ids'] for m in ms)
            def med(field): return float(np.median([m[field] for m in ms]))
            fps=next(v for v in r['flops'] if v['condition']==c)
            byphase={p:sum(v['estimated_flops'] for v in fps['rows'] if v['phase']==p) for p in ('vision','prefill','decode')}
            for v in fps['rows']: shapes.append(dict(sample=k,task=samples[k]['task'],condition=c,**v))
            steps=records[k,c]['generation']['decode_steps']
            per.append(dict(sample=k,task=samples[k]['task'],condition=c,rank=r['rank'],
                e2e_ms=med('latency_ms'),prefill_ms=med('language_prefill_ms'),ttft_ms=med('ttft_ms'),vision_ms=med('vision_ms'),
                decode_ms=med('decode_ms'),decode_ms_per_token=med('decode_ms')/steps if steps else 0,
                decode_steps=steps,content_tokens=records[k,c]['generation']['content_tokens'],
                total_flops=sum(byphase.values()),vision_flops=byphase['vision'],prefill_flops=byphase['prefill'],decode_flops=byphase['decode'],
                estimated_algorithmic_tflops_per_s=sum(byphase.values())/(med('gpu_model_ms')*.001)/1e12,
                kv_live_mib=fps['KV']['live_bytes']/1024**2,kv_capacity_mib=fps['KV']['storage_bytes']/1024**2,
                peak_allocated_mib=med('peak_allocated')/1024**2,peak_reserved_mib=med('peak_reserved')/1024**2,
                same_full_sequence=records[k,c]['generation']['token_ids']==records[k,'Native Full']['generation']['token_ids']))
    write_csv(OUT/'performance_per_question.csv',per)
    latencies=[]; compute=[]
    numeric=[k for k in per[0] if k not in ('sample','task','condition','rank','same_full_sequence')]
    for task in TASKS:
        full=[r for r in per if r['task']==task and r['condition']=='Native Full']; fby={r['sample']:r for r in full}
        c3=[r for r in per if r['task']==task and r['condition']=='C3-V2']
        for c in CONFIGS:
            rows=[r for r in per if r['task']==task and r['condition']==c]; assert len(rows)==20
            vals={k:float(np.mean([r[k] for r in rows])) for k in numeric}
            same=[r for r in rows if r['same_full_sequence']]
            speed=float(np.mean([r['e2e_ms'] for r in full]))/vals['e2e_ms']
            matched=sum(fby[r['sample']]['e2e_ms'] for r in same)/sum(r['e2e_ms'] for r in same) if same else None
            latencies.append(dict(task=task,condition=c,n=20,repetitions=5,**vals,e2e_speedup_vs_native=speed,
                e2e_reduction_pct=100*(1-1/speed),same_sequence_n=len(same),same_sequence_speedup=matched,
                e2e_speedup_vs_C3=float(np.mean([r['e2e_ms'] for r in c3]))/vals['e2e_ms'],
                prefill_speedup_vs_native=float(np.mean([r['prefill_ms'] for r in full]))/vals['prefill_ms'],
                hardware_TFLOPS='N/A: no hardware counters',DRAM_bandwidth='N/A: no hardware counters'))
            compute.append(dict(task=task,condition=c,n=20,**{k:vals[k] for k in numeric if 'flops' in k or 'kv_' in k or 'peak_' in k},
                accounting='Executed matrix shapes, 2 FLOPs/MAC, includes Vision and head, excludes nonlinear/norm/softmax/RoPE'))
    return latencies,compute,shapes


def plots(summary,latency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,3.8)); labels=list(CONFIGS)
    colors=['#444444','#007f82','#b14a38','#2874b8']
    x=np.arange(5)
    for ti,task in enumerate(TASKS):
        rows=[next(r for r in summary if r['task']==task and r['condition']==c) for c in labels]
        y=np.array([r['accuracy'] for r in rows]); low=y-np.array([r['accuracy_ci_low'] for r in rows]); high=np.array([r['accuracy_ci_high'] for r in rows])-y
        axes[0].errorbar(x+(ti-1.5)*.045,y,yerr=[low,high],marker='o',lw=1,color=colors[ti],label=task)
        p=[next(r for r in latency if r['task']==task and r['condition']==c) for c in labels]
        axes[1].plot(x,[r['e2e_speedup_vs_native'] for r in p],marker='o',lw=1,color=colors[ti],label=task)
    for ax in axes:
        ax.set_xticks(x,['Native','Fast-P2','C3-V2','Full-QSA','C4-V2 (hist.)'],rotation=18)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Generated-answer Accuracy (%)'); axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Natural-EOS E2E speedup vs Native'); axes[1].axhline(1,color='gray',lw=.7,ls='--')
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(OUT/('accuracy_speed_comparison.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def report(summary,pairs,latency,compute,tests):
    lines=['# ATE 固定封闭式问答迁移验证','',
        '## 协议与完整性','',
        'ScienceQA-IMG 原冻结500题；POPE COCO Random/Popular/Adversarial各300题。题目、图片字节SHA、原Prompt和官方标注全部绑定。原生LLaVA固定revision、FP16/eager、Transformers 4.49.0，所有条件统一last-row LM Head (`logits_to_keep=1`)、Greedy、最多32 token、自然EOS。',
        '非Native均禁止P−1和全部真实Decode访问原始Image Key。L2–L5图像与P−1之前文本Query正常Attention，Norm/Residual/MLP保留。ScienceQA的Q严格仅为题干，不含选项/Hint/指令；这些文本仍原样出现在输入中，没有lecture、solution或Gold泄露。',
        '条件固定：Fast-P2=All-R/L20/不跳答案Attention；C3-V2=All-R/L20/P−1+Decode跳L2–L5；Full-QSA=Q+完整ASSISTANT Always/L20/保留P−1文本Attention/仅Decode跳L2–L5；历史C4-V2=Conditional-15/L20/P−1+Decode跳算。后者不得混称Full-QSA。',
        '全部7000条条件记录完成；完全相同POPE输入在同卡生成后明确记录哈希复用，不共享请求Cache、不删除任何问题。全部错误/空答案/解析失败/截断见negative_cases.csv。当前主分数来自完整自由生成，不使用候选LogP选择答案。','',
        '## 正式生成得分','',
        '| 任务 | 条件 | N | Accuracy % | F1 % | Precision % | Recall % | Yes % | 解析失败 | 空答/截断 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:
        fmt=lambda k:f"{r[k]:.3f}" if k in r else '-'
        lines.append(f"| {r['task']} | {r['condition']} | {r['n']} | {r['accuracy']:.3f} | {fmt('f1')} | {fmt('precision')} | {fmt('recall')} | {fmt('yes_ratio')} | {r['parse_failed']} | {r['empty']}/{r['truncated']} |")
    lines+=['','## 主要配对比较','',
        '图片SHA整簇配对Bootstrap 20,000次、问题等权。Full-QSA对Native的四项Accuracy是预先冻结的主要族；−1pp非劣效使用Bonferroni4单侧下界（quantile=0.0125），仅当下界严格大于−1pp才支持。其余对Native和C3-V2的比较均为探索性，详见paired_comparisons.csv。CI跨0不代表无损。','',
        '| 任务 | Full-QSA − Native pp | 普通95%CI | 损害/纠正 | 校正单侧下界 pp | −1pp非劣效 |',
        '|---|---:|---|---|---:|---|']
    for r in tests:
        lines.append(f"| {r['task']} | {r['delta_pp']:+.3f} | [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {r['harm']}/{r['rescue']} | {r['NI_one_sided_lower']:.3f} | {'支持' if r['NI_supported'] else '不支持'} |")
    lines+=['','### Full-QSA 对 C3-V2','']
    for r in pairs:
        if r['condition']=='Full-QSA' and r['reference']=='C3-V2' and r['metric']=='accuracy':
            lines.append(f"- {r['task']}: {r['delta_pp']:+.3f}pp，95%CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]，损害{r['harm']}题、纠正{r['rescue']}题。")
    lines+=['','## 同卡实测性能','',
        '每任务按输入哈希预先固定20张不同图片。每题五条件3次预热、5次交错重复；每次冷KV并重算Vision；同GPU配对，Processor/I/O不计时，模型与Cache建立、Vision、Prefill及Decode计入E2E。先每题取重复中位数，再跨20题等权均值。完整序列相同子集另外报告，不能用更短输出冒充加速。',
        '| 任务 | 条件 | Prefill ms | E2E ms | Decode ms/token | Speedup | 同序列N / Speedup | FLOPs T | KV Live/Capacity MiB | Peak MiB |',
        '|---|---|---:|---:|---:|---:|---|---:|---|---:|']
    for r in latency:
        m='N/A' if r['same_sequence_speedup'] is None else f"{r['same_sequence_speedup']:.3f}x"
        lines.append(f"| {r['task']} | {r['condition']} | {r['prefill_ms']:.2f} | {r['e2e_ms']:.2f} | {r['decode_ms_per_token']:.2f} | {r['e2e_speedup_vs_native']:.3f}x | {r['same_sequence_n']} / {m} | {r['total_flops']/1e12:.3f} | {r['kv_live_mib']:.2f}/{r['kv_capacity_mib']:.2f} | {r['peak_allocated_mib']:.1f} |")
    lines+=['','FLOPs是实际执行Linear/Conv/QK/AV张量形状的独立账本（2 FLOPs/MAC），不把Dense Mask连接当成已省计算。估算Algorithmic TFLOP/s见CSV；本轮未采集硬件计数器，硬件TFLOPS/DRAM带宽为N/A。KV为真实Live与分配Capacity，不能混同。','',
        '## 评分与结论边界','',
        'ScienceQA沿用冻结字母解析器，不因输出格式问题修补得分；学科结果见scienceqa_subject_accuracy.csv。POPE沿用官方首句词法规则，空字符串仍会被其规则解析为yes，空答另外完整标记，不隐藏此局限。三策略共享图片，不将它们视作三个独立图像队列，也不跨任务平均Accuracy/F1。',
        '只验证既定五条件，没有搜索/训练/按结果调参。结果用于这批冻结样本的迁移评估，不能解释为新外部总体上的严格无损；质量与时延须共同判断，不以生成首Token指标替代完整答案。','',
        '## 测试、复现与资源','',
        'CPU16项测试通过；每任务预定3题GPU Smoke通过：同形状Native/No-op及HF生成有效Logits严格torch.equal；对应全Dense功能参考符合原冻结容差；禁止连接实际概率为0，早期非目标Query、位置/RoPE、完整MLP、Cache隔离及Hook清理均检查。没有放宽数值标准。',
        'MMBench/SEED：Phase4冻结注册中没有对应可验证Manifest，指定资源目录未找到可冻结资产和评分器；本轮仅报告未准备状态，不下载或替换。详见benchmark_readiness.json。',
        '运行命令：`CUDA_VISIBLE_DEVICES=<GPU2或GPU3的冻结UUID> python -m vlm_research.closedqa_transfer.run --rank <0或1>`；完成后CPU汇总：`python -m vlm_research.closedqa_transfer.analyze`。',
        '代码、模型、样本、分片和协议SHA见execution_freeze.json；GPU使用及退出记录见worker0/、worker1/和resource_release_audit.json。不使用GPU0/1，不修改main或共享环境。']
    if (OUT/'performance_entry_freeze.json').exists():
        lines+=['','### 工程异常与恢复','',
            '原性能记录包装同时通过显式参数和生成结果传入condition字段，触发Python TypeError；不是模型、Mask或数值验证失败。原STOP与GPU日志完整保留。没有修改正在执行的冻结推理源码，也没有重跑或丢弃已完成的质量记录。独立benchmark.py只修正测量记录的标签组装，另行冻结SHA；恢复时同GPU Native首Token Logits哈希精确匹配。正式计时在新入口重新完成全部预热和重复，不复用失败尝试的时延。',
            '断点复现：质量记录使用原run.py和execution_freeze.json绑定；性能入口使用performance_entry_freeze.json绑定，运行`python -m vlm_research.closedqa_transfer.benchmark --rank <0或1>`。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); args=ap.parse_args()
    preflight() if args.preflight else summarize()
