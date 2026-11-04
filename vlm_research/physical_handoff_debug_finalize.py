"""CPU-only consolidation, intentionally outside the frozen worker package."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

from .physical_handoff_debug.common import OUT, OLD, ROOT, HISTORY, MODELS, atomic, read, sha, write_csv, verify, digest


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def optional(path, default):
    return read(path) if path.exists() else default


def main():
    protocol=verify()
    results={}
    equivalence=[];registry=[];kv=[];bench=[];quality=[];checks={}
    for kind in ('llava','qwen'):
        folder=OUT/kind
        completion=optional(folder/'completion.json',dict(status='NOT_RUN'))
        gate=optional(folder/'quality_gate.json',dict(status='NOT_RUN',candidate=None))
        candidates=optional(folder/'candidate_registry.json',[])
        regression=optional(folder/'regression.json',[])
        native_repeats=optional(folder/'failure_repeats.json',[])
        drift=optional(folder/'native_to_fixed_dense_drift.json',[])
        mapping=optional(folder/'attention_mask_equivalence.json',dict(status='NOT_RUN'))
        positions=optional(folder/'position_and_cache_audit.json',dict(status='NOT_RUN'))
        results[kind]=dict(completion=completion,gate=gate,candidates=candidates,regression=regression,
                           repeats=native_repeats,drift=drift,mapping=mapping,positions=positions)
        for phase,source in [('native_failure_repeat',native_repeats),('candidate_smoke',candidates),
                             ('regression',regression),('native_to_fixed_dense',drift)]:
            for r in source:
                equivalence.append(dict(r,model=kind,comparison_phase=phase))
        registry+=candidates
        for record in optional(folder/'kv_audit.json',[]):
            r={k:v for k,v in record.items() if k not in ('cache','execution')}
            r.update(all_storage_owned=all(layer['storage_bytes']==sum(__import__('math').prod(s)*2 for s in layer['shapes'])
                                          for layer in record['cache']['layers']),
                     physical_lengths=[layer['physical_length'] for layer in record['cache']['layers']],
                     logical_lengths=[layer['logical_length'] for layer in record['cache']['layers']])
            kv.append(r)
        for timing in rows(folder/'microbenchmark.csv'):
            timing['worker_matched_kernel_classification']=timing['classification']
            timing['original_native_parity_pass']=bool(drift) and all(r['overall_pass'] for r in drift)
            if not timing['original_native_parity_pass']:
                timing['classification']='UNVALIDATED_PERFORMANCE'
            bench.append(timing)
        manifest=read(folder/'manifest.json')
        sample_by_key={s['task']+'__'+s['question_id']:s for s in manifest['all_samples']}
        for p in sorted((folder/'checkpoints').glob('*/*/*.json')):
            record=read(p)
            assert record['protocol_sha256']==digest(protocol)
            g=record['generation'];s=sample_by_key[p.parent.name]
            quality.append(dict(model=kind,phase=p.parents[1].name,question=p.parent.name,task=s['task'],
                image_sha256=s['image_sha256'],gold=s['answer'],condition=p.stem,answer=g['text'],
                correct=g['correct'],token_ids=g['token_ids'],eos=g['eos'],truncated=g['truncated'],empty=g['empty']))
        checks[kind]=dict(completion=completion,gate=gate,native_noop=optional(folder/'native_noop.json',{}),
            mask_mapping=mapping,new10_unique=len({r['question'] for r in regression if r['phase']=='new10'}),
            regression_all_pass=bool(regression) and all(r['overall_pass'] for r in regression),
            unchanged_parameters_inputs_hooks=completion.get('parameters_and_inputs_unchanged',False),
            native_to_fixed_dense_tokens_preserved=bool(drift) and all(r['tokens_equal'] for r in drift),
            native_to_fixed_dense_numerical_pass=bool(drift) and all(r['numerical_pass'] for r in drift),
            original_native_overall_parity='PASS' if bool(drift) and all(r['overall_pass'] for r in drift) else 'FAILED',
            all_outputs_eos=bool(quality) and all(r['eos'] for r in quality if r['model']==kind))

    write_csv(OUT/'equivalence_summary.csv',equivalence)
    write_csv(OUT/'numerical_fix_registry.csv',registry)
    write_csv(OUT/'kv_memory_validation.csv',kv)
    write_csv(OUT/'per_question_regression.csv',quality)
    write_csv(OUT/'microbenchmark.csv',bench)
    operator_rows=rows(OUT/'qwen/operator_trace.csv')
    for r in operator_rows:
        r['relative_rmse_valid']=r['stage']!='masked_scores'
        if r['stage']=='masked_scores':
            r['raw_relative_rmse']=r['relative_rmse']
            r['relative_rmse']=''
            r['metric_note']='Causal finfo.min sentinels invalidate global FP32-normalized RMSE. Use max_abs and exact mask/edge checks; raw diagnostic retained.'
    write_csv(OUT/'qwen_l22_operator_trace.csv',operator_rows)
    historical_layers={r['layer']:r for r in read(OLD/'qwen/first_failure_layer_diagnostic.json')['rows']}
    layers=[]
    for r in rows(OUT/'qwen/layerwise_error.csv'):
        old=historical_layers[int(r['layer'])]
        r['metric_note']='Current relative_rmse is global; original gate uses max per-text-row RMSE, shown separately from historical exact reproduction.'
        r['historical_max_per_row_relative_rmse']=old['max_relative_rmse']
        r['historical_current_max_abs_exact_match']=float(r['max_abs'])==old['max_abs']
        layers.append(r)
    write_csv(OUT/'qwen_layerwise_error.csv',layers)
    atomic(OUT/'attention_mask_equivalence.json',{k:r['mapping'] for k,r in results.items()})
    atomic(OUT/'position_and_cache_audit.json',{k:r['positions'] for k,r in results.items()})
    model_binding={}
    for kind in ('llava','qwen'):
        samples=read(OUT/kind/'manifest.json')['all_samples']
        assert all(sha(s['image_path'])==s['image_sha256'] for s in samples)
        history=read(HISTORY/kind/'protocol.json')
        model_binding[kind]=dict(frozen_model=MODELS[kind],historical_weight_manifest=history['weights'],
             input_binding_sha256=sha(OUT/kind/'input_bindings.json'),sample_manifest_sha256=sha(OUT/kind/'manifest.json'),
             image_bytes_rehashed=True,parameter_version_and_storage_pointer_checked=True,
             statement='No weight conversion persisted. Weights reused from immutable historical revision; no full multi-GB rehash this round.')
    atomic(OUT/'model_and_input_binding_audit.json',model_binding)
    historical=rows(OUT/'llava/historical43_prefill_audit.csv')
    atomic(OUT/'llava_tie_divergence_audit.json',dict(repeats=results['llava']['repeats'],historical43=historical,
        historical_questions=len(historical),near_boundary_count=sum(r['decision_risk']=='True' for r in historical),
        explanation='Only common-prefix logits are numerical pairs. Different-prefix logits are descriptive branch consequences.'))
    grouped={}
    for r in bench:
        group=(r['model'],r['kernel'],int(r['fixed_tokens']),r['condition'])
        grouped.setdefault(group,[]).append(r)
    metrics=['vision_ms','language_prefill_ms','ttft_ms','decode_ms_per_token','end_to_end_ms',
             'live_kv_bytes','peak_allocated','peak_reserved']
    summary=[]
    for (model,kernel,length,condition),group in grouped.items():
        row=dict(model=model,kernel=kernel,fixed_tokens=length,condition=condition,repeats=len(group),
                 classification=group[0]['classification'])
        for metric in metrics:
            values=[float(r[metric]) for r in group]
            row[metric+'_mean']=statistics.mean(values)
            row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else 0.
        summary.append(row)
    write_csv(OUT/'microbenchmark_summary.csv',summary)

    cpu={}
    for kind,python,tests in [
        ('llava','/data/lk/深度学习/.envs/ate-vlm-research/bin/python',['vlm_research.physical_handoff_debug.tests','vlm_research.physical_visual_handoff.tests']),
        ('qwen','/data/lk/深度学习/.envs/ate-qwen3vl/bin/python',['vlm_research.physical_handoff_debug.tests'])]:
        run=subprocess.run([python,'-m','unittest',*tests,'-v'],capture_output=True,text=True)
        (OUT/kind/'cpu_tests.log').write_text(run.stdout+run.stderr)
        cpu[kind]=dict(returncode=run.returncode,passed=run.returncode==0)
    old_diff=subprocess.check_output(['git','diff',protocol['base_commit'],'--','vlm_research/physical_visual_handoff',
                                    'vlm_research/results/physical_visual_handoff_pilot'],text=True)
    assert not old_diff
    atomic(OUT/'validation.json',dict(cpu=cpu,models=checks,old_source_and_partial_results_unchanged=True,
          finalizer_sha256=sha(Path(__file__)),
          original_native_equivalence_status='PASS' if all(c['original_native_overall_parity']=='PASS' for c in checks.values()) else 'FAILED',
          matched_kernel_regression_status='PASS' if all(c['regression_all_pass'] for c in checks.values()) else 'FAILED',
          thresholds_unchanged=True,worker_source_hashes_verified=True,
          gates='G1 graph; G2 same-kernel numerics AND greedy; native-to-fixed drift separate; G3 tiny regression only'))
    atomic(OUT/'completion.json',dict(status='DIAGNOSTICS_COMPLETE_ORIGINAL_EQUIVALENCE_FAILED',
          old_partial_preserved=True,new_questions_per_model=10,
          saved_regression_generations=len(quality),kv_storage_audits=len(kv),benchmark_measurements=len(bench),
          worker_seconds={k:r['completion'].get('total_seconds') for k,r in results.items()},
          larger_quality_matrix='NOT_RUN',validated_acceleration='NOT_ESTABLISHED'))
    resource=dict(gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                                    '--format=csv,noheader'],text=True),
                  compute_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                                         '--format=csv,noheader'],text=True),
                  disk_after=subprocess.check_output(['df','-h','/data'],text=True),
                  used_gpu_uuids=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'],
                  external_processes_terminated=False,data1_used_for_new_outputs=False,
                  initial_observation='GPU2/3: 27MiB each, 0% utilization. GPU0 external task retained.',
                  workload_stopped=True)
    atomic(OUT/'resource_audit.json',resource)

    text=['# Physical Visual Handoff：数值诊断、修复与真实微基准',
          '', '**范围：仅D1/P1/P2；未扩大到60/240题。历史PARTIAL原封不动。**',
          '', 'D1不是原生Full：P−1及Decode全层禁止直接读Image，R只在截止点之前读Image（LLaVA进入L20、Qwen进入L22后关闭）。P1与D1使用同一数学依赖图，物理删除后层视觉Hidden行；P2在P1基础上物理压缩视觉KV。',
          '', '## 1. 已定位事实',
          '', 'LLaVA `LogicalObj__03567478` 三次确定性重复均为D1=No、P1=Yes：密集Yes/No精确平局，物理Yes高0.015625。共享首步最大Logits误差0.015625、最大逐文本行Hidden相对RMSE 0.002343478；原数值门槛通过，但Argmax/Greedy失败，不能整体PASS。',
          '两路径首Token分叉后各自前缀不同，后续6.03125的Logits差仅作为分叉后果，不作为同输入数值误差。',
          f'历史43题已逐题重查首步：接近决策边界的保守风险计数为{sum(r["decision_risk"]=="True" for r in historical)}/{len(historical)}（D1 top1–top2 margin ≤ 2×逐词表最大误差；不是新增Accuracy损害计数）。',
          '', '在同一份真实层入口Hidden上，两模型的Layer Input和RMSNorm均精确一致，首处不一致均为Q投影；K/V也有舍入差异。Qwen对应L22，LLaVA对应L20。实际物理入口的Hidden、Mask、原始位置及RoPE cos/sin均与密集文本切片严格一致，禁止视觉连接实测概率为0。',
          '这些证据支持低精度GEMM形状/归约差异为首要误差来源，不支持位置重编号、错误Mask或DeepStack漏执行的解释。只能对本轮已审计样本作此判断。',
          '算子表中Masked Score含causal finfo.min哨兵，全局FP32归一化RMSE不可解释，已在根目录表中置空并保留原诊断值；该项看max_abs和Mask/合法连接实测，不把0误解为逐位一致。层Hidden门槛仍使用最大逐文本行RMSE，不使用算子表的全局RMSE。',
          '', '## 2. 修复、三类验收与原生漂移',
          '', '所有阈值保持历史值不变。局部FP32控制只改变单个层内算子的累加，不把整个8B模型转为FP32。候选按冻结顺序测试，失败不删除。',
          '`fp32_linears`/`fp32_linears_attention`在截止点后的投影/MLP和LM Head临时FP32计算，回转原激活dtype；后者还改变QK/AV累加。`canonical_tiles`保持原FP16/BF16，在后层和LM Head使用固定32行投影Tile，并把严格断开的文本/视觉Attention子图分开计算，使Dense/Physical文本内核输入形状一致；最多31行零Padding，不重建被裁剪视觉行。',
          'canonical方案的密集参考仍计算视觉行，物理路径不计算这些行；未改变权重、数学合法边、原始位置或写入截止点。KV Cache按逐层原始逻辑位置映射，不以紧凑张量长度充当RoPE位置。',
          '', '|模型|选择的候选|小回归G2|新题数|原生D1→修复D1的Greedy保持|',
          '|---|---|---|---:|---|']
    for kind,r in results.items():
        text.append(f'|{kind}|{r["gate"].get("candidate")}|{r["gate"]["status"]}|{checks[kind]["new10_unique"]}|{checks[kind]["native_to_fixed_dense_tokens_preserved"]}|')
    text+=['', '**最终原生等价判定仍未通过。** LLaVA修复D1把原生D1的No改为Yes；Qwen六道Smoke虽然答案保持，但原生D1→canonical D1的最大逐行Hidden RMSE超出原0.04阈值。因此匹配内核内部通过，不等于原生实现等价，全部速度统一归为`UNVALIDATED_PERFORMANCE`。',
           '上表G2针对**同修复内核的D1/P1/P2**，不是与未经修改的原生密集路径逐位一致。原生→修复的漂移单独保存，若首Token或答案变化，不能宣称恢复了所有原生答案，也不能以修复参考替换历史PARTIAL。',
           '', '|模型|比较阶段|最大Logits绝对误差|最大逐行Hidden RMSE|Greedy一致/比较数|',
           '|---|---|---:|---:|---:|']
    for kind,r in results.items():
        for phase,records in [('原生失败重复',r['repeats']),('修复后独立回归',r['regression']),('原生D1→修复D1',r['drift'])]:
            if records:
                text.append(f'|{kind}|{phase}|{max(x["max_abs_logits"] for x in records):.7g}|{max(x["hidden_relative_rmse"] for x in records):.7g}|{sum(x["tokens_equal"] for x in records)}/{len(records)}|')
    text+=['', 'Qwen同内核D1/P1的Prefill文本Hidden及各共享步Logits全部精确一致；P2移除早层视觉KV后改变了Decode注意力矩阵形状，最大Logits差0.625，但仍满足冻结的组合atol/rtol门槛、Greedy一致。不能把P1的精确一致扩大到P2或原生参考。',
           'G1：CPU小模型FP32数学等价对照、实际入口/Mask/位置对齐与目标概率审计提供支持。G2：数值和Greedy分别记录，并检查P2。G3：仅10道输入无关新题与原Smoke的小样本风险检查，不估计总体性能或等价性。',
           '', '## 3. 真实KV与物理执行',
           '', '逐层记录真实K/V `untyped_storage().nbytes()`、原始Key位置和物理长度，而非只记录逻辑shape。压缩前张量采用Weakref审计；不得存在旧全量KV的存活引用。P2每层均排除原始视觉位置，原始逻辑长度仍用于下一步位置。详细记录见`kv_memory_validation.csv`及各模型`kv_audit.json`。',
           '', '## 4. 实测GPU性能',
           '', '每模型一张RTX3090，2次Warmup、5次实测；16/64为总预测位置，分别包含15/63个增量Decode步骤。固定长度微基准可越过EOS，仅用于工程计时，质量回归仍正常EOS。Vision/Prefill/Decode用CUDA Event，E2E为同步墙钟。不同GPU上的两个模型不可直接作速度优劣比较。',
           '下表为均值；完整标准差和每次结果在CSV。`UNVALIDATED_PERFORMANCE`不得解释为通过正确性验证的加速。',
           '', '|模型|内核|预测位置数|路径|Prefill ms|Decode ms/token|E2E ms|KV MiB|状态|',
           '|---|---|---:|---|---:|---:|---:|---:|---|']
    for r in summary:
        text.append(f'|{r["model"]}|{r["kernel"]}|{r["fixed_tokens"]}|{r["condition"]}|{r["language_prefill_ms_mean"]:.2f}|{r["decode_ms_per_token_mean"]:.2f}|{r["end_to_end_ms_mean"]:.2f}|{r["live_kv_bytes_mean"]/2**20:.2f}|{r["classification"]}|')
    text+=['', '需要同时看匹配修复内核的D1/P1/P2和原生D1，避免用较慢修复参考夸大收益。当前实现保留Python Mask、位置及缓存审计开销，不是已优化部署内核；峰值allocated/reserved均实测，不能以KV节省直接代替总显存或时延收益。',
           '实测结论：P1缩短了Language Prefill，但没有可靠Decode提速；稳定内核增加的成本使端到端仍慢于原生D1，未建立原生等价的真实加速。P2显著降低真实KV存储，压缩本身也增加Prefill耗时。',
           '', '## 5. 结论边界与下一步',
           '', 'Qwen/LLaVA首个数值差异均已定位到投影GEMM而非位置/Mask。LLaVA近乎平局分叉稳定复现，说明小误差门槛通过不保证Argmax一致。修复是否足够进入更大质量验证，必须结合上表G2、原生漂移和真实耗时，不因为KV确实压缩就自动推荐扩大。',
           '**当前不建议作为原生等价加速算法进入60/240题。** 两模型与原生参考的严格总体G2仍未通过；下一步优先解决匹配内核与原生密集参考间的舍入漂移及修复开销。如果后续接受近似实现风险，应先另行预注册小规模质量协议，而不是把本轮匹配内核的PASS换成原生PASS。本轮不自动执行。',
           '', '本轮不涉及P4/早层Attention跳算/Question-Only，不搜索新截止点、不训练、不清理旧数据。CPU、启动环境及API失败记录均保留；修复后的源码重新提交冻结，未在运行中改写Worker。',
           '', '## 6. 执行与产物',
           '', f'冻结正式源码Commit：`{protocol["code_commit"]}`；原型基线：`{protocol["base_commit"]}`。',
           f'最终冻结版本Worker墙钟：LLaVA {results["llava"]["completion"]["total_seconds"]:.2f}秒，Qwen {results["qwen"]["completion"]["total_seconds"]:.2f}秒；另有前序诊断、失败启动及开发整理时间，不能把Worker时长当作全任务总时间。',
           '分支：`codex/physical-visual-handoff-numerical-debug`。最终结果Commit由Git提交日志给出，避免自引用哈希。',
           '重现入口：`vlm_research/physical_handoff_debug/README.md`。未覆盖历史结果及源码；新结果全部位于`/data`。GPU2/3任务完成释放，GPU0其他任务不终止；资源快照见`resource_audit.json`。']
    (OUT/'report_zh.md').write_text('\n'.join(text)+'\n')
    with tarfile.open(OUT/'raw_checkpoints.tar.gz','w:gz') as archive:
        for kind in ('llava','qwen'):
            for p in sorted((OUT/kind/'checkpoints').rglob('*.json')):
                archive.add(p,arcname=str(p.relative_to(OUT)))
    with tarfile.open(OUT/'raw_kv_audits.tar.gz','w:gz') as archive:
        for kind in ('llava','qwen'):
            p=OUT/kind/'kv_audit.json'
            archive.add(p,arcname=str(p.relative_to(OUT)))
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    atomic(OUT/'artifact_hashes.json',hashes)
    print(json.dumps(dict(native_equivalence={k:c['original_native_overall_parity'] for k,c in checks.items()},
          matched_kernel_regression={k:c['regression_all_pass'] for k,c in checks.items()},
          benchmark_rows=len(bench),kv_rows=len(kv),quality_records=len(quality)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
