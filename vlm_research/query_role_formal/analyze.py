"""Replay all 960 checkpoints and raw first logits before declaring COMPLETE."""
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from first_answer_attention.engine import first_metrics
from paper_rq1.scorers import gqa_prediction
import query_role_attention.analyze as shared
from query_role_attention.engine import target_positions
from .common import (OUT,MODEL,TASKS,CONDITIONS,COMPARISONS,SEED,read,write,sha256,
    digest,csv_write,verify,divergence,repeated_trigram)


def main():
    started=time.perf_counter();p=verify();binding=digest(p)
    gpu=read(OUT/'gpu_completion.json');smoke=read(OUT/'gpu_smoke_validation.json')
    assert gpu['passed'] and gpu['checkpoints']==960 and gpu['binding']==binding
    assert smoke['passed'] and smoke['questions']==3 and len(smoke['historical_A'])==3
    assert all(r['passed'] for r in smoke['historical_A'])
    samples=read(OUT/'sample_manifest.json')['samples'];mappings=read(OUT/'input_mappings.json')
    assert len(samples)==240 and len({s['question_id'] for s in samples})==240
    assert len({s['image_sha256'] for s in samples})==240
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==960
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    generations={c:[] for c in CONDITIONS};rows=[];checkpoint_hashes={};raw_hashes={}
    for i,s in enumerate(samples):
        assert sha256(s['image_path'])==s['image_sha256']
        for c in CONDITIONS:
            path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json"
            row=read(path);g=row['generation']
            assert row['binding']==binding and row['sample_sha256']==digest(s) and row['condition']==c
            assert row['code_commit']==p['code_commit'] and row['result_sha256']==digest(g)
            assert row['input_hashes']==mappings[i]['input_hashes']
            assert row['prompt_sha256']==s['input_mapping']['chat_prompt_sha256'] and row['gold']==s['answer']
            assert g['query_layout']==mappings[i]['role_layout']
            assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
            assert g['prediction']==gqa_prediction(g['text'])
            assert g['correct']==int(g['prediction']==s['answer'])
            assert g['own_cache'] and g['own_prefix'] and g['length']==len(g['token_ids'])
            assert 1<=g['length']<=32 and g['eos']==(g['token_ids'][-1]==2)
            assert g['empty']==(not bool(g['text'])) and g['truncated']==(g['length']==32 and not g['eos'])
            assert g['repeated_trigram']==repeated_trigram(g['token_ids'])
            assert sha256(row['raw']['path'])==row['raw']['sha256']
            with np.load(row['raw']['path']) as data:logits=torch.from_numpy(data['first_logits'].copy())
            assert torch.isfinite(logits).all() and int(logits.argmax())==g['token_ids'][0]
            if c=='Full':full=g;full_logits=logits
            metrics=first_metrics(full_logits,logits)
            assert all(value==g[key] for key,value in metrics.items())
            assert digest(logits.tolist())==g['first_logits_sha256']
            assert digest(full_logits.tolist())==g['full_first_logits_sha256']
            assert g['first_divergence']==divergence(full['token_ids'],g['token_ids'])
            assert g['sequence_changed']==(full['token_ids']!=g['token_ids'])
            assert g['full_correct_to_wrong']==int(full['correct'] and not g['correct'])
            assert g['full_wrong_to_correct']==int(not full['correct'] and g['correct'])
            assert len(g['hook_events'])==4*g['length'] and len(g['cache_lengths'])==g['length']
            for t in range(g['length']):
                cache=g['cache_lengths'][t];n=g['query_layout']['prompt_length']
                assert cache==dict(t=t,before=n+t-1 if t else 0,after=n+t,
                    consumed_token=g['token_ids'][t-1] if t else None,absolute_query=n+t-1)
                events=g['hook_events'][t*4:(t+1)*4]
                assert [e['layer'] for e in events]==[2,3,4,5]
                positions=target_positions(c,g['query_layout'],bool(t))
                for e in events:
                    assert e['phase']==('decode' if t else 'prefill')
                    assert e['target_count']==len(positions) and e['local_positions_sha256']==digest(positions)
                    assert e['applied']==bool(positions) and e['non_target_rows_exact']
                    assert not positions or e['target_exact_zero']
            expected=0 if c=='Full' else 4*(576 if c=='I-Bypass' else g['query_layout']['counts']['R'] if c=='R-Bypass' else g['length'])
            assert g['intervened_query_layer_updates']==expected
            generations[c].append(g)
            rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],prompt_sha256=row['prompt_sha256'],condition=c,
                question=s['question'],gold=s['answer'],text=g['text'],prediction=g['prediction'],
                token_ids=json.dumps(g['token_ids']),correct=g['correct'],
                correct_to_wrong=g['full_correct_to_wrong'],wrong_to_correct=g['full_wrong_to_correct'],
                first_top1_flip=g['first_top1_flip'],first_logp_loss=g['first_logp_loss'],
                first_divergence=g['first_divergence'],sequence_changed=g['sequence_changed'],
                eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                generated_tokens=g['length'],I_prefill_queries=576,R_prefill_queries=g['query_layout']['counts']['R'],
                A_queries_including_decode=g['length'],intervened_query_layer_updates=expected,
                binding=binding,code_commit=row['code_commit'],
                raw_logits_path=row['raw']['path'],raw_logits_sha256=row['raw']['sha256']))
            checkpoint_hashes[str(path.relative_to(OUT))]=sha256(path)
            raw_hashes[row['raw']['path']]=row['raw']['sha256']
    scores=np.array([[generations[c][i]['correct'] for c in CONDITIONS] for i in range(240)],dtype=float)
    draws=shared.paired_bootstrap(samples,scores,repetitions=20000,seed=SEED)
    comparisons=[]
    for a,b in COMPARISONS:
        ia,ib=CONDITIONS.index(a),CONDITIONS.index(b)
        lo,hi=np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5])
        comparisons.append(dict(comparison=f'{a} - {b}',a=a,b=b,questions=240,image_clusters=240,
            delta_accuracy_pp=float(100*(scores[:,ia]-scores[:,ib]).mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            b_correct_a_wrong=int(((scores[:,ib]==1)&(scores[:,ia]==0)).sum()),
            b_wrong_a_correct=int(((scores[:,ib]==0)&(scores[:,ia]==1)).sum()),
            bootstrap_repetitions=20000,seed=SEED,interval='Ordinary paired exploratory image-cluster 95%'))
    summaries=[]
    for k,c in enumerate(CONDITIONS):
        lo,hi=np.percentile(100*draws[:,k],[2.5,97.5])
        summaries.append(dict(condition=c,**shared.summary(generations[c],generations['Full']),
            ci95_accuracy_low_pct=float(lo),ci95_accuracy_high_pct=float(hi),
            first_top1_flip_pct=100*np.mean([g['first_top1_flip'] for g in generations[c]]),
            repeated_trigram=sum(g['repeated_trigram'] for g in generations[c])))
    tasks=[]
    for task in TASKS:
        indices=[i for i,s in enumerate(samples) if s['task']==task]
        assert len(indices)==40
        for c in CONDITIONS:tasks.append(dict(task=task,condition=c,**shared.summary(
            [generations[c][i] for i in indices],[generations['Full'][i] for i in indices])))
    for name,data in [('per_question.csv',rows),('accuracy_summary.csv',summaries),
        ('task_accuracy.csv',tasks),('paired_comparisons.csv',comparisons),
        ('negative_tail_cases.csv',sorted([r for r in rows if r['condition']!='Full'],key=lambda r:r['first_logp_loss'],reverse=True)[:30])]:
        csv_write(OUT/name,data)
    shared.OUT=OUT;shared.plot(summaries,comparisons,tasks)
    apps=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
    table=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    pids=[read(path)['pid'] for path in OUT.glob('environment_*.json')]
    assert pids and all(not Path(f'/proc/{pid}').exists() for pid in pids)
    write(OUT/'gpu_release.json',dict(our_gpu_processes_exited=True,pids=pids,physical_gpu=1,
        table=table,current_compute_processes=apps,currently_idle=not bool(apps),no_external_process_terminated=True))
    for path,info in p['model_files'].items():assert sha256(path)==info['sha256'],path
    verify()
    write(OUT/'validation.json',dict(passed=True,questions=240,conditions=4,checkpoints=960,
        no_missing_or_duplicate=True,image_clusters=240,task_counts={t:40 for t in TASKS},
        gpu_smoke_sha256=sha256(OUT/'gpu_smoke_validation.json'),cpu_validation_sha256=sha256(OUT/'cpu_validation.json'),
        historical_A_same_shape_exact=True,metric_replay_max_abs_difference=0.,raw_logits_verified=960,
        token_text_and_score_replay_exact=True,input_image_sha256_reverified=True,
        model_full_sha256_unchanged=True,sources_and_historical_artifacts_unchanged=True,
        gpu_processes_exited=True,exploratory=True,noninferiority_claim=False,
        raw_arrays_on_github=False,raw_hashes=raw_hashes,checkpoint_hashes=checkpoint_hashes))
    report(summaries,comparisons,tasks,generations,p,gpu)
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name not in ('worker.lock','completion.json')]
    write(OUT/'completion.json',dict(complete=True,status='COMPLETE',formal_questions=240,
        formal_generations=960,binding=binding,frozen_code_commit=p['code_commit'],
        analysis_seconds=time.perf_counter()-started,artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(files)}))
    print(json.dumps(dict(accuracy=summaries,comparisons=comparisons),ensure_ascii=False,indent=2),flush=True)
    print('COMPLETE: 960 raw arrays/checkpoints replayed; figures, model hashes, GPU release verified',flush=True)


def report(summaries,comparisons,tasks,generations,p,gpu):
    by={r['condition']:r for r in summaries};i,r,a=[by[c] for c in CONDITIONS[1:]]
    pair=lambda left,right:next(x for x in comparisons if x['a']==left and x['b']==right)
    fmt=lambda x:f"{x['delta_accuracy_pp']:+.3f} pp，95%CI [{x['ci95_low_pp']:+.3f}, {x['ci95_high_pp']:+.3f}]"
    lines=['# Query-Role Attention Dormancy：正式240题结果','',
        '**COMPLETE：240题、240个Image SHA256簇、四条件960次新自由生成。Smoke不计入正式结果。**','',
        '## 主要结果','',
        '|条件|正确/240|Accuracy|ΔFull|原对→错|原错→对|首Token翻转|序列变化|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in summaries:lines.append(f"|{v['condition']}|{v['correct']}/240|{v['accuracy_pct']:.3f}%|{v['delta_full_pp']:+.3f}pp|{v['correct_to_wrong']}|{v['wrong_to_correct']}|{v['first_top1_flip']}|{v['sequence_changed']}|")
    lines+=['','## 配对比较','',
        '固定seed=20261102，20,000次Image SHA256整簇配对Bootstrap，题目加权普通探索性95%CI。五项比较均报告，不作等价、非劣效或多重比较确认性声明。','']
    for v in comparisons:lines.append(f"- {v['comparison']}：{fmt(v)}；B原对→A错 {v['b_correct_a_wrong']}，B原错→A对 {v['b_wrong_a_correct']}。")
    lines+=['','## 六任务Accuracy','', '|任务|Full|I-Bypass|R-Bypass|A-Bypass|','|---|---:|---:|---:|---:|']
    for task in TASKS:lines.append('|'+task+'|'+'|'.join(f"{next(v['accuracy_pct'] for v in tasks if v['task']==task and v['condition']==c):.1f}%" for c in CONDITIONS)+'|')
    lines+=['','## 首Token与异常','', '|条件|signed LogP loss均值|中位|P5/P95|EOS|空答案|截断|重复3gram|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for v in summaries:lines.append(f"|{v['condition']}|{v['first_logp_loss_mean']:+.6f}|{v['first_logp_loss_median']:+.6f}|{v['first_logp_loss_p5']:+.6f}/{v['first_logp_loss_p95']:+.6f}|{v['eos']}|{v['empty']}|{v['truncated']}|{v['repeated_trigram']}|")
    relay=[g['query_layout']['counts']['R'] for g in generations['Full']]
    lines+=['','Signed loss固定本机Full首Token，等于logP_Full−logP干预，负值保留；不是Gold Loss，也不能代替最终Accuracy。重复仅为非EOS Token的重复3gram描述性标记。完整负面案例与所有翻转保存在CSV和检查点。','',
        '## 七个研究问题','',
        f"1. **I-Bypass：** 相对Full {fmt(pair('I-Bypass','Full'))}，损害{i['correct_to_wrong']}题、纠正{i['wrong_to_correct']}题。该证据只涉及L2–L5的视觉Query输出更新，不等于视觉Token整体无用。",
        f"2. **R是否比I更有损害：** R−I为{fmt(pair('R-Bypass','I-Bypass'))}。"+('本批探索性配对区间完全为负，支持R干预更有损害。' if pair('R-Bypass','I-Bypass')['ci95_high_pp']<0 else '区间未完全为负，不能只凭点估计认定R总体更有损害。'),
        f"3. **A-Bypass低敏感性：** 相对Full {fmt(pair('A-Bypass','Full'))}；损害{a['correct_to_wrong']}题、纠正{a['wrong_to_correct']}题。必须同时观察损害和纠正，零净变化不证明无损。",
        f"4. **角色依赖：** R−A为{fmt(pair('R-Bypass','A-Bypass'))}。"+('本批结果支持目标层Attention更新的作用随Query角色改变，而不是整层统一冗余。' if pair('R-Bypass','A-Bypass')['ci95_high_pp']<0 else '当前配对不确定性不足以据此确认清晰的R/A角色差异。'),
        '5. **读写不对称假说：** 若Prefill角色干预比A更有损害，该模式与“保留重要Prefill计算、简化答案阶段计算”相容；具体支持程度必须受上述真实损益和区间约束。本轮不把研究动机当作已证实机制。',
        f'6. **尚未证明：** I固定576个Query，R为{min(relay)}–{max(relay)}个（均值{np.mean(relay):.2f}），A为每条件自身P−1及全部真实Decode，包含EOS预测。Query数量、位置和语义不同，不能按总体损害排列单位Query贡献。R还含指令/模板，不是唯一视觉写入节点；未做中介分析或证明唯一信息路径。这批240题也不是独立确认集。',
        '7. **优化启示：** 后续应针对角色和时间范围建立候选，而非直接跳过整个L2–L5；输出消融不删除QKV、Attention计算或KV缓存。其他当前/未来Query仍可能需要这些K/V。本轮无实际加速、物理剪枝或显存收益证据。','',
        '## 执行与正确性','',
        f"基于修复Commit `{p['base_commit']}`，独立分支`codex/query-role-attention-boundary-240-formal`，执行源码Commit `{p['code_commit']}`。旧Partial与Smoke修复目录均原样保留。",
        '任务书给出的Manifest哈希为抄写不完整的字符串；本轮没有改变Manifest，原文件及独立结果目录副本的实际64位SHA256均为`'+p['manifest_note']['verified_original_sha256']+'`，见protocol.json。',
        f"先通过{read(OUT/'cpu_validation.json')['tests']}项CPU回归，冻结新源码及协议，再用三道原冻结问题重新通过GPU Smoke。HF/Full/No-op在相同LM Head形状下用torch.equal逐位检查；历史A对照保留原E控制器，仅为Forward显式加入logits_to_keep=1，三题全部逐步精确一致；不重新调查旧根因、不放宽容差。",
        '逐次验证Hook目标严格零、非目标行没有被直接修改，真实位置、Cache长度及输入哈希正确；Smoke验证32层MLP调用和Cache不别名；参数版本、正常/异常清理通过。正式960条结果从Token解码、评分及原始首Token向量重算，所有首Token指标差异为0。模型文件运行前后完整SHA256通过。',
        f"只使用物理GPU1，UUID `{p['gpu']['uuid']}`，进程内cuda:0，只加载一份模型，四条件串行。正式采集{gpu['formal_seconds']:.2f}秒，含加载/Smoke总计{gpu['seconds_including_load_smoke']:.2f}秒；峰值allocated {gpu['memory']['peak_allocated_mib']:.2f} MiB。GPU进程已退出，释放审计见gpu_release.json。",
        '原始首Token向量保留服务器资源目录，960个路径及SHA256绑定在检查点/validation.json，不将模型权重或大型数组上传GitHub。960份检查点、逐题CSV、五项配对区间及全部PDF/PNG提交到独立分支。','',
        '**本轮结束，不追加条件、训练、扩样或实际加速实现。**','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
        raise
