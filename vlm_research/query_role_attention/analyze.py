"""CPU-only completeness checks, image-cluster paired statistics and figures."""
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from first_answer_attention.engine import first_metrics
from paper_rq1.scorers import gqa_prediction
from .common import (ROOT, OUT, SEED, TASKS, CONDITIONS, COMPARISONS, read, write,
                     csv_write, verify, sha256, digest)


def paired_bootstrap(samples, scores, repetitions=20000, seed=SEED):
    images = sorted({s['image_sha256'] for s in samples})
    index = {h:i for i,h in enumerate(images)}
    totals = np.zeros((len(images), scores.shape[1]), dtype=float)
    counts = np.zeros(len(images), dtype=float)
    for i,s in enumerate(samples):
        k = index[s['image_sha256']]
        totals[k] += scores[i]
        counts[k] += 1
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 500):
        ids = rng.integers(0, len(images), size=(min(500,repetitions-start), len(images)))
        draws.append(totals[ids].sum(1)/counts[ids].sum(1)[:,None])
    return np.concatenate(draws)


def summary(generations, full):
    losses = np.array([g['first_logp_loss'] for g in generations], dtype=float)
    return dict(questions=len(generations), correct=sum(g['correct'] for g in generations),
        accuracy_pct=100*np.mean([g['correct'] for g in generations]),
        delta_full_pp=100*np.mean([g['correct']-f['correct'] for g,f in zip(generations,full)]),
        correct_to_wrong=sum(f['correct'] and not g['correct'] for g,f in zip(generations,full)),
        wrong_to_correct=sum(not f['correct'] and g['correct'] for g,f in zip(generations,full)),
        first_top1_flip=sum(g['first_top1_flip'] for g in generations),
        sequence_changed=sum(g['token_ids'] != f['token_ids'] for g,f in zip(generations,full)),
        first_logp_loss_mean=float(losses.mean()), first_logp_loss_median=float(np.median(losses)),
        first_logp_loss_p5=float(np.percentile(losses,5)), first_logp_loss_p95=float(np.percentile(losses,95)),
        first_logp_loss_min=float(losses.min()), first_logp_loss_max=float(losses.max()),
        eos=sum(g['eos'] for g in generations), truncated=sum(g['truncated'] for g in generations),
        empty=sum(g['empty'] for g in generations), mean_generated_tokens=float(np.mean([g['length'] for g in generations])),
        mean_intervened_query_layer_updates=float(np.mean([g['intervened_query_layer_updates'] for g in generations])))


def main():
    started = time.perf_counter()
    p = verify(); binding = digest(p)
    assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_completion.json')['binding'] == binding
    samples = read(OUT/'sample_manifest.json')['samples']
    assert len(samples) == 240 and len({s['image_sha256'] for s in samples}) == 240
    assert len(list((OUT/'checkpoints').glob('*/*.json'))) == 960
    generations = {c:[] for c in CONDITIONS}
    per_question, checkpoint_hashes, raw_hashes = [], {}, {}
    max_metric_difference = 0.
    for sample in samples:
        first_full = None
        for condition in CONDITIONS:
            path = OUT/f"checkpoints/{sample['task']}__{sample['question_id']}/{condition}.json"
            row = read(path); g = row['generation']
            assert row['binding'] == binding and row['sample_sha256'] == digest(sample)
            assert row['result_sha256'] == digest(g) and row['condition'] == condition
            assert g['prediction'] == gqa_prediction(g['text'])
            assert g['correct'] == int(gqa_prediction(g['text']) == sample['answer'])
            assert g['own_cache'] and g['own_prefix'] and 1 <= len(g['token_ids']) <= 32
            assert g['eos'] == (g['token_ids'][-1] == 2)
            assert sha256(row['raw']['path']) == row['raw']['sha256']
            with np.load(row['raw']['path']) as raw:
                logits = torch.from_numpy(raw['first_logits'].copy())
            assert torch.isfinite(logits).all() and int(logits.argmax()) == g['token_ids'][0]
            if condition == 'Full': first_full = logits
            metrics = first_metrics(first_full,logits)
            for key,value in metrics.items():
                difference = abs(float(value)-float(g[key])); max_metric_difference = max(max_metric_difference,difference)
                assert difference == 0., (path,key,difference)
            assert digest(logits.tolist()) == g['first_logits_sha256']
            assert digest(first_full.tolist()) == g['full_first_logits_sha256']
            assert len(g['hook_events']) == 4*g['length']
            assert all(e['non_target_rows_exact'] and (not e['applied'] or e['target_exact_zero']) for e in g['hook_events'])
            generations[condition].append(g)
            per_question.append(dict(task=sample['task'], question_id=sample['question_id'],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'], condition=condition,
                question=sample['question'], gold=sample['answer'], text=g['text'], prediction=g['prediction'],
                correct=g['correct'], correct_to_wrong=g['full_correct_to_wrong'], wrong_to_correct=g['full_wrong_to_correct'],
                token_ids=json.dumps(g['token_ids']), eos=g['eos'], truncated=g['truncated'], empty=g['empty'],
                generated_tokens=g['length'], first_top1_flip=g['first_top1_flip'], first_logp_loss=g['first_logp_loss'],
                full_first_token=g['full_first_token'], first_token=g['first_token'], sequence_changed=g['sequence_changed'],
                I_prefill_queries=g['query_layout']['counts']['I'], R_prefill_queries=g['query_layout']['counts']['R'],
                A_queries_including_decode=g['length'], intervened_query_layer_updates=g['intervened_query_layer_updates'],
                raw_logits_path=row['raw']['path'], raw_logits_sha256=row['raw']['sha256']))
            checkpoint_hashes[str(path.relative_to(OUT))] = sha256(path)
            raw_hashes[row['raw']['path']] = row['raw']['sha256']
    scores = np.array([[generations[c][i]['correct'] for c in CONDITIONS] for i in range(240)],dtype=float)
    draws = paired_bootstrap(samples,scores)
    comparisons = []
    for a,b in COMPARISONS:
        ia,ib = CONDITIONS.index(a),CONDITIONS.index(b)
        lo,hi = np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5])
        comparisons.append(dict(comparison=f'{a} - {b}', a=a, b=b, questions=240, image_clusters=240,
            delta_accuracy_pp=float(100*(scores[:,ia]-scores[:,ib]).mean()), ci95_low_pp=float(lo), ci95_high_pp=float(hi),
            b_correct_a_wrong=int(((scores[:,ib]==1)&(scores[:,ia]==0)).sum()),
            b_wrong_a_correct=int(((scores[:,ib]==0)&(scores[:,ia]==1)).sum()),
            bootstrap_repetitions=20000, interval='Ordinary paired exploratory image-cluster 95%'))
    summaries = [dict(condition=c, **summary(generations[c],generations['Full'])) for c in CONDITIONS]
    task_rows = []
    for task in TASKS:
        indices = [i for i,s in enumerate(samples) if s['task']==task]
        for c in CONDITIONS:
            task_rows.append(dict(task=task,condition=c,**summary([generations[c][i] for i in indices],
                                                                [generations['Full'][i] for i in indices])))
    csv_write(OUT/'per_question.csv',per_question)
    csv_write(OUT/'accuracy_summary.csv',summaries)
    csv_write(OUT/'task_accuracy.csv',task_rows)
    csv_write(OUT/'paired_comparisons.csv',comparisons)
    csv_write(OUT/'negative_tail_cases.csv',sorted([r for r in per_question if r['condition']!='Full'],
                                                  key=lambda r:r['first_logp_loss'],reverse=True)[:30])
    plot(summaries,comparisons,task_rows)
    apps = subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'],text=True).strip()
    gpu_table = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                         '--format=csv,noheader'],text=True).strip()
    our_pids = [read(path)['pid'] for path in OUT.glob('environment_*.json')]
    assert all(not Path(f'/proc/{pid}').exists() for pid in our_pids)
    write(OUT/'gpu_release.json',dict(our_gpu_processes_exited=True, pids=our_pids, physical_gpu=1,
        device_table=gpu_table, current_compute_processes=apps, currently_idle=not bool(apps)))
    for path,record in p['model_files'].items():
        assert sha256(path) == record['sha256'], 'Model file hash changed'
    validation = dict(passed=True, questions=240, conditions=4, checkpoints=960, no_missing_or_duplicate=True,
        image_clusters=240, task_counts={t:40 for t in TASKS}, gpu_smoke_sha256=sha256(OUT/'gpu_smoke.json'),
        cpu_validation_sha256=sha256(OUT/'cpu_validation.json'), metric_replay_max_abs_difference=max_metric_difference,
        raw_logits_verified=960, raw_arrays_on_github=False, model_full_sha256_unchanged=True,
        source_binding_unchanged=True, gpu_processes_exited=True, exploratory=True, noninferiority_claim=False,
        raw_hashes=raw_hashes, checkpoint_hashes=checkpoint_hashes)
    write(OUT/'validation.json',validation)
    report(summaries,comparisons,task_rows,generations,samples)
    artifacts = [f for f in OUT.rglob('*') if f.is_file() and f.name not in ['worker.lock','completion.json']]
    write(OUT/'completion.json',dict(complete=True, binding=binding, frozen_code_commit=p['code_commit'],
        analysis_seconds=time.perf_counter()-started,
        artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(artifacts)}))
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
    print('CPU analysis, model/file replay, GPU release and 960 checkpoints validated',flush=True)


def plot(summaries, comparisons, task_rows):
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    labels = CONDITIONS[1:]; colors=['#326b9b','#c34a4a','#27836c']
    fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
    for i,c in enumerate(labels):
        row=next(r for r in comparisons if r['a']==c and r['b']=='Full')
        v=row['delta_accuracy_pp']; lo=row['ci95_low_pp']; hi=row['ci95_high_pp']
        ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',color=colors[i],capsize=5,ms=7,lw=1.7)
        ax.annotate(f'{v:+.2f}',(i,v),xytext=(8,6),textcoords='offset points',fontsize=9)
    ax.axhline(0,color='#555555',lw=.8,ls='--'); ax.set_xticks(range(3),labels)
    ax.set_ylabel('Accuracy change vs Full (pp)');ax.set_title('L2-L5 Attention: Query-Role Sensitivity')
    ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'figure_query_roles.pdf');fig.savefig(OUT/'figure_query_roles.png',dpi=300);plt.close(fig)
    matrix=np.array([[next(r['delta_full_pp'] for r in task_rows if r['task']==t and r['condition']==c)
                      for c in labels] for t in TASKS]);limit=max(1.,float(abs(matrix).max()))
    fig,ax=plt.subplots(figsize=(6.3,4.2),layout='constrained')
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(3),labels);ax.set_yticks(range(6),TASKS)
    for i in range(6):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',
            color='white' if abs(matrix[i,j])>.62*limit else 'black')
    ax.set_title('Task-Level Accuracy Change vs Full');fig.colorbar(im,ax=ax,label='Accuracy change (pp)')
    fig.savefig(OUT/'figure_task_roles.pdf');fig.savefig(OUT/'figure_task_roles.png',dpi=300);plt.close(fig)


def report(summaries, comparisons, task_rows, generations, samples):
    by={r['condition']:r for r in summaries}
    r,i,a=by['R-Bypass'],by['I-Bypass'],by['A-Bypass']
    strongest=min(CONDITIONS[1:],key=lambda c:by[c]['delta_full_pp'])
    lines=['# Query-Role Attention Dormancy：L2-L5机制探索','',
        '状态：完整完成，240题、240个不同Image SHA256簇、六任务各40题，960次本机配对完整生成。',
        '固定历史3000题中的输入哈希子样本，不是独立确认；未按预测、正确性或敏感性筛选。','',
        '## 直接判断','',
        f"I/R/A相对Full的Accuracy点估计变化依次为{i['delta_full_pp']:+.3f}、{r['delta_full_pp']:+.3f}、{a['delta_full_pp']:+.3f} pp。",
        f'本批观察到最大净损害的角色是{strongest}。是否存在明确的角色差异，须同时阅读以下配对区间，不能只比较点估计或把跨零区间视为等价。','',
        '本轮能检验同一层段的功能必要性是否依赖Query角色；不能把“死亡区”当成已证实、与Query无关的网络属性。','',
        '|条件|正确/240|Accuracy|ΔFull pp|对→错|错→对|首Token翻转|完整Token序列变化|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in summaries:
        lines.append(f"|{v['condition']}|{v['correct']}/240|{v['accuracy_pct']:.3f}%|{v['delta_full_pp']:+.3f}|{v['correct_to_wrong']}|{v['wrong_to_correct']}|{v['first_top1_flip']}|{v['sequence_changed']}|")
    lines += ['', '## 预声明配对比较','', '|比较|ΔAccuracy pp|普通95%CI pp|基线对→干预错|基线错→干预对|',
              '|---|---:|---|---:|---:|']
    for v in comparisons:
        lines.append(f"|{v['comparison']}|{v['delta_accuracy_pp']:+.3f}|[{v['ci95_low_pp']:+.3f}, {v['ci95_high_pp']:+.3f}]|{v['b_correct_a_wrong']}|{v['b_wrong_a_correct']}|")
    lines += ['', '按Image SHA256整簇20,000次配对Bootstrap，seed=20261102；题目加权，普通探索性95%CI，无多重比较确认性声明。经验区间即使退化为[0,0]也不是总体安全保证，不作非劣效或等价宣称。','',
              '## 六类任务','', '|任务|Full|I-Bypass|R-Bypass|A-Bypass|','|---|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('|'+task+'|'+'|'.join(f"{next(v['accuracy_pct'] for v in task_rows if v['task']==task and v['condition']==c):.1f}%" for c in CONDITIONS)+'|')
    lines += ['', '## 首Token、负面尾部与生成行为','',
              '|条件|平均signed LogP loss|中位|P5/P95|EOS|截断|空答案|平均生成Token数|',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    for v in summaries:
        lines.append(f"|{v['condition']}|{v['first_logp_loss_mean']:+.6f}|{v['first_logp_loss_median']:+.6f}|{v['first_logp_loss_p5']:+.6f}/{v['first_logp_loss_p95']:+.6f}|{v['eos']}|{v['truncated']}|{v['empty']}|{v['mean_generated_tokens']:.2f}|")
    relay=[g['query_layout']['counts']['R'] for g in generations['Full']]
    lines += ['', 'LogP loss始终固定本机Full首Token，等于logP_Full−logP_intervened；允许负值，既不是Gold Loss，也不等于Accuracy下降。全部逐题损益和最严重30个首Token损害保存，不隐藏负值、空答案或截断。','',
        '## 科学问题与解释边界','',
        f"1. **Image Query的重要性：** I-Bypass改变{ i['sequence_changed']}条完整Token序列，损害{i['correct_to_wrong']}题、纠正{i['wrong_to_correct']}题。仅限L2-L5输出更新；不能推断视觉Token或所有视觉计算可移除。",
        f"2. **Relay Query的重要性：** R-Bypass损害{r['correct_to_wrong']}题、纠正{r['wrong_to_correct']}题。R包括问题、指令、模板和空白，不只是Question；关闭的是整个Attention更新，不是单独视觉读取。",
        f"3. **Answer Query的重要性：** A-Bypass损害{a['correct_to_wrong']}题、纠正{a['wrong_to_correct']}题；是否复现历史低敏感，应以实际损害和区间限定，不能以零净变化宣称安全。GPU Smoke与历史E条件逐步有效Logits和Token精确一致。",
        '4. **R是否比I/A更有损害：** 以上R−I和R−A是预声明同图配对比较；区间若完全为负，支持本批范围内R更有损害；否则证据不足。不能从点估计排序直接推广总体。',
        f'5. **Query数量混杂：** I固定576个Prefill Query；R为{min(relay)}–{max(relay)}个（均值{np.mean(relay):.2f}）；A每条件为1个P−1加该条件所有真实Decode Query，含EOS预测。Accuracy差异不是单位Query贡献，不能做简单数量归一化因果解释。',
        '6. **读写分离假说：** 若R损害明显而A损害较小，这是Query角色依赖的功能证据，与“Prefill构建信息、答案局部读取较廉价”相容；不是视觉记忆写入这一唯一机制的证明。文本处理、通用上下文整合及任务格式均可能参与。没有做中介分析、激活修补或定向来源消融。',
        '7. **未来真实旁路：** 本轮所有Q/K/V、Attention矩阵乘法、o_proj和MLP都实际执行。功能消融不能证明任何算子已可安全物理省略。即使未来针对某角色省略Attention输出路径，也必须保留被其他当前/未来Query读取的K/V、正确位置和因果Cache；I/R的KV尤其不能因其某次输出低敏感而直接删除。需要另行实现和验证，当前无速度、显存或FLOPs收益。','',
        '## 正确性与资源','',
        'CPU角色/层范围、Prefill/Decode、异常清理、独立Cache及输入哈希选样测试先通过；三道不同Prompt长度的7B真实Smoke再通过，才运行正式矩阵。Full与原生Hugging Face Greedy、Full重复推理、I/R/A No-op的全部有效Logits及Token精确一致，没有放宽容差。',
        '每次Hook验证目标行精确为0，直接作用外行逐位不变；这不要求后续层的非目标状态不受因果传播影响。MLP32层正常调用，原生Cache不被其他条件污染，模型参数版本和输入哈希不变，Hook退出和异常清理通过。',
        '960份首Token原始词表Logits在服务器资源目录，路径/SHA256记录于逐题检查点和validation.json；不声称GitHub包含这些大数组。全部指标从原始向量重算精确一致，模型文件末次完整SHA256复核通过。',
        f"GPU1耗时（含加载与Smoke）{read(OUT/'gpu_completion.json')['seconds_including_load_smoke']/60:.2f}分钟；峰值allocated {read(OUT/'gpu_completion.json')['memory']['peak_allocated_mib']:.1f} MiB。GPU进程已退出，释放状态见gpu_release.json。",
        f"冻结代码Commit：`{read(OUT/'protocol.json')['code_commit']}`。仅推送独立分支，不修改main；本轮结束，不自动扩样或追加条件。",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
