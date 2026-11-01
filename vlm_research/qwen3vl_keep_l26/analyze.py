import csv
import json

import numpy as np

from ..qwen3vl_early_windows.analyze import bootstrap
from .engine import CONDITIONS
from .run import OUT, read, sha256, verify, write


def csv_write(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    protocol=verify()
    samples=read(OUT/'sample_manifest.json')['samples']
    records=[]
    complete=[]
    for shard in range(3):
        execution=read(OUT/f'shard_{shard}/complete.json')
        assert execution['passed'] and execution['count']==80
        assert execution['protocol_sha256']==sha256(OUT/'protocol.json')
        complete.append(execution)
        assert len(list((OUT/f'shard_{shard}/checkpoints').glob('*.json')))==80
    assert len({r['code_commit'] for r in complete})==1
    for sample in samples:
        record=read(OUT/f'shard_{sample["shard"]}/checkpoints/{sample["task"]}__{sample["question_id"]}.json')
        assert record['protocol_sha256']==sha256(OUT/'protocol.json') and record['input_binding']==sample['input_binding']
        assert all(record[k]==sample[k] for k in ('task','question_id','image_id','image_sha256','question','answer'))
        assert list(record['conditions'])==list(CONDITIONS)
        for name,layers in CONDITIONS.items():
            r=record['conditions'][name]
            assert r['masked_layers']==layers and r['decode_mask_calls']==0 and r['independent_fresh_cache']
            assert len(r['mask_audits'])==len(layers)
            assert all(a['image_probability_max']==0 and a['mask_changed_entries']==sample['layout']['image_tokens'] for a in r['mask_audits'].values())
            assert r['generated_tokens']==len(r['token_ids'])==len(r['forward_calls'])
            assert r['initial_cache_lengths']==[0]*36
            assert r['correct']==int(r['prediction']==sample['answer'])
        records.append(record)
    assert len(records)==240 and len({(r['task'],r['question_id']) for r in records})==240
    names=list(CONDITIONS)
    scores=np.array([[r['conditions'][c]['correct'] for c in names] for r in records])
    intervals=bootstrap((scores[:,1:]-scores[:,:1])*100.,[r['image_sha256'] for r in records],repetitions=protocol['bootstrap_repetitions'])[:2]
    summary,flat,tasks=[],[],[]
    for j,name in enumerate(names):
        results=[r['conditions'][name] for r in records]
        summary.append(dict(condition=name,n=240,correct=int(scores[:,j].sum()),accuracy_percent=float(scores[:,j].mean()*100),
            delta_full_pp=float((scores[:,j]-scores[:,0]).mean()*100),
            paired_ci95_low_pp=float(intervals[0,j-1]) if j else 0.,paired_ci95_high_pp=float(intervals[1,j-1]) if j else 0.,
            correct_to_wrong=int(((scores[:,0]==1)&(scores[:,j]==0)).sum()),wrong_to_correct=int(((scores[:,0]==0)&(scores[:,j]==1)).sum()),
            first_token_flips=sum(r['first_top1_flip'] for r in results),first_token_flip_percent=float(np.mean([r['first_top1_flip'] for r in results])*100),
            mean_first_token_logp_loss=float(np.mean([r['first_logp_loss'] for r in results])),
            eos=sum(r['eos'] for r in results),truncated=sum(r['truncated'] for r in results),empty=sum(r['empty'] for r in results)))
        for record,result in zip(records,results):
            full=record['conditions']['Full']
            flat.append({**{k:record[k] for k in ('task','question_id','image_id','image_sha256','question','answer','input_binding','physical_gpu')},
                'condition':name,**{k:result[k] for k in ('text','prediction','correct','first_token_id','full_first_token_id','first_top1_flip','first_logp_loss','generated_tokens','eos','truncated','empty','seconds')},
                'token_ids':json.dumps(result['token_ids']),'full_correct':full['correct'],
                'correct_to_wrong':int(full['correct'] and not result['correct']),'wrong_to_correct':int(not full['correct'] and result['correct'])})
    for task in sorted({r['task'] for r in records}):
        subset=[r for r in records if r['task']==task]
        assert len(subset)==40
        for name in names:
            correct=sum(r['conditions'][name]['correct'] for r in subset)
            tasks.append(dict(task=task,condition=name,n=40,correct=correct,accuracy_percent=correct/40*100))
    csv_write('accuracy_summary.csv',summary)
    csv_write('task_accuracy.csv',tasks)
    csv_write('paired_comparisons.csv',[{'comparison':r['condition']+' - Full',**r} for r in summary[1:]])
    csv_write('per_question.csv',flat)
    write(OUT/'integrity.json',dict(passed=True,questions=240,generations=720,mask_conditions=480,
        unique_image_clusters=len({r['image_sha256'] for r in records}),all_masks_verified=True,cache_isolation_verified=True,
        formal_wall_seconds_max_shard=max(r['seconds'] for r in complete),execution=complete,
        code_commit=complete[0]['code_commit'],protocol_sha256=sha256(OUT/'protocol.json')))
    lines=['# Qwen3-VL：仅保留L26视觉读取的完整答案验证','',
        '复用上一轮240题（六类各40）及全部动态输入绑定；未重新采样。模型Revision、BF16/Eager、官方Chat Template与上一轮一致。仅原始Prefill P−1施加Image来源Hard Mask；其他Prefill Query、后续Decode和DeepStack全部原生。三个条件各自全新空KV Cache、Greedy最多32 Token，不复用扫描实验作为生成结果。','',
        '## 总体结果','','|条件|正确/240|Accuracy|ΔFull (pp)|配对95%CI (pp)|原对→错|原错→对|首Token翻转|',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["condition"]}|{r["correct"]}|{r["accuracy_percent"]:.3f}%|{r["delta_full_pp"]:+.3f}|[{r["paired_ci95_low_pp"]:.3f}, {r["paired_ci95_high_pp"]:.3f}]|{r["correct_to_wrong"]}|{r["wrong_to_correct"]}|{r["first_token_flips"]}|')
    lines+=['','## 六任务结果','','|任务|Full|Keep L26|All OFF|','|---|---:|---:|---:|']
    for task in sorted({r['task'] for r in records}):
        lines.append('|'+task+'|'+'|'.join(f'{r["accuracy_percent"]:.1f}%' for r in tasks if r['task']==task)+'|')
    keep=summary[1]
    lines+=['','## 结论与边界','',
        f'Keep L26相对Full变化{keep["delta_full_pp"]:+.3f} pp，配对95%CI [{keep["paired_ci95_low_pp"]:.3f}, {keep["paired_ci95_high_pp"]:.3f}]；损害{keep["correct_to_wrong"]}题、纠正{keep["wrong_to_correct"]}题。首Token翻转与完整答案正确率分别报告，不能相互替代。',
        '本批题已用于挑选L26，因此本轮是同队列功能验证，不是独立确认。CI采用Image SHA256整簇配对Bootstrap（100,000次，seed=20261101），保留题目加权估计；主要比较为Keep L26−Full，All OFF为描述性参照。未预设非劣效/等价检验，CI含0不等于无损。',
        'Keep L26只表示P−1在L26允许直接读取Image Key，不代表整个模型只需一层视觉融合。其他文本Query仍可读图、P−1残差仍可能携带视觉信息，Decode完全原生；All OFF也不是纯文本模型或删除图像。',
        '','## 正确性与执行','',
        '复用已验证的动态Image Key与Softmax前−inf实现。每个目标层均直接检查原生Attention概率为0；所有Mask在Prefill结束即移除，Decode无Mask调用。各卡Smoke对比原生Full、监控Full和No-op的生成Token及首Token Logits逐元素精确一致，检查非P−1 Query、DeepStack、异常Hook清理与36层Cache长度。每个生成条件使用独立原生DynamicCache，初始36层长度均0。输入和权重版本/存储不变。',
        f'正式运行最长Shard {max(r["seconds"] for r in complete)/60:.2f}分钟；物理GPU0/2/3各80题，未使用GPU1。冻结代码Commit `{complete[0]["code_commit"]}`。',
        'EOS/截断/空答案：'+str([(r['condition'],r['eos'],r['truncated'],r['empty']) for r in summary])+'。',
        '逐题生成文本、Token ID、正确性和翻转见per_question.csv；完整Mask/Cache审计见各Shard检查点。GPU释放另见gpu_release.json。本轮到此结束，不新增其他条件。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
