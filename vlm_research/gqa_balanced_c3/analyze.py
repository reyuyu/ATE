"""Recompute exact-match accuracy from complete, hashed paired generations."""
import csv
import gzip
import json
import sqlite3

import numpy as np

from .common import *


def write_rows(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(first),lineterminator='\n')
        writer.writeheader()
        for row in (first,):
            writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def bootstrap(values, image_ids):
    unique, inverse = np.unique(image_ids,return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse,weights=values[:,i]) for i in range(values.shape[1])],axis=1)
    rng = np.random.default_rng(SEED)
    draws = np.empty((20000,values.shape[1]))
    for start in range(0,20000,64):
        ix = rng.integers(len(unique),size=(min(64,20000-start),len(unique)))
        draws[start:start+len(ix)] = sums[ix].sum(1)/sizes[ix].sum(1)[:,None]*100
    return draws


def main():
    freeze = freeze_check(); binding = digest(freeze)
    manifest = samples(); expected = {s['question_id']:s for s in manifest}
    partition = read(OUT/'partition.json')
    assert all(read(OUT/f'worker{r}/completion.json')['status']=='COMPLETE' for r in range(2))
    records = {}
    for rank in range(2):
        path = OUT/f'worker{rank}/results.sqlite'
        db = sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        for row in db.execute('SELECT payload,sha256 FROM results'):
            r = unpack(row); key = (r['question_id'],r['condition'])
            s = expected[key[0]]
            assert key not in records and key[1] in LABELS
            assert r['rank']==rank==partition['image_shard'][s['image_sha256']]
            assert r['binding']==binding and r['sample_sha256']==digest(s)
            assert r['image_sha256']==s['image_sha256']
            assert r['score']==score(s,r['generation']['text'])
            records[key] = r
        db.close()
    assert len(records)==len(manifest)*2==264124
    values = np.array([[records[s['question_id'],label]['score'] for label in LABELS] for s in manifest])
    clusters = np.array([s['image_sha256'] for s in manifest])
    draws = bootstrap(values,clusters)
    means = values.mean(0)*100
    summary = []
    for i,label in enumerate(LABELS):
        gens = [records[s['question_id'],label]['generation'] for s in manifest]
        low,high = np.quantile(draws[:,i],[.025,.975])
        summary.append(dict(condition=label,questions=len(manifest),image_clusters=len(set(clusters)),
            correct=int(values[:,i].sum()),accuracy_percent=float(means[i]),
            accuracy_ci_low=float(low),accuracy_ci_high=float(high),eos=sum(g['eos'] for g in gens),
            truncated=sum(g['truncated'] for g in gens),empty=sum(g['empty'] for g in gens),
            repeated=sum(g['repeated'] for g in gens),mean_generated_tokens=float(np.mean([len(g['token_ids']) for g in gens]))))
    low,high = np.quantile(draws[:,1]-draws[:,0],[.025,.975])
    paired = dict(comparison=LABELS[1]+' minus '+LABELS[0],questions=len(manifest),
        delta_accuracy_pp=float(means[1]-means[0]),ci95_low_pp=float(low),ci95_high_pp=float(high),
        correct_to_wrong=int(((values[:,0]==1)&(values[:,1]==0)).sum()),
        wrong_to_correct=int(((values[:,0]==0)&(values[:,1]==1)).sum()),
        first_token_flips=sum(records[s['question_id'],LABELS[0]]['generation']['first_token_id'] !=
            records[s['question_id'],LABELS[1]]['generation']['first_token_id'] for s in manifest),
        generated_sequence_changes=sum(records[s['question_id'],LABELS[0]]['generation']['token_ids'] !=
            records[s['question_id'],LABELS[1]]['generation']['token_ids'] for s in manifest))
    def rows():
        for s in manifest:
            for label in LABELS:
                r = records[s['question_id'],label]; g = r['generation']
                yield dict(question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                    structural_type=s['types']['structural'],semantic_type=s['types']['semantic'],
                    condition=label,gold=s['answer'],text=g['text'],prediction=prediction(g['text']),
                    accuracy=r['score'],token_ids=json.dumps(g['token_ids']),eos=g['eos'],
                    truncated=g['truncated'],empty=g['empty'],repeated=g['repeated'],
                    first_logits_sha256=g['first_logits_sha256'])
    write_rows(OUT/'per_question.csv',rows())
    write_rows(OUT/'accuracy_summary.csv',summary)
    write_rows(OUT/'paired_comparisons.csv',[paired])
    by_type = []
    for field in ('structural','semantic'):
        for t in sorted({s['types'][field] for s in manifest}):
            ix = np.array([s['types'][field]==t for s in manifest])
            for i,label in enumerate(LABELS):
                by_type.append(dict(grouping=field,group=t,condition=label,questions=int(ix.sum()),
                    accuracy_percent=float(values[ix,i].mean()*100)))
    write_rows(OUT/'task_accuracy.csv',by_type)
    negatives = []
    for s,v in zip(manifest,values):
        if v[0]==1 and v[1]==0:
            negatives.append(dict(question_id=s['question_id'],image_id=s['image_id'],image_path=s['image_path'],
                question=s['question'],gold=s['answer'],C3=records[s['question_id'],LABELS[0]]['generation']['text'],
                reaccess=records[s['question_id'],LABELS[1]]['generation']['text']))
    if negatives:
        write_rows(OUT/'negative_cases.csv',negatives)
    atomic(OUT/'validation.json',dict(status='PASS',at=now(),records=len(records),questions=len(manifest),
        every_checkpoint_digest_verified=True,paired_same_gpu=True,no_duplicates_or_missing=True,
        full_official_val_question_set=True,all_actual_eos_and_exceptions_retained=True,
        accuracy_recomputed_from_official_answers=True,bootstrap_replicates=20000,
        gpu_smoke=[read(OUT/f'worker{r}/smoke.json')['status'] for r in range(2)]))
    report = f'''# 完整GQA val_balanced：C3-V2 vs L11–L24回看

状态：COMPLETE。共132,062题、10,234个Image ID、{len(set(clusters)):,}个图片SHA簇，264,124条新生成。

| 条件 | Accuracy | 95%图片簇Bootstrap CI |
|---|---:|---:|
| C3-V2 | {means[0]:.4f}% | [{summary[0]['accuracy_ci_low']:.4f}, {summary[0]['accuracy_ci_high']:.4f}]% |
| C3-V2 + L11–L24回看 | {means[1]:.4f}% | [{summary[1]['accuracy_ci_low']:.4f}, {summary[1]['accuracy_ci_high']:.4f}]% |

回看相对C3-V2：{paired['delta_accuracy_pp']:+.4f}pp，配对95%CI [{low:+.4f}, {high:+.4f}]pp。
原对→错 {paired['correct_to_wrong']:,}题；原错→对 {paired['wrong_to_correct']:,}题。

## 固定算法
两条件均维持L0–L19原生R视觉写入、L2–L5仅P−1和Decode的物理Attention跳算；MLP/Residual/Norm不变。
C3-V2在L20前移除视觉Hidden，答案阶段无Image KV。
回看条件仅在L11–L24让P−1及真实Decode读图，保留这14层Image KV，视觉Hidden在L25前移除。
L20–L24的普通R依然不能新增读图，不是完整Native Prefill，也不是最近TextVQA的Native-Prefill Decode-only条件。

## 评分与边界
Greedy、max_new_tokens=32、自然EOS、H1末行LM Head、独立语言KV Cache。
沿用固定LLaVA GQA输出转换及官方短答案严格匹配；不使用候选LogP代理Accuracy。
20,000次按Image SHA整簇配对Bootstrap、问题等权；类型分组为描述性结果。
这是完整验证集val_balanced，**不是隐藏test/testdev**。图片复用本地MMEB版本，未证明与Stanford原图分辨率完全相同，不能直接当作官方排行榜复现。
没有复用历史3000题分数，也没有运行Native Full质量矩阵或性能测试。本轮仅回答两个固定C3条件的全验证集Accuracy。

详细结果见accuracy_summary.csv、paired_comparisons.csv、task_accuracy.csv及per_question.csv；负面案例全部保留。
源码、模型、输入、数据协议哈希见execution_freeze.json；CPU/GPU验证及资源释放记录保存在同目录。
'''
    (OUT/'report_zh.md').write_text(report)
    atomic(OUT/'completion.json',dict(status='COMPLETE',at=now(),questions=len(manifest),generations=len(records),
        accuracy=summary,comparison=paired,report_sha256=sha(OUT/'report_zh.md')))
    for rank in range(2):
        src = OUT/f'worker{rank}/results.sqlite'
        with src.open('rb') as stream, gzip.open(src.with_suffix('.sqlite.gz'),'wb',compresslevel=6) as target:
            import shutil
            shutil.copyfileobj(stream,target,1024*1024)
    atomic(OUT/'result_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ('result_hashes.json',) and p.suffix not in ('.sqlite','-wal','-shm')})
    print(json.dumps(dict(summary=summary,comparison=paired),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
