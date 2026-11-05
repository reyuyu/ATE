"""Separate full500 and nested199 estimates; never mix their baselines."""
import gzip
import hashlib
import tarfile

import numpy as np

from .common import *
from .run import verify_freeze
from ..fastp2_bypass_writer.common import hardware


def main():
    frozen = verify_freeze()
    assert read(OUT / 'driver_exit.json')['returncodes'] == [0, 0]
    assert all(read(OUT / f'worker{r}/completion.json')['status'] == 'COMPLETE' for r in range(2))
    samples, partition = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
    subset = set(partition['subset199_keys'])
    assert len(samples) == 500 and len(subset) == 199
    reused = {(r['key'], r['condition']): r for r in read(OUT / 'source_manifest.json')['reused_records']}
    prior_layers = (11, 12) if FIRST_LAYER == 10 else (11,) if FIRST_LAYER == 12 else ()
    priors = {f'Decode-L{i}-L24': Path(f'/data/lk/ATE-worktrees/textvqa-decode-l{i}-l24-500/vlm_research/results/textvqa_decode_l{i}_l24_500')
              for i in prior_layers}
    extra = tuple(priors)
    prior_hashes = {c: read(p / 'result_hashes.json')['checkpoint_sha256'] for c, p in priors.items()}
    prior_freezes = {c: read(p / 'execution_freeze.json') for c, p in priors.items()}
    for label, prior in priors.items():
        assert prior_freezes[label]['devices'] == frozen['devices']
        assert read(prior / 'partition.json')['image_shard'] == partition['image_shard']
    baselines = BASELINES+extra
    labels = baselines+(CONDITION,)
    records, hashes, rows = {}, {}, []
    for key in sorted(samples):
        sample = samples[key]
        full = historical(key, 'Native')
        for label in labels:
            path = (checkpoint(key) if label == CONDITION else
                    priors[label] / 'checkpoints' / (key+'.json') if label in extra else historical_path(key, label))
            value = load(path)
            if label in extra:
                assert sha(path) == prior_hashes[label][str(path.relative_to(priors[label]))]
                assert value['sources'] == prior_freezes[label]['sources']
            elif label != CONDITION:
                assert sha(path) == reused[key, label]['sha256']
            else:
                assert value['sources'] == frozen['sources']
                assert value['rank'] == partition['image_shard'][sample['image_sha256']]
                hashes[str(path.relative_to(OUT))] = sha(path)
            assert value['condition'] == label and value['key'] == key
            assert value['sample_sha256'] == digest(sample)
            assert value['input_binding'] == full['input_binding']
            g = value['generation']
            assert value['score'] == score(sample, g['text'])
            assert 1 <= len(g['token_ids']) <= 32
            assert g['first_logits_sha256'] == full['generation']['first_logits_sha256']
            records[key, label] = value
            rows.append(dict(key=key, question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], question=sample['question'],
                gold_answers=sample['answer'], in_previous199=key in subset, condition=label,
                history_reused=label != CONDITION, answer=g['text'], score=value['score'],
                token_ids=g['token_ids'], first_logits_sha256=g['first_logits_sha256'],
                input_binding_sha256=digest(value['input_binding']), eos=g['eos'], empty=g['empty'],
                truncated=g['truncated'], repeated=g['repeated'], content_tokens=g['content_tokens']))
    assert len(rows) == 500*len(labels) and len(hashes) == 500
    assert {str(p.relative_to(OUT)) for p in (OUT / 'checkpoints').glob('*.json')} == set(hashes)
    write_csv(OUT / 'per_question.csv', rows)
    summaries, comparisons, cases = [], [], []
    for cohort, keys in [('full500', sorted(samples)), ('previous199', sorted(subset))]:
        images = sorted({samples[k]['image_sha256'] for k in keys})
        cluster_index = {h: i for i, h in enumerate(images)}
        membership = np.array([cluster_index[samples[k]['image_sha256']] for k in keys])
        sizes = np.bincount(membership)
        values = np.array([[records[k, label]['score'] for label in labels] for k in keys])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        sums = np.zeros((len(images), len(labels)))
        np.add.at(sums, membership, values)
        weights = np.random.default_rng(SEED).multinomial(len(images),
            np.full(len(images), 1/len(images)), size=20000)
        boot = weights @ sums/(weights @ sizes)[:, None]*100
        means = values.mean(0)*100
        for j, label in enumerate(labels):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            gs = [records[k, label]['generation'] for k in keys]
            summaries.append(dict(cohort=cohort, questions=len(keys), image_clusters=len(images),
                condition=label, accuracy=float(means[j]), ci_low=float(lo), ci_high=float(hi),
                eos=sum(g['eos'] for g in gs), empty=sum(g['empty'] for g in gs),
                truncated=sum(g['truncated'] for g in gs), repeated=sum(g['repeated'] for g in gs)))
        for b, baseline in enumerate(baselines):
            delta = values[:, -1]-values[:, b]
            lo, hi = np.percentile(boot[:, -1]-boot[:, b], [2.5, 97.5])
            comparisons.append(dict(cohort=cohort, questions=len(keys), image_clusters=len(images),
                candidate=CONDITION, baseline=baseline, delta_accuracy_pp=float(means[-1]-means[b]),
                ci_low=float(lo), ci_high=float(hi),
                soft_score_increased=int(np.sum(delta > 1e-10)), soft_score_decreased=int(np.sum(delta < -1e-10)),
                sequence_changes=sum(records[k, CONDITION]['generation']['token_ids'] != records[k, baseline]['generation']['token_ids'] for k in keys),
                first_token_flips=sum(records[k, CONDITION]['generation']['token_ids'][0] != records[k, baseline]['generation']['token_ids'][0] for k in keys)))
            if cohort == 'full500':
                for i, key in enumerate(keys):
                    if abs(delta[i]) > 1e-10:
                        cases.append(dict(key=key, question=samples[key]['question'],
                            image_sha256=samples[key]['image_sha256'], in_previous199=key in subset,
                            baseline=baseline, candidate=CONDITION,
                            baseline_answer=records[key, baseline]['generation']['text'],
                            candidate_answer=records[key, CONDITION]['generation']['text'],
                            baseline_score=float(values[i, b]), candidate_score=float(values[i, -1]),
                            score_change=float(delta[i]), effect='recovery' if delta[i] > 0 else 'harm'))
    write_csv(OUT / 'accuracy_summary.csv', summaries)
    write_csv(OUT / 'paired_comparisons.csv', comparisons)
    write_csv(OUT / 'harm_recovery_cases.csv', cases)
    write_csv(OUT / 'negative_cases.csv', [r for r in cases if r['effect'] == 'harm'])
    atomic(OUT / 'result_hashes.json', dict(count=500, checkpoint_sha256=hashes))
    pids = read(OUT / 'driver_pids.json')['workers']
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    atomic(OUT / 'gpu_release_audit.json', dict(at=now(), workers_exited=True, task_pids=pids,
        resources=hardware(), workers=[read(OUT / f'worker{r}/release.json') for r in range(2)]))
    smoke = [read(OUT / f'worker{r}/smoke.json') for r in range(2)]
    assert all(s['status'] == 'PASS' and len(s['checks']) == 3 for s in smoke)
    atomic(OUT / 'validation.json', dict(status='PASS', CPU_tests=14,
        GPU_smoke=[s['status'] for s in smoke], new_answers=500, reused_answers=500*len(baselines),
        first_logits_exact_native_all=True, no_missing_or_duplicate_records=True,
        inference_sources=frozen['sources'], analysis_source_sha256=sha(Path(__file__)),
        performance_requests=0))
    atomic(OUT / 'completion.json', dict(at=now(), status='COMPLETE_QUALITY_ONLY',
        new_answers=500, reused_answers=500*len(baselines), subset199_is_nested_not_additional_generation=True,
        code_commit=frozen['code_commit'], performance_requests=0,
        formal_quality_wall_seconds=max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2))))
    lines = [f'# TextVQA：Decode L{FIRST_LAYER}–L24完整答案验证', '',
        '只新增一个条件、500条完整Greedy答案；原500题全部保留，另取其中先前冻结的199题子集单独汇总，不重复生成、不混用分母。', '',
        '## 固定协议',
        f'- Native Prefill/P−1在全部32层正常；只限制真实增量Decode的原始Image Key访问，开放L{FIRST_LAYER}–L24（含两端，共{len(KEEP)}层）。',
        '- 所有Decode文本Attention、MLP、Residual、RoPE和独立真实KV保持原生；不是Attention输出置零，不做物理退出。',
        '- LLaVA-1.5-7B固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager、Transformers4.49.0、H1 last-row、原Prompt、Greedy max32、自然EOS。',
        f'- 官方十答案TextVQA共识软评分。{", ".join(baselines)}共{500*len(baselines)}条历史记录经输入、源码及首Logits绑定复用，运行前同时核验固定权重。',
        '- 14项CPU测试、GPU0/1各3题Smoke通过；Native/No-op严格一致，Mask实际概率为0，独立Dense参考、KV及Hook验证通过。', '',
        '## 得分：分别列出两种样本范围',
        '| 条件 | 原500题 | 其中的原199题子集 |', '|---|---:|---:|']
    lookup = {(r['cohort'], r['condition']): r for r in summaries}
    for label in labels:
        lines.append(f"| {label} | {lookup['full500', label]['accuracy']:.2f}% | {lookup['previous199', label]['accuracy']:.2f}% |")
    lines += ['', '## 同题配对变化',
        '按Image SHA整簇20,000次配对Bootstrap，固定seed 20261106；点估计问题等权。普通95%CI仅作探索描述，不宣称无损或非劣效。',
        f'| 范围 | L{FIRST_LAYER}–L24相对 | ΔAccuracy pp | 95%CI | 软分上升/下降题数 |',
        '|---|---|---:|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['cohort']} | {r['baseline']} | {r['delta_accuracy_pp']:+.2f} | [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['soft_score_increased']}/{r['soft_score_decreased']} |")
    lines += ['', f'- `per_question.csv`保存全部{500*len(labels)}条新旧答案与Token；`harm_recovery_cases.csv`保留全部逐题增减。',
        '- `accuracy_summary.csv`记录EOS、空答案、重复与截断，不剔除失败或负面样本。',
        '- 没有性能测试、额外层组合、物理优化或训练。GPU0/1 Worker已退出并释放资源，本轮到此停止。', '',
        '## 复现', '```bash', f'export ATE_TEXTVQA_DECODE_FIRST_LAYER={FIRST_LAYER}',
        'python -m vlm_research.textvqa_decode_l11_l24.run --prepare',
        'python -m vlm_research.textvqa_decode_l11_l24.run --freeze',
        'python -m vlm_research.textvqa_decode_l11_l24.run --driver',
        'python -m vlm_research.textvqa_decode_l11_l24.analyze', '```', '']
    (OUT / 'report_zh.md').write_text('\n'.join(lines))
    for label, prior in priors.items():
        atomic(OUT / (label.lower().replace('decode-', 'prior_').replace('-', '_')+'_binding.json'), dict(questions=500,
            execution_freeze_sha256=sha(prior / 'execution_freeze.json'),
            result_hashes_sha256=sha(prior / 'result_hashes.json'),
            checkpoint_sha256=prior_hashes[label], all_inputs_and_first_logits_match=True,
            identical_physical_devices_and_image_assignment=True))
    archive = OUT / 'raw_checkpoints.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(OUT / 'checkpoints', arcname='checkpoints')
        for rank in range(2):
            stream.add(OUT / f'worker{rank}/smoke.json', arcname=f'worker{rank}/smoke.json')
    with tarfile.open(archive, 'r:gz') as stream:
        for filename, expected in hashes.items():
            assert hashlib.sha256(stream.extractfile(filename).read()).hexdigest() == expected
    with gzip.open(OUT / 'sample_manifest.json.gz', 'wb') as stream:
        stream.write((OUT / 'sample_manifest.json').read_bytes())
    atomic(OUT / 'packaging.json', dict(archive_sha256=sha(archive),
        manifest_sha256=sha(OUT / 'sample_manifest.json'), gzip_sha256=sha(OUT / 'sample_manifest.json.gz'),
        archive_members_verified=True))
    print({'accuracy': summaries, 'paired': comparisons}, flush=True)


if __name__ == '__main__':
    main()
