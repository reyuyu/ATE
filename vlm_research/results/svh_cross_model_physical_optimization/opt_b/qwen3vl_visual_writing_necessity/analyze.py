"""Independent answer recalculation, clustered paired statistics, and paper figures."""
import ast
import csv
import json
from pathlib import Path
import tarfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import (ROOT, OUT, MODEL, CONDITIONS, SEED, read, seal, key, verify,
    checkpoint, digest, sha256, writer_layers)


def csv_file(name, rows):
    path = OUT / name
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(values, images, repetitions=20000, seed=SEED):
    """Image-cluster pairs, with questions equally weighted in every resample."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    _, inverse = np.unique(images, return_inverse=True)
    k = int(inverse.max())+1
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, values.shape[1]))
    for start in range(0, repetitions, 128):
        stop = min(start+128, repetitions)
        weights = rng.multinomial(k, np.full(k, 1/k), size=stop-start)
        draws[start:stop] = (weights @ sums)/(weights @ counts)[:, None]
    assert np.isfinite(draws).all()
    return draws


def audit():
    from tokenizers import Tokenizer
    p = verify()
    reference = read(OUT / 'llava_reference_summary.json')
    reference_blob = __import__('subprocess').check_output(['git', 'show', reference['commit']+
        ':vlm_research/results/gqa3000_visual_writing_necessity/condition_summary.csv'], cwd=ROOT)
    assert __import__('hashlib').sha256(reference_blob).hexdigest() == reference['sha256']
    samples = read(OUT / 'sample_manifest.json')['samples']
    specs = read(OUT / 'input_bindings.json')
    shards = read(OUT / 'shards.json')['keys']
    scorer = read(OUT / 'official_gqa_prediction.json')
    assert sha256(scorer['path']) == scorer['sha256']
    node = ast.parse(scorer['function_source']).body[0]
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), scorer['path'], 'exec'), scope)
    official_prediction = scope['gqa_prediction']
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    eos = read(MODEL / 'generation_config.json')['eos_token_id']
    eos = [eos] if isinstance(eos, int) else eos
    by = {key(s): s for s in samples}
    records, hashes, flat, harms = {}, [], [], []
    image_sets = [{by[k]['image_sha256'] for k in shard} for shard in shards]
    assert image_sets[0].isdisjoint(image_sets[1])
    max_row_error = 0.
    for shard in range(2):
        done = read(OUT / f'shard{shard}/complete.json')
        smoke = read(OUT / f'shard{shard}/smoke.json')
        assert done['passed'] and smoke['passed'] and done['binding'] == smoke['binding'] == digest(p)
        assert done['generations'] == len(shards[shard])*38
        paths = sorted((OUT / f'shard{shard}/checkpoints').glob('*/*.json'))
        assert len(paths) == done['generations']
        for path in paths:
            r = read(path)
            identifier = key(r)
            sample, spec, c = by[identifier], specs[identifier], r['condition']
            assert identifier in shards[shard] and c in CONDITIONS and (identifier, c) not in records
            assert r['binding'] == digest(p) and r['code_commit'] == p['code_commit']
            assert r['gpu_uuid'] == p['gpus'][shard]['uuid']
            assert r['sample_sha256'] == digest(sample) and r['input_binding'] == spec['binding']
            assert r['image_sha256'] == sample['image_sha256'] and r['gold'] == sample['answer']
            g = r['generation']
            assert digest(g) == r['generation_sha256']
            assert g['writer_layers'] == writer_layers(c) and g['P1_Decode_image_off']
            assert g['logits_to_keep'] == 1 and g['independent_native_cache'] and g['cache_initial_empty']
            assert g['all_logits_finite'] and 1 <= len(g['token_ids']) <= 32
            # Re-decode tokens and re-execute the pinned official answer parser.
            text = tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
            assert text == g['text'], 'Independent Token IDs-to-text reconstruction failed'
            prediction = official_prediction(text)
            assert prediction == g['prediction'] and g['correct'] == int(prediction == sample['answer'].lower())
            assert g['eos'] == (g['token_ids'][-1] in eos)
            assert g['truncated'] == (len(g['token_ids']) == 32 and not g['eos'])
            assert g['first_token_id'] == g['token_ids'][0] and g['generated_tokens'] == len(g['token_ids'])
            assert g['decode_queries'] == len(g['token_ids'])-1
            assert all(v == spec['layout']['input_length']+len(g['token_ids'])-1 for v in g['cache_lengths'])
            a = g['source_audit']
            assert a['steps'] == len(g['token_ids']) and a['layer_events'] == 36*a['steps']
            assert a['blocked_probability_max'] == 0 and a['row_sum_max_error'] <= float(2**-7)
            assert a['non_target_mask_exact'] and a['BOS_Self_legal'] and a['hook_cleanup'] and not a['noop']
            image_count, n = len(spec['roles']['I']), spec['layout']['input_length']
            assert a['prefill_blocked_connections_per_layer'] == [image_count*(1+(0 if i in writer_layers(c) else len(spec['roles']['R']))) for i in range(36)]
            assert a['decode_blocked_connections_per_layer'] == [image_count]*36
            for t, position in enumerate(a['actual_positions']):
                assert position['step'] == t and position['shape'] == ([3, 1, 1] if t else [3, 1, n])
                assert position['cache_position'] == ([n+t-1] if t else [0, n-1])
            max_row_error = max(max_row_error, a['row_sum_max_error'])
            records[identifier, c] = r
            hashes.append(dict(path=str(path.relative_to(OUT)), sha256=sha256(path)))
        print('AUDIT', shard, len(paths), 'records independently rescored', flush=True)
    assert len(records) == 114000
    for sample in samples:
        identifier = key(sample)
        base = records[identifier, 'All-Writer']
        control = records[identifier, 'Drop-L35']
        assert control['structural_negative_control_exact']
        assert control['effective_logits_sha256'] == base['effective_logits_sha256']
        assert control['generation']['token_ids'] == base['generation']['token_ids']
        for c in CONDITIONS:
            r, full = records[identifier, c], base['generation']
            g = r['generation']
            assert g['first_token_flip'] == int(g['first_token_id'] != full['first_token_id'])
            row = dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], gold=sample['answer'], condition=c,
                prediction=g['prediction'], text=g['text'], token_ids=json.dumps(g['token_ids']), correct=g['correct'],
                All_Writer_correct=full['correct'], correct_to_wrong=int(full['correct'] and not g['correct']),
                wrong_to_correct=int(not full['correct'] and g['correct']), first_token_flip=g['first_token_flip'],
                text_changed=int(g['text'] != full['text']), eos=g['eos'], truncated=g['truncated'], empty=g['empty'],
                repeated_trigram=g['repeated_trigram'], generated_tokens=g['generated_tokens'], gpu_uuid=r['gpu_uuid'],
                input_binding=r['input_binding'], generation_sha256=r['generation_sha256'])
            flat.append(row)
            if row['correct_to_wrong']:
                harms.append(dict(row, All_Writer_text=full['text']))
    csv_file('per_question.csv', flat)
    csv_file('harm_cases.csv', harms)
    seal(OUT / 'checkpoint_hashes.json', hashes)
    seal(OUT / 'validation.json', dict(passed=True, generations=114000, questions=3000, image_clusters=2409,
        independently_decoded_and_officially_rescored=114000, Drop_L35_exact_controls=3000,
        strict_dual_GPU_smoke=True, same_image_same_GPU=True, raw_record_sha256_verified=True,
        masked_probability_max=0., row_sum_max_error=max_row_error, no_tolerance_relaxation=True,
        NaN_Inf_OOM=False, Gold_NLL_not_computed=True, binding=digest(p)))
    return samples, records


def summarize(samples, records, subset, bootstrap=True):
    acc = np.array([[records[key(s), c]['generation']['correct'] for c in CONDITIONS] for s in samples])
    delta = (acc-acc[:, [0]])*100.
    if bootstrap:
        draws = paired_bootstrap(delta, [s['image_sha256'] for s in samples])
        ordinary = np.quantile(draws, [.025, .975], axis=0)
        adjusted = np.quantile(draws, [.025/36, 1-.025/36], axis=0)
        np.savez_compressed(OUT / f'bootstrap_{subset}.npz', delta_accuracy_pp=draws)
    else:
        ordinary = adjusted = np.full((2, 38), np.nan)
    result = []
    for j, c in enumerate(CONDITIONS):
        gs = [records[key(s), c]['generation'] for s in samples]
        bs = [records[key(s), 'All-Writer']['generation'] for s in samples]
        result.append(dict(subset=subset, condition=c, questions=len(samples),
            image_clusters=len({s['image_sha256'] for s in samples}), accuracy_pct=float(acc[:, j].mean()*100),
            correct=int(acc[:, j].sum()), delta_accuracy_pp=float(delta[:, j].mean()),
            ci_low=float(ordinary[0, j]), ci_high=float(ordinary[1, j]),
            Bonferroni36_low=float(adjusted[0, j]) if j >= 2 else '',
            Bonferroni36_high=float(adjusted[1, j]) if j >= 2 else '',
            correct_to_wrong=int(((acc[:, 0] == 1) & (acc[:, j] == 0)).sum()),
            wrong_to_correct=int(((acc[:, 0] == 0) & (acc[:, j] == 1)).sum()),
            first_token_flips=sum(g['first_token_flip'] for g in gs),
            text_changes=sum(g['text'] != b['text'] for g, b in zip(gs, bs)),
            eos=sum(g['eos'] for g in gs), truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs), repeated=sum(g['repeated_trigram'] for g in gs),
            mean_generated_tokens=float(np.mean([g['generated_tokens'] for g in gs]))))
    print('SUMMARY', subset, len(samples), flush=True)
    return result


def figures(overall, tasks):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 220})
    x = np.arange(36)
    drops = overall[2:]
    y = np.array([r['delta_accuracy_pp'] for r in drops])
    low = np.array([r['Bonferroni36_low'] for r in drops])
    high = np.array([r['Bonferroni36_high'] for r in drops])
    def save(fig, stem):
        for ext in ('png', 'pdf'):
            fig.savefig(OUT / f'{stem}.{ext}')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
    ax.vlines(x, low, high, color='#087e8b', alpha=.35, lw=3, label='Bonferroni-36 family 95% CI')
    ax.plot(x, y, 'o-', ms=3, lw=1.2, color='#087e8b', label='Qwen3-VL-8B single-layer R-to-Image Drop')
    worst = int(y.argmin())
    ax.scatter([worst], [y[worst]], color='#c73e48', s=36, zorder=5, label=f'Largest observed harm: L{worst}')
    ax.axhline(0, color='.3', ls='--', lw=.8)
    ax.set(xlabel='Dropped visual writing layer', ylabel=r'$\Delta$ Accuracy vs All-Writer (pp)',
        xticks=np.arange(0, 36, 2), title='Qwen3-VL Visual Writing Necessity')
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    save(fig, 'figure_A_visual_writing_necessity')
    names = sorted({r['subset'] for r in tasks})
    matrix = np.array([[next(r['delta_accuracy_pp'] for r in tasks if r['subset'] == task and r['condition'] == f'Drop-L{i}') for i in x] for task in names])
    limit = max(float(np.abs(matrix).max()), .1)
    fig, ax = plt.subplots(figsize=(9, 3.3), layout='constrained')
    image = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
    ax.set(yticks=np.arange(6), yticklabels=names, xticks=np.arange(0, 36, 2),
        xlabel='Dropped writing layer', title='Task-specific Accuracy Change')
    fig.colorbar(image, ax=ax, label=r'$\Delta$ Accuracy (pp)')
    save(fig, 'figure_B_task_heatmap')
    colors = ['#087e8b', '#c73e48', '#4c956c', '#735d9b', '#d08c18', '#386cb0']
    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout='constrained')
    for i, (task, values, color) in enumerate(zip(names, matrix, colors)):
        ax.plot(x, values, color=color, marker=['o', 's', '^', 'D', 'v', 'x'][i], ms=2.8,
            lw=1, ls='-' if i < 3 else '--', label=task)
    ax.axhline(0, color='.35', lw=.8)
    ax.set(xlabel='Dropped writing layer', ylabel=r'$\Delta$ Accuracy (pp)', xticks=np.arange(0, 36, 2))
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=.18)
    save(fig, 'figure_C_task_curves')
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
    ax.bar(x-.18, [r['correct_to_wrong'] for r in drops], width=.36, color='#c73e48', label='Correct to wrong')
    ax.bar(x+.18, [r['wrong_to_correct'] for r in drops], width=.36, color='#4c956c', label='Wrong to correct')
    ax.set(xlabel='Dropped writing layer', ylabel='Questions', xticks=np.arange(0, 36, 2), title='Harm and Rescue')
    ax.legend(frameon=False)
    save(fig, 'figure_D_harm_rescue')
    reference = read(OUT / 'llava_reference_summary.json')['rows']
    reference = [r for r in reference if r['subset'] == 'overall']
    llava = [next(r for r in reference if r['condition'] == f'Drop-L{i}') for i in range(32)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout='constrained')
    ax.plot(x/35, y, 'o-', ms=3, color='#087e8b', label=f"Qwen3-VL, 36 layers; All-Writer {overall[0]['accuracy_pct']:.2f}%")
    ax.plot(np.arange(32)/31, [float(r['delta_accuracy_pp']) for r in llava], 's--', ms=3,
        color='#a23b72', label='LLaVA, 32 layers; All-Writer 87.03%')
    ax.axhline(0, color='.35', ls='--', lw=.8)
    ax.set(xlabel='Relative dropped layer depth', ylabel=r'$\Delta$ Accuracy vs own All-Writer (pp)',
        title='Visual Writing: LLaVA and Qwen3-VL')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.18)
    save(fig, 'figure_E_llava_vs_qwen')
    csv_file('figure_E_plot_data.csv', [dict(model='Qwen3-VL', layer=i, relative_depth=i/35, delta_accuracy_pp=y[i]) for i in range(36)] +
        [dict(model='LLaVA', layer=i, relative_depth=i/31, delta_accuracy_pp=float(llava[i]['delta_accuracy_pp'])) for i in range(32)])


def report(overall, tasks, disjoint):
    drops = overall[2:]
    worst = sorted(drops, key=lambda r: r['delta_accuracy_pp'])[:5]
    unchanged = [r['condition'] for r in drops if r['text_changes'] == 0]
    all_writer, no_writer = overall[:2]
    corrected_harm = [r['condition'] for r in drops if r['Bonferroni36_high'] < 0]
    small_text_change = [r['condition'] for r in drops if r['text_changes'] <= 30]
    qwen_peak = int(worst[0]['condition'].removeprefix('Drop-L'))
    lines = ['# Qwen3-VL GQA3000 视觉写入单层必要性', '',
        '**状态：COMPLETE。3,000题、2,409个图片簇、38条件、114,000条完整答案，逐条独立解码与官方评分复算完成。**', '',
        '目录中的旧 resource_preflight.json 与 preflight_report_zh.md 保留最初GPU繁忙、尚未启动时的记录；本轮恢复执行以 protocol.json、双卡完成记录及 completion.json 为准。', '',
        '## 协议与对照', '',
        '固定官方 Qwen3-VL revision、BF16/Eager、Transformers 4.57.6、原生 mRoPE/DeepStack。所有条件的 P−1 与真实 Decode 均不直接读图；All-Writer 不是原生 Full。R 包含 vision_end 及末图像后、P−1前的模板位置。每条件独立真实 KV Cache。', '',
        '本轮没有计算 Gold NLL。以下主指标均为完整生成答案的 GQA Accuracy，而非首 Token LogP。', '',
        f"All-Writer：{all_writer['accuracy_pct']:.4f}%；No-Writer：{no_writer['accuracy_pct']:.4f}%，差值 {no_writer['delta_accuracy_pp']:+.4f} pp。", '',
        '| 单层Drop | Accuracy | ΔAcc pp | 普通95%CI | Bonferroni36校正95%CI | 损害/纠正 |',
        '| --- | ---: | ---: | --- | --- | ---: |']
    for r in drops:
        lines.append(f"| {r['condition']} | {r['accuracy_pct']:.4f}% | {r['delta_accuracy_pp']:+.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | [{r['Bonferroni36_low']:.4f}, {r['Bonferroni36_high']:.4f}] | {r['correct_to_wrong']}/{r['wrong_to_correct']} |")
    lines += ['', '## 核心研究判断', '',
        '1. 最大点估计损害层为 ' + '、'.join(f"{r['condition']} ({r['delta_accuracy_pp']:+.3f} pp)" for r in worst) + '。Bonferroni36校正区间完全低于0的层为：' + ('、'.join(corrected_harm) if corrected_harm else '无') + '。这区分了观察到的峰与足够强的单层证据；不能只凭峰值声称少数层决定视觉写入。',
        f'2. Qwen最大损害层L{qwen_peak}的相对深度为{qwen_peak/35:.3f}；LLaVA历史L11对应{11/31:.3f}。图E按各模型自身层相对深度和All-Writer基线对齐。峰位是否对应仅是描述性比较，不把模型差异直接归因于架构。',
        '3. 六任务分层结果见 task_accuracy.csv、图B与图C；不同任务的峰值只作描述，不用未校正的任务结果作确认性显著判断。',
        '4. 完整文本严格不变的单层条件：' + ('、'.join(unchanged) if unchanged else '无') + '；文本变化不超过1%的条件：' + ('、'.join(small_text_change) if small_text_change else '无') + '。净Accuracy不变并不代表没有逐题损害，图D保留损害和纠正数量。',
        '5. 单层Drop不等于联合删除安全。这些曲线可用于提出固定写入预算的后续假说，但本轮未测试18层或23层预算，不能声称前半层/前60%安全，也没有实现实际推理加速。', '',
        '## 六任务差异', '', '| 任务 | All-Writer Accuracy | 最大点估计损害层 | ΔAcc pp |', '| --- | ---: | --- | ---: |']
    for task in sorted({r['subset'] for r in tasks}):
        rows = [r for r in tasks if r['subset'] == task]
        baseline = next(r for r in rows if r['condition'] == 'All-Writer')
        peak = min((r for r in rows if r['condition'].startswith('Drop-')), key=lambda r: r['delta_accuracy_pp'])
        lines.append(f"| {task} | {baseline['accuracy_pct']:.3f}% | {peak['condition']} | {peak['delta_accuracy_pp']:+.3f} |")
    lines += ['', '## 图片隔离与统计边界', '',
        f"与历史240题写入图片不重叠的内部子集包含 {disjoint[0]['questions']} 题、{disjoint[0]['image_clusters']} 个图片簇；其逐层结果独立保存为 image_disjoint_summary.csv。它不是完全独立外部数据。", '',
        '以图片SHA256整簇配对重采样20,000次、问题等权、固定种子。普通95%区间与36项Bonferroni校正区间同时报告；极端分位数受有限Bootstrap次数影响。CI包含0不证明无损或等价。历史队列的扩大机制分析不构成外部泛化验证。', '',
        '## 正确性与资源', '',
        '28项CPU测试、双GPU各3题真实Smoke通过：Native/No-op全部有效Logits严格torch.equal，实际Mask概率为0，非目标Query、原生DeepStack/mRoPE、MLP和独立Cache验证通过。每题Drop-L35与All-Writer的所有有效Logits哈希、Token IDs严格一致。未放宽容差。异常、EOS、截断、空答案、重复输出及所有负面案例均保留。', '',
        '运行源码Commit：`' + read(OUT/'protocol.json')['code_commit'] + '`。两卡正式采集耗时分别为 ' + '、'.join(f"{read(OUT/f'shard{i}/complete.json')['elapsed_seconds']/3600:.3f}小时" for i in (0, 1)) + '；GPU释放审计由执行监督器在分析完成后写入。', '',
        '| 条件 | EOS | 截断 | 空答案 | 重复三元组 |', '| --- | ---: | ---: | ---: | ---: |']
    for r in [all_writer, no_writer, worst[0]]:
        lines.append(f"| {r['condition']} | {r['eos']} | {r['truncated']} | {r['empty']} | {r['repeated']} |")
    lines += ['',
        '原子检查点完整归档在 raw_checkpoints_shard0.tar.gz 与 raw_checkpoints_shard1.tar.gz；索引与SHA256见 checkpoint_hashes.json。到此停止，不扩展实验。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def package():
    for shard in range(2):
        destination = OUT / f'raw_checkpoints_shard{shard}.tar.gz'
        with tarfile.open(destination, 'w:gz', compresslevel=6) as archive:
            for path in sorted((OUT / f'shard{shard}/checkpoints').glob('*/*.json')):
                archive.add(path, arcname=str(path.relative_to(OUT)), recursive=False)
        assert destination.stat().st_size < 95*1024**2, 'Split archive before pushing: GitHub single-file limit'
        with tarfile.open(destination, 'r:gz') as archive:
            members = archive.getmembers()
            assert len(members) == 57000
            for member in members:
                source = OUT / member.name
                blob = archive.extractfile(member).read()
                assert __import__('hashlib').sha256(blob).hexdigest() == sha256(source)
        print('PACKAGED and verified', shard, destination.stat().st_size, flush=True)


def main():
    started = time.time()
    tests = read(OUT / 'analysis_tests.json')
    assert tests['passed'] and tests['tests'] == 4
    assert tests['analysis_sha256'] == sha256(__file__)
    assert tests['test_source_sha256'] == sha256(Path(__file__).with_name('analysis_tests.py'))
    samples, records = audit()
    overall = summarize(samples, records, 'overall')
    history_images = set(read(OUT/'historical_overlap.json')['historical_images'])
    independent = [s for s in samples if s['image_sha256'] not in history_images]
    disjoint = summarize(independent, records, 'image_disjoint')
    tasks = []
    for task in sorted({s['task'] for s in samples}):
        tasks += summarize([s for s in samples if s['task'] == task], records, task)
    csv_file('accuracy_summary.csv', overall)
    csv_file('layer_paired_comparisons.csv', overall[2:])
    csv_file('task_accuracy.csv', tasks)
    csv_file('image_disjoint_summary.csv', disjoint)
    figures(overall, tasks)
    report(overall, tasks, disjoint)
    package()
    seal(OUT / 'analysis_sources.json', {str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(Path(__file__).parent.glob('*.py'))})
    seal(OUT / 'analysis_environment.json', dict(executable=__import__('sys').executable,
        versions={name: __import__('importlib.metadata', fromlist=['version']).version(name) for name in ('numpy', 'matplotlib', 'tokenizers')}))
    seal(OUT / 'artifact_hashes.json', {str(path.relative_to(OUT)): sha256(path) for path in sorted(OUT.glob('*'))
        if path.is_file() and path.name not in ('artifact_hashes.json', 'completion.json', 'analysis.log', 'supervisor.log')})
    seal(OUT / 'completion.json', dict(status='COMPLETE', questions=3000, image_clusters=2409, conditions=38,
        generations=114000, independently_rescored=114000, archived_records_verified=114000,
        analysis_elapsed_seconds=time.time()-started, execution_commit=read(OUT/'protocol.json')['code_commit']))
    print('COMPLETE: 114000 independent scores, clustered CIs, five paper figures and raw archives', flush=True)


if __name__ == '__main__':
    main()
