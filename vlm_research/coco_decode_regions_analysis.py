"""Official frozen COCO scoring, paired image statistics and publication outputs."""
import csv
import json
from pathlib import Path
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import sha256, digest
from paper_rq1.scorers import caption_scores
from paper_rq1_single_layer_gqa6.run import atomic_json
from coco_decode_regions import OUT, ROOT, read, verify, conditions, history_record, checkpoint, old


def save(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else -1)


def interval(values, indices):
    return tuple(float(x) for x in np.quantile(values[indices].mean(1), [.025, .975]))


def figures(summary, scores, comparisons, samples, rows):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})
    keys = list(conditions()); bykey = {r['condition']: r for r in summary}
    labels = ['Full', 'None', 'L0 only', 'Early', 'Middle', 'Late',
              'R10\nseed 02', 'R10\nseed 03', 'R10\nseed 04',
              'R11\nseed 02', 'R11\nseed 03', 'R11\nseed 04']
    colors = ['#555555', '#9a9a9a', '#b67a48', '#197b78', '#b34c66', '#426eaf']+['#8aa9a2']*3+['#a0adc8']*3
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.6), layout='constrained')
    for ax, metric in zip(axes, ['CIDEr', 'SPICE']):
        values = np.array([bykey[k][metric] for k in keys])
        lower = np.array([bykey[k][f'{metric}_ci95_low'] for k in keys])
        upper = np.array([bykey[k][f'{metric}_ci95_high'] for k in keys])
        ax.bar(range(12), values, color=colors, width=.65)
        ax.errorbar(range(12), values, yerr=[values-lower, upper-values], fmt='none',
                    ecolor='#333333', capsize=3, lw=1)
        ax.axhline(values[0], color='#555555', lw=.9, ls='--')
        ax.set_xticks(range(12), labels, fontsize=9); ax.set_ylabel(metric)
        ax.set_ylim(0, max(upper)*1.18); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(i, upper[i]+max(upper)*.025, f'{value:.3f}', ha='center', fontsize=8)
    fig.suptitle('Decode Visual Re-access by Layer Region | 60 paired COCO images', fontsize=12)
    for ext in ['png', 'pdf']: fig.savefig(OUT/f'figure_1_all_conditions.{ext}', dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), layout='constrained')
    for row, metric in enumerate(['CIDEr', 'SPICE']):
        for col, (regions, size) in enumerate([(['Early', 'Middle'], 10), (['Late'], 11)]):
            ax = axes[row, col]
            randomkeys = [f'Random{size}-{seed}' for seed in [20261102, 20261103, 20261104]]
            low, high = min(bykey[k][metric] for k in randomkeys), max(bykey[k][metric] for k in randomkeys)
            ax.axhspan(low, high, color='#888888', alpha=.14, label='Random seed range')
            for j, region in enumerate(regions):
                r = bykey[region]; value = r[metric]
                ax.errorbar(j, value, yerr=[[value-r[f'{metric}_ci95_low']], [r[f'{metric}_ci95_high']-value]],
                            fmt='o', color=colors[keys.index(region)], capsize=4, ms=7)
            for j, k in enumerate(randomkeys):
                ax.scatter(len(regions)+j, bykey[k][metric], marker='s', color='#666666', s=36)
            ax.set_xticks(range(len(regions)+3), regions+['R seed02', 'R seed03', 'R seed04'], fontsize=9)
            ax.set_ylabel(metric); ax.set_title(f'L0 + {size} layers: region vs fixed random sets')
            ax.grid(axis='y', alpha=.15); ax.set_xlim(-.6, len(regions)+2.6)
    fig.suptitle('Matched retained-layer counts; random seeds are not independent images', fontsize=11)
    for ext in ['png', 'pdf']: fig.savefig(OUT/f'figure_2_regions_vs_random.{ext}', dpi=260)
    plt.close(fig)

    # Explicit posthoc illustration: both negative and counterexamples, never aggregate filtering.
    qids = [s['question_id'] for s in samples]; selected = []; selection = []
    def pick(label, values, ascending=True):
        order = sorted(range(60), key=lambda i: (values[i] if ascending else -values[i], samples[i]['image_sha256']))
        i = next(i for i in order if qids[i] not in selected)
        selected.append(qids[i]); selection.append(dict(question_id=qids[i], rule=label, value=float(values[i])))
    for region in ['Early', 'Late']:
        diff = scores[region]['SPICE']-scores['Full']['SPICE']
        pick(f'{region}: largest SPICE degradation', diff)
        pick(f'{region}: largest SPICE improvement / least harm', diff, False)
    pick('Middle: largest SPICE improvement / least harm', scores['Middle']['SPICE']-scores['Full']['SPICE'], False)
    pick('Early minus Random10-seed02 SPICE counterexample',
         scores['Early']['SPICE']-scores['Random10-20261102']['SPICE'])
    atomic_json(OUT/'qualitative_selection.json', dict(posthoc=True, criteria=selection,
        note='All 60 images retained in numerical analysis. Extremal illustrative cases do not estimate prevalence.'))
    lookup = {(r['question_id'], r['condition']): r for r in rows}
    fig, axes = plt.subplots(6, 2, figsize=(13, 19), gridspec_kw={'width_ratios': [1, 2.1]}, layout='constrained')
    qualitative = []
    for j, selection_row in enumerate(selection):
        q = selection_row['question_id']; s = samples[qids.index(q)]
        with Image.open(s['image_path']) as image: axes[j, 0].imshow(image.convert('RGB'))
        axes[j, 0].axis('off'); axes[j, 0].set_title(q.replace('caption:', 'COCO '), fontsize=10)
        ax = axes[j, 1]; ax.axis('off')
        texts = []
        for key in ['Full', 'Decode None', 'L0 Only', 'Early', 'Middle', 'Late']:
            r = lookup[(q, key)]
            caption = textwrap.fill(r['caption'], width=105)
            text = f"{key} [C {r['CIDEr']:.2f}; S {r['SPICE']:.3f}]: {caption}"
            texts.append((key, text))
            qualitative.append(dict(question_id=q, selection_rule=selection_row['rule'],
                                     condition=key, caption=r['caption'], CIDEr=r['CIDEr'], SPICE=r['SPICE']))
        y = .98
        line_step = .89/sum(text.count('\n')+1.6 for _, text in texts)
        for key, text in texts:
            ax.text(0, y, text, va='top', fontsize=7.8, linespacing=1.15,
                    color=colors[keys.index(key)], transform=ax.transAxes)
            y -= line_step*(text.count('\n')+1.6)
        ax.text(0, .01, textwrap.fill(selection_row['rule'], 100), fontsize=8, color='#555555', transform=ax.transAxes)
    fig.suptitle('Qualitative examples: native Prefill, decode-only visual access (posthoc, includes harms)', fontsize=13)
    for ext in ['png', 'pdf']: fig.savefig(OUT/f'qualitative_6_images.{ext}', dpi=200)
    plt.close(fig); save('qualitative_captions.csv', qualitative)


def main():
    started = time.time(); p = verify(); manifest = read(OUT/'sample_manifest.json')
    samples = manifest['samples']; keys = list(conditions()); completed = []
    for i in range(3):
        c = read(OUT/f'shard{i}/completion.json'); assert c['generations'] == 200
        assert c['protocol_sha256'] == digest(p)
        assert read(OUT/f'shard{i}/smoke.json')['passed']; completed.append(c)
    refs = {s['question_id']: s['answer'] for s in samples}
    historical = {'Full': {}, 'Decode None': {}}
    old_protocol = old.verify()
    from rq2_visual_handoff.engine import CONDITIONS
    assert CONDITIONS['A'] == (True, True), 'Historical A must be native relay and native direct access'
    for s in samples:
        q = s['question_id']; historical['Full'][q] = history_record(s, 'Full')
        path = old.OUT/f"checkpoints/{q.replace(':','_')}/DN.json"
        record = old.checkpoint(path, digest(old_protocol), s, 'DN')
        assert record['first_logits_exact_historical_Full']
        with np.load(old.OUT/record['raw_file']) as raw:
            historical['Decode None'][q] = record['generation'], raw['first_logits'].copy()
    scores = {}; per = {}; summary = []; rows = []; generations = {}; audit_counts = []
    rng = np.random.default_rng(p['bootstrap']['seed'])
    indices = rng.integers(0, 60, (p['bootstrap']['repetitions'], 60))
    for key in keys:
        pred = {}; generations[key] = {}
        for s in samples:
            q = s['question_id']; base, hf = historical['Full'][q]
            if key in ['Full', 'Decode None']:
                g, first = historical[key][q]
                assert np.array_equal(np.asarray(first), hf.numpy())
            else:
                path = OUT/f"shard{manifest['assignment'][q]}/checkpoints/{q.replace(':','_')}/{key}.json"
                r = checkpoint(path, s, key, p); g = r['generation']
                with np.load(path.with_suffix('.npz')) as raw: assert np.array_equal(raw['first_logits'], hf.numpy())
                assert r['hooks_inputs_weights_unchanged'] and r['first_logits_exact_historical']
                a = r['audit']; assert a['blocked_max'] == 0 and a['retained_min'] > 0
                assert a['rows'] == 32*len(g['token_ids']) and a['prefill_rows'] == 32
                assert a['blocked_rows'] == (32-len(conditions()[key]))*g['decode_steps']
                audit_counts.append(dict(question_id=q, condition=key, **a))
            if key == 'Full':
                assert g['condition'] == 'A' and not g['noop']
            else:
                assert g['prefill_blocked'] == []
                assert g['decode_blocked'] == sorted(set(range(32))-set(conditions()[key]))
            assert g['token_ids'][0] == base['token_ids'][0] and g['decode_steps'] == len(g['token_ids'])-1
            assert g['eos'] == (g['token_ids'][-1] == 2) and len(g['token_ids']) <= 96
            pred[q] = g['text']; generations[key][q] = g
        score_file = OUT/f'scores/{key}.json'
        binding = digest(dict(refs=refs, predictions=pred, scorer=p['scorer_source']))
        if score_file.exists():
            sr = read(score_file); assert sr['binding'] == binding
            overall, image_scores = sr['overall'], sr['per_image']
        else:
            overall, image_scores = caption_scores(refs, pred)
            atomic_json(score_file, dict(binding=binding, overall=overall, per_image=image_scores))
        per[key] = image_scores
        scores[key] = {metric: np.array([image_scores[s['question_id']][metric] for s in samples])
                       for metric in ['CIDEr', 'SPICE']}
        entry = dict(condition=key, n_images=60, retained_layers=len(conditions()[key]),
                     retained_layer_ids=json.dumps(conditions()[key]),
                     provenance='historical strictly bound + 3-GPU replay' if key in ['Full', 'Decode None'] else 'new', **overall)
        for metric in ['CIDEr', 'SPICE']:
            assert abs(overall[metric]-scores[key][metric].mean()) < 1e-10
            lo, hi = interval(scores[key][metric], indices)
            entry[f'{metric}_ci95_low'] = lo; entry[f'{metric}_ci95_high'] = hi
            entry[f'delta_{metric}_vs_Full'] = overall[metric]-summary[0][metric] if summary else 0.
        entry.update(eos=sum(g['eos'] for g in generations[key].values()),
                     truncated=sum(g['truncated'] for g in generations[key].values()),
                     empty=sum(not g['text'] for g in generations[key].values()))
        summary.append(entry)
        for s in samples:
            q = s['question_id']; g = generations[key][q]; base = generations['Full'][q]
            diff = divergence(g['token_ids'], base['token_ids'])
            assert diff == -1 or diff >= 1, 'Prefill / first token incorrectly intervened'
            rows.append(dict(question_id=q, image_id=s['raw_row']['id'], image_sha256=s['image_sha256'],
                physical_gpu=p['gpus'][manifest['assignment'][q]]['physical_gpu'], condition=key,
                caption=g['text'], token_ids=json.dumps(g['token_ids']), first_divergence_zero_based=diff,
                changed=diff != -1, first_token_same=True, eos=g['eos'], truncated=g['truncated'],
                empty=not bool(g['text']), length=len(g['token_ids']), decode_steps=g['decode_steps'],
                **image_scores[q]))
        print(f'Official scores: {key}: {overall}', flush=True)
    comparisons = []
    pairs = [(k, 'Full') for k in keys if k != 'Full']+[('L0 Only', 'Decode None')]
    pairs += [(region, f'Random10-{seed}') for region in ['Early', 'Middle'] for seed in [20261102, 20261103, 20261104]]
    pairs += [('Late', f'Random11-{seed}') for seed in [20261102, 20261103, 20261104]]
    for a, b in pairs:
        for metric in ['CIDEr', 'SPICE']:
            values = scores[a][metric]-scores[b][metric]; lo, hi = interval(values, indices)
            comparisons.append(dict(condition=a, reference=b, metric=metric, delta=float(values.mean()),
                ci95_low=lo, ci95_high=hi, improved=int((values > 0).sum()), worsened=int((values < 0).sum()),
                tied=int((values == 0).sum()), n_images=60, status='paired image exploratory percentile CI; fixed-corpus CIDEr IDF'))
    random_ranges = []
    for size in [10, 11]:
        for metric in ['CIDEr', 'SPICE']:
            vals = [next(r[metric] for r in summary if r['condition'] == f'Random{size}-{seed}') for seed in [20261102, 20261103, 20261104]]
            random_ranges.append(dict(random_layers_plus_L0=size, metric=metric,
                seed20261102=vals[0], seed20261103=vals[1], seed20261104=vals[2],
                minimum=min(vals), maximum=max(vals), range=max(vals)-min(vals)))
    save('condition_summary.csv', summary); save('per_image.csv', rows)
    save('paired_comparisons.csv', comparisons); save('random_seed_ranges.csv', random_ranges)
    save('mask_audit.csv', audit_counts)
    figures(summary, scores, comparisons, samples, rows)
    write_report(p, summary, comparisons, random_ranges, completed, rows)
    validation = dict(passed=True, images=60, conditions=12,
        new_checkpoints=600, historical_generations=120, total_generations=720,
        original_60_image_binding_exact=True, all_prefill_first_token_exact=True,
        masked_probabilities_exact_zero=True, native_cache_positions_lengths=True,
        cpu_tests=read(OUT/'cpu_validation.json')['tests'], three_gpu_smokes=True,
        all_conditions_eos={r['condition']:r['eos'] for r in summary},
        empty={r['condition']:r['empty'] for r in summary},
        truncated={r['condition']:r['truncated'] for r in summary},
        analysis_seconds=time.time()-started, execution_commit=read(OUT/'execution.json')['commit'],
        analysis_source_sha256=sha256(Path(__file__)))
    if (OUT/'validation.json').exists():
        prior = read(OUT/'validation.json')
        dynamic = {'analysis_seconds', 'analysis_source_sha256'}
        assert {k:v for k,v in prior.items() if k not in dynamic} == {k:v for k,v in validation.items() if k not in dynamic}
        atomic_json(OUT/'publication_validation.json', validation)
    else:
        atomic_json(OUT/'validation.json', validation)
    print('Analysis complete', flush=True)


def write_report(p, summary, comparisons, ranges, completed, rows):
    minutes = ', '.join(f"{c['formal_seconds']/60:.2f}" for c in completed)
    lines = ['# COCO Decode 视觉重访问：层分区与固定随机对照', '',
             '## 范围与协议',
             '严格复用历史60张COCO及全部参考Caption/原始Prompt。LLaVA-1.5-7B固定revision，FP16/eager/Transformers4.49.0；Greedy最多96 Token。',
             '全部Prefill（包括P−1）原生，仅真实增量Decode的原始Image Key在未保留层被Softmax前−inf屏蔽。不是o_proj OFF、Relay OFF、P−1-only、物理KV删除或加速。',
             'Full/Decode None的120条生成严格绑定历史检查点、输入、模型与原始Logits；三张同架构GPU均完整复现所分配Smoke图片的两种序列，并核对全部新增600条的首Token全词表Logits精确一致。',
             '', '## 官方分数（原始尺度）',
             '|条件|读图层数|CIDEr|ΔCIDEr|SPICE|ΔSPICE|EOS/60|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['condition']}|{r['retained_layers']}|{r['CIDEr']:.6f}|{r['delta_CIDEr_vs_Full']:+.6f}|{r['SPICE']:.6f}|{r['delta_SPICE_vs_Full']:+.6f}|{r['eos']}|")
    lines += ['', '## 同图配对差异',
              '官方历史COCO评分器（PTBTokenizer、CIDEr、SPICE）与参考Caption不变。20,000次Image SHA256整簇配对Bootstrap；此处60张图均唯一。',
              '沿用原实验口径：CIDEr固定原60图参考语料的document frequencies，对官方逐图贡献配对重采样；这是条件于原语料的描述性区间，不是重估新语料IDF的总体区间。SPICE重采样官方逐图F分数。',
              '区间未经多重比较校正，全部为探索性；包含0不证明等价。三个随机种子为固定条件，不作为独立图像。',
              '|条件 − 参考|指标|差值|配对95%CI|逐图提升/损害/相同|', '|---|---|---:|---|---|']
    for r in comparisons:
        lines.append(f"|{r['condition']} − {r['reference']}|{r['metric']}|{r['delta']:+.6f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|{r['improved']}/{r['worsened']}/{r['tied']}|")
    lines += ['', '## 固定随机种子分数和范围', '|随机层数（另含L0）|指标|seed02|seed03|seed04|范围|', '|---|---|---:|---:|---:|---|']
    for r in ranges:
        lines.append(f"|{r['random_layers_plus_L0']}|{r['metric']}|{r['seed20261102']:.6f}|{r['seed20261103']:.6f}|{r['seed20261104']:.6f}|[{r['minimum']:.6f}, {r['maximum']:.6f}]|")
    lines += ['', '## 中文核心发现',
              '1. 本批60图中，Middle最接近Full：CIDEr差+0.043758，配对95%CI [−0.072026, +0.156257]；SPICE差−0.008289，CI [−0.023038, +0.006618]。两区间包含0，既不能证明提升，也不能证明无损或等价。',
              '2. Early明显弱于固定Random10：CIDEr差−0.182346至−0.305648，SPICE差−0.038252至−0.051805；六个未校正探索性配对区间均在0以下。不能据此宣称层功能专门化。',
              '3. Middle对三个Random10的SPICE点估计均更高，差+0.012783至+0.026336，其中seed04的区间跨0；CIDEr仅对seed03的探索性区间在0以上。随机对照不支持不加限定的“中层必然更优”。',
              '4. Late对Full的SPICE损失−0.046118，CI [−0.073458, −0.020963]；对三个Random11的SPICE点估计也都更低，seed04的区间跨0，另两个在0以下。Late的CIDEr差区间均跨0。',
              '5. 固定L0并未带来稳定改善：L0 Only−Decode None的CIDEr差+0.038470，CI [−0.026495, +0.106537]；SPICE差−0.004209，CI [−0.016066, +0.007261]。不能把L0的特殊性当作已被证明的收益。',
              '', '## 解释边界与案例',
              'Early/Middle各保留L0+10层，与Random10逐种子比较；Late保留L0+11层，与Random11比较。完整配对差异见上表，不能根据本轮结果挑选更好随机集合。',
              'L0 Only与Decode None专门对照固定L0的影响；保留L0不等于其他层专门化。',
              '定性图含6张原图及Full/None/L0 Only/Early/Middle/Late完整文本，按预声明的实测SPICE损害、改善及随机反例极值规则事后挑选，详见qualitative_selection.json；不反映总体频率。',
              '例如COCO 401024中Late把剪头发的剪刀描述为切蛋糕的刀，而Full/Early/Middle保留原描述；COCO 514668中Early丢失车内场景。反例COCO 91080中Late的SPICE高于Full。COCO 381563的Early虽有SPICE提升，却出现“sandwich”描述，说明指标上升本身也不能替代图文事实核查；本轮没有新增幻觉评价指标。',
              '本轮只在历史探索60图上，不能宣称独立确认、功能专门化、严格无损、非劣效或真实推理加速。概率重新分配以及其他文本路径仍然存在；注意力质量评分差异不等同来源信息的独立贡献。',
              '', '## 验证与资源',
              '10项CPU回归与三卡真实Smoke通过；Full/no-op所有生成Logits精确一致，原生HF生成Token一致，全部Prefill Logits/其他Query Attention/Hidden保持精确一致。',
              '12条件Smoke与全部新增真实Decode逐层验证目标Image概率严格0、保留层实际读图、绝对位置/每层Cache长度正确；每条件独立Cache，Hook清理、输入与参数版本不变。无静默跳样，无NaN/Inf/OOM。',
              f"新增生成600条；三卡正式耗时分别：{minutes}分钟。GPU1从未使用。",
              f"完整EOS={sum(r['eos'] for r in summary)}/720；截断={sum(r['truncated'] for r in summary)}；空文本={sum(r['empty'] for r in summary)}。",
              f"执行代码Commit：{read(OUT/'execution.json')['commit']}。主分支未修改。", '']
    lines += ['## 执行留痕',
              'GPU生成与Smoke没有失败。CPU汇总曾因历史Full缺少新字段prefill_blocked、历史图片ID位于raw_row.id而各中止一次；两次失败日志原样保留。修复仅涉及读取历史Schema，没有修改模型、Mask、容差、生成条件或重跑GPU。', '']
    lines += ['随后重新核验出版产物时，原validation.json的不可覆盖保护触发；原始成功评分验证保留，最终出版源码验证另存publication_validation.json，不覆盖完成的检查点。', '']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


if __name__ == '__main__': main()
