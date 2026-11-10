"""CPU-only publication audit. Never imports torch or opens a GPU context."""
import argparse
import ast
import base64
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import string
import subprocess
import sys


MAIN = Path('/data/lk/深度学习/ATE')
WORKTREES = Path('/data/lk/ATE-worktrees')
FAST = WORKTREES / 'fastv-k3-svh-sixbench-3090'
SPARSE = WORKTREES / 'sparsevlm192-svh-sixbench-3090'
FAST_OUT = FAST / 'vlm_research/results/fastv_k3_svh_sixbench_3090'
SPARSE_OUT = SPARSE / 'vlm_research/results/sparsevlm192_svh_sixbench_3090'
COUNTS = {'GQA': 12578, 'ScienceQA-IMG': 2017, 'POPE': 8910,
          'MME': 2374, 'MMBench': 4377, 'TextVQA': 5000}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + '\n')


def table(path, values):
    assert values
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def copy(source, target):
    assert source.is_file() and not source.is_symlink(), source
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def source_closure(root, frozen):
    todo = [p for p in frozen if p.startswith('vlm_research/')]
    done = set()
    while todo:
        relative = todo.pop()
        if relative in done:
            continue
        done.add(relative)
        path = root / relative
        package = relative.removesuffix('.py').replace('/', '.').rsplit('.', 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module or ''
                names = [base] + [base + '.' + v.name for v in node.names]
            elif isinstance(node, ast.Import):
                names = [v.name for v in node.names]
            for name in names:
                if not name.startswith('vlm_research'):
                    continue
                for candidate in [name.replace('.', '/') + '.py', name.replace('.', '/') + '/__init__.py']:
                    if (root / candidate).is_file() and candidate not in done:
                        todo.append(candidate)
    return done


def pure_scorer(root, official):
    """Load the frozen lexical scorer without importing the inference engine."""
    path = root / 'vlm_research/information_horizon_alignment/score.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    scope = dict(Path=Path, json=json, hashlib=hashlib, ROOT=root,
                 AUTHOR=official / 'author', OUT=official)
    exec(compile(tree, str(path), 'exec'), scope)
    entry = read(official / 'matching_util_api.json')
    content = base64.b64decode(entry['content'])
    assert hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == entry['sha'] == '84ee053d7d267ffe7ad397d3a99c2f5e80716c56'
    tree = ast.parse(content)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             {'can_infer_option', 'can_infer_text', 'can_infer'} or isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_VERBOSE_ANSWER_RE' for t in n.targets)]
    choices = dict(cp=__import__('copy'), os=__import__('os'), re=re, string=string,
                   logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_matching_util.py', 'exec'), choices)
    scope['matcher'] = lambda: choices['can_infer']

    def score(sample, text):
        if sample['benchmark'] == 'GQA':
            parsed = text.rstrip('.').lower()
            return dict(parsed=parsed, score=float(parsed == sample['answer']))
        return scope['score'](sample, text)
    return score


def aggregate(benchmark, items, samples):
    if benchmark == 'MMBench':
        groups = defaultdict(list)
        for r in items:
            groups[samples[r['key']]['circular_group']].append(r['scoring']['score'])
        return 100 * sum(all(v) for v in groups.values()) / len(groups), len(groups)
    if benchmark == 'MME':
        tasks = defaultdict(list)
        for r in items:
            s = samples[r['key']]
            tasks[s['task']].append((s['pair_id'], r['scoring']['score']))
        total = 0.
        for task, rows in tasks.items():
            pairs = defaultdict(list)
            for key, value in rows:
                pairs[key].append(bool(value))
            assert pairs and all(len(v) == 2 for v in pairs.values()), task
            total += 100 * sum(v for _, v in rows) / len(rows)
            total += 100 * sum(all(v) for v in pairs.values()) / len(pairs)
        assert len(tasks) == 14
        return total, len(items)
    return 100 * sum(r['scoring']['score'] for r in items) / len(items), len(items)


def audit(method, destination):
    assert sys.version_info[:2] == (3, 11), 'Use original Python 3.11: Python >=3.12 changes floating-point sum'
    assert destination.resolve() not in (FAST.resolve(), SPARSE.resolve(), MAIN.resolve())
    source = FAST if method == 'fastv' else SPARSE
    original = FAST_OUT if method == 'fastv' else SPARSE_OUT
    results = original / 'complete_with_mme_authorized' if method == 'fastv' else original
    target = destination / 'vlm_research/results' / original.name
    target.mkdir(parents=True, exist_ok=True)
    receipt = read(results / 'completion.json')
    protocol = read(original / 'protocol.json')
    assert receipt['completed'] == receipt['requested'] == sum(COUNTS.values())
    assert receipt['complete_predictions'] and receipt['independent_rescore_passed']
    for name, expected in protocol['sources'].items():
        assert sha(source / name) == expected, name
    assert sha(FAST_OUT / 'sample_manifest.jsonl.gz') == protocol['manifest_sha256']
    assert sha(FAST_OUT / 'partition.json') == protocol['partition_sha256']
    with gzip.open(FAST_OUT / 'sample_manifest.jsonl.gz', 'rt') as stream:
        samples = {s['key']: s for s in map(json.loads, stream)}
    assert Counter(s['benchmark'] for s in samples.values()) == COUNTS
    assignment = read(FAST_OUT / 'partition.json')['assignment']
    copy(original / 'protocol.json', target / 'protocol.json')
    copy(SPARSE_OUT / 'mme_protocol.json', target / 'mme_protocol.json')
    copy(SPARSE_OUT / 'mme_authorization.json', target / 'mme_authorization.json')
    mme_protocol = read(SPARSE_OUT / 'mme_protocol.json')
    for name, expected in mme_protocol['sources'].items():
        snapshot = target / 'mme_execution_source' / name
        if (SPARSE / name).is_file() and sha(SPARSE / name) == expected:
            copy(SPARSE / name, snapshot)
        else:
            raw = subprocess.check_output(['git', '-C', str(MAIN), 'show',
                '9ba5d63ba00d74c14393cc81dee70ba3071f0bec:' + name])
            assert hashlib.sha256(raw).hexdigest() == expected, name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(raw)
        assert sha(snapshot) == expected, name
    for name in ['completion.json', 'validation.json', 'gpu_smoke.json', 'cpu_tests.json',
                 'cpu_tests.log', 'gpu_release_audit.json']:
        copy((results if name == 'completion.json' else original) / name, target / 'evidence' / name)
    for name in ['score_summary.csv', 'task_details.csv', 'checkpoint_binding_audit.json', 'data_acceptance.json', 'artifact_sha256.json']:
        copy(results / name, target / 'evidence' / name)
    if method == 'sparse':
        for name in ['environment.json', 'reference_device_recovery.json']:
            copy(original / name, target / 'evidence' / name)
        failed = original / 'revisions/20261110T182631+0800'
        for name in ['validation.json', 'protocol.json', 'completion.json', 'gpu_smoke.json']:
            if (failed / name).exists():
                copy(failed / name, target / 'evidence/prior_failed_gate' / name)
    for relative in source_closure(source, protocol['sources']):
        copy(source / relative, destination / relative)
    for folder in ['fastv_svh', 'sparsevlm_svh']:
        if (source / 'vlm_research' / folder / 'README.md').exists() and (folder == 'fastv_svh' or method == 'sparse'):
            copy(source / 'vlm_research' / folder / 'README.md', destination / 'vlm_research' / folder / 'README.md')
    if method == 'sparse':
        ext = 'vlm_research/external_baselines/sparsevlm_icml2025'
        for name in ['official_source/LICENSE', 'official_source_manifest.json', 'source_receipt.json']:
            copy(source / ext / name, destination / ext / name)
    official = target / 'official'
    raw = subprocess.check_output(['git', '-C', str(MAIN), 'show',
        'e2022e7e195fcd121ac2275c4c42edfa30642e17:vlm_research/results/seven_benchmark_native_physical_quality/official/matching_util_api.json'])
    write(official / 'matching_util_api.json', json.loads(raw))
    copy(Path('/data/lk/ATE-resources/Information-Horizon/llava/eval/m4c_evaluator.py'),
         official / 'author/llava/eval/m4c_evaluator.py')
    score = pure_scorer(source, official)
    smoke = read(original / 'gpu_smoke.json')
    validation = read(original / 'validation.json')
    assert smoke['passed'] and len(smoke['samples']) == (3 if method == 'fastv' else 6)
    assert digest(protocol) == smoke['protocol_sha256'] == validation['protocol_sha256']
    if 'gpu_smoke_sha256' in validation:
        assert sha(original / 'gpu_smoke.json') == validation['gpu_smoke_sha256']
    for s in smoke['samples']:
        assert s['passed'] and s['native_noop_all_valid_logits_exact'] and s['inputs_weights_unchanged']
        assert s['pruning_disabled_frozen_svh_tokens_logits_exact' if method == 'fastv' else 'pruning_disabled_svh_tokens_logits_exact']
        if method == 'sparse':
            assert s['dense_reference_tokens_exact'] and s['exception_hooks_restored']
            for hidden in s['hidden_audits']:
                assert hidden['relative_rmse'] <= protocol['numerical_gate']['hidden_relative_rmse']
                assert hidden['cosine_distance'] <= protocol['numerical_gate']['hidden_cosine_distance']
    roots = [(original, digest(protocol), set(COUNTS))] if method == 'sparse' else [
        (original, digest(protocol), set(COUNTS) - {'MME'}),
        (original / 'mme_authorized', digest(read(SPARSE_OUT / 'mme_protocol.json')), {'MME'})]
    records, inventory, legacy = {}, [], []
    preserved = {}
    for path in [results / 'per_question.csv', results / 'per_question.jsonl.gz', FAST_OUT / 'sample_manifest.jsonl.gz', FAST_OUT / 'partition.json']:
        preserved[str(path)] = sha(path)
    expected_visual = [576] * 3 + ([192] * 17 if method == 'fastv' else [309] * 4 + [204] * 9 + [113] * 4) + [0] * 12
    for root, expected_protocol, allowed in roots:
        for rank in [0, 1]:
            folder = root / f'worker{rank}'
            finished = read(folder / 'completion.json')
            assert finished['status'] == 'COMPLETE_SHARD' and finished['completed'] == finished['expected']
            database = folder / 'results.sqlite'
            wal = folder / 'results.sqlite-wal'
            assert not wal.exists() or wal.stat().st_size == 0, 'Audit requires a quiescent checkpoint'
            preserved[str(database)] = sha(database)
            # immutable=1 prevents even shared-memory journal writes. WAL must be empty.
            connection = sqlite3.connect(database.as_uri() + '?mode=ro&immutable=1', uri=True)
            assert connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            row_digest, count = hashlib.sha256(), 0
            for key, payload, checksum in connection.execute('SELECT key,payload,sha256 FROM results ORDER BY key'):
                r = json.loads(payload)
                assert digest(r) == checksum and r['key'] == key and key not in records
                sample = samples[key]
                assert r['benchmark'] == sample['benchmark'] and r['benchmark'] in allowed
                assert r['protocol_sha256'] == expected_protocol
                assert r['rank'] == assignment[key] == rank
                assert r['image_sha256'] == sample['image_sha256']
                assert r['gold'] == sample['answer'] and r['question'] == sample['question']
                binding, g = r['input_binding'], r['generation']
                assert binding['prompt'] == sample['prompt'] and len(binding['image_positions']) == 576
                assert binding['Pminus1'] == binding['expanded_length'] - 1
                assert binding['R'] == list(range(binding['image_positions'][-1] + 1, binding['Pminus1']))
                assert r['scoring'] == score(deepcopy(sample), g['text']), key
                if r['sample_sha256'] != digest(sample):
                    assert method == 'fastv' and sample['benchmark'] == 'MMBench'
                    replay = deepcopy(sample)
                    assert score(replay, g['text']) == r['scoring']
                    assert {k for k in sample if sample[k] != replay[k]} == {'options'}
                    assert digest(replay) == r['sample_sha256']
                    legacy.append(key)
                assert g['all_effective_logits_finite'] and g['independent_cache'] and g['logits_to_keep'] == 1
                assert [x['visual_rows'] for x in g['physical_rows']] == expected_visual
                assert g['kv']['persistent_original_visual_tokens'] == 0
                assert all(x.get('image_tokens', 0) == 0 for x in g['kv']['layers'])
                if method == 'sparse':
                    assert g['mean_visual_rows'] == sum(expected_visual) / 32
                    assert len(g['stages']) == 3
                    for stage in g['stages']:
                        assert stage['score_only'] and not stage['residual_injection']
                        assert stage['pminus1_image_probability_max'] == 0
                        assert stage['official_recycling_ast_unmodified']
                        assert stage['original_text_positions_preserved'] and stage['original_text_hidden_preserved']
                records[key] = r
                row_digest.update((key + '\t' + checksum + '\n').encode())
                count += 1
            connection.close()
            assert count == finished['completed']
            inventory.append(dict(path=str(database), rows=count, sha256=preserved[str(database)],
                                  ordered_record_sha256=row_digest.hexdigest(), journal='empty WAL; immutable read-only'))
            print(method, 'verified shard', database, count, flush=True)
    assert set(records) == set(samples) and len(records) == sum(COUNTS.values())
    assert Counter(r['benchmark'] for r in records.values()) == COUNTS
    assert legacy == (['MMBench:1737'] if method == 'fastv' else [])
    summaries, comparisons, anomalies = [], [], []
    for prior in receipt['benchmarks']:
        benchmark = prior['benchmark']
        items = [r for r in records.values() if r['benchmark'] == benchmark]
        value, units = aggregate(benchmark, items, samples)
        assert abs(value - prior['fusion']) < 1e-9, (benchmark, value, prior['fusion'])
        counts = {k: sum(r['generation'][k] for r in items) for k in ['empty', 'truncated', 'eos', 'repeated_3gram']}
        assert all(counts[k] == prior[k] for k in ['empty', 'truncated', 'eos'])
        failed = sum(r['scoring'].get('parse_failed', False) for r in items)
        summaries.append(dict(benchmark=benchmark, method=receipt['method'], metric=prior['metric'],
            expected=COUNTS[benchmark], completed=len(items), aggregation_units=units,
            generation_status='COMPLETE', score=value,
            historical_svh_reference=prior['historical_svh_reference'],
            delta_vs_historical_svh=value-prior['historical_svh_reference'],
            comparison_status='UNPAIRED_HISTORICAL_REFERENCE',
            historical_image_bytes_identical=False if benchmark == 'MME' else 'NOT_BOUND',
            **counts, parse_failed=failed))
        comparisons.append(dict(benchmark=benchmark, comparison=receipt['method'] + ' minus original SVH',
            status='PARTIAL_MISSING_BOUND_SVH_PREDICTIONS', expected=COUNTS[benchmark], paired=0,
            delta_paired='', ci_low='', ci_high='', harm='', recovered='',
            aggregate_reference_difference=value-prior['historical_svh_reference'],
            reason='Historical SVH row-level predictions and pixel/input bindings unavailable; MME also has authorized byte exception' if benchmark == 'MME' else
                   'Historical SVH row-level predictions and pixel/input bindings unavailable'))
        anomalies.append(dict(benchmark=benchmark, **counts, parse_failed=failed))
    assert all(sha(Path(p)) == expected for p, expected in preserved.items()), 'Original predictions changed during audit'
    for name, expected in read(results / 'artifact_sha256.json').items():
        assert sha(results / name) == expected, name
    table(target / 'accuracy_summary.csv', summaries)
    table(target / 'paired_comparisons.csv', comparisons)
    table(target / 'negative_results.csv', anomalies)
    write(target / 'raw_inventory.json', dict(databases=inventory, preserved_files_sha256=preserved,
          original_files_unchanged=True, uploaded_raw_predictions=False,
          sample_manifest_sha256=protocol['manifest_sha256'], partition_sha256=protocol['partition_sha256']))
    dependencies = {p: v for p, v in protocol['sources'].items() if not p.startswith('vlm_research/')}
    write(target / 'source_sha256.json', dict(frozen_sources=protocol['sources'], external_dependency_sources=dependencies,
          mme_execution_sources=mme_protocol['sources'],
          published_sources={p: sha(destination / p) for p in sorted(source_closure(source, protocol['sources']))}))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'vlm_research.fusion_results_delivery.test_audit', '-v'],
                           cwd=destination, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    (target / 'publication_cpu_tests.log').write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0
    status = dict(status='PARTIAL_PAIRED_COMPARISON', generation_status='COMPLETE', score_status='COMPLETE',
        completed=len(records), expected=sum(COUNTS.values()), baseline_comparison_status='MISSING_BOUND_SVH_CONTROLS',
        historical_mme_strictly_comparable=False, gpu_started_by_publication_audit=False,
        original_checkpoint_checksums_verified=True, independent_lexical_rescore=True,
        publication_cpu_tests_passed=True, publication_cpu_tests=7,
        publication_python=sys.version, mme_execution_sources_verified=True,
        physical_shapes_all_predictions_verified=True, saved_smoke_validation=validation,
        smoke_rerun=False, saved_smoke_passed=smoke['passed'],
        current_audit_utc=datetime.now(timezone.utc).isoformat(),
        evidence_scope='Saved real GPU smoke plus CPU revalidation of all stored predictions; not a fresh GPU numerical test',
        legacy_exact_scorer_side_effect_keys=legacy)
    write(target / 'validation.json', status)
    write(target / 'completion.json', {k: status[k] for k in ['status', 'generation_status', 'score_status', 'completed', 'expected', 'baseline_comparison_status']})
    write(target / 'publication.json', dict(source_worktree=str(source),
        source_head=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        publication_branch=subprocess.check_output(['git', '-C', str(destination), 'branch', '--show-current'], text=True).strip(),
        base_main_commit='2bda69c738c85537594858707ce7f6e1035dd59a',
        frozen_protocol_unmodified=True, historical_no_push_flags_are_not_publication_status=True,
        no_gpu_execution=True, no_pr=True, no_main_modification=True,
        exclusions=['model weights', 'images', 'datasets', 'full predictions', 'SQLite checkpoints', 'large logs'],
        raw_data_retained_on_server=True))
    lines = ['# 六项融合评测核查与发布', '',
        f"方法：**{receipt['method']}**。六项完整生成与本机独立评分均已完成：**35,256 / 35,256**。",
        '总体标记 **PARTIAL_PAIRED_COMPARISON**：缺少绑定输入的历史 SVH 逐题对照，不能把汇总分差当作严格配对结论。', '',
        '| 基准 | 完成/要求 | 本次完整分数 | 原始 SVH 历史分数 | 历史参考差值 |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['benchmark']} | {s['completed']}/{s['expected']} | {s['score']:.6f} | {s['historical_svh_reference']:.6f} | {s['delta_vs_historical_svh']:+.6f} |")
    lines += ['', '## 评分与比较边界',
        '- 除 MME 为总分点数外，其余均为百分比，差值单位为百分点。MME 为 14 类 Acc+Acc+ 总分；MMBench 为 1,176 个 Circular 组的组级正确率，不是 4,377 行的简单平均。',
        '- MMBench 为冻结 VLMEvalKit exact-matching 的离线复现，不宣称新的官方服务器提交成绩。GQA/SQA-IMG/POPE 为冻结准确率；TextVQA 为官方词法 Soft Accuracy。',
        '- POPE 严格保留冻结的 8,910 题，不扩成其他版本的 9,000 题。所有分母含空答案、截断与解析失败。',
        '- 原始 SVH 仅有历史汇总分；缺少逐题预测以及 Pixel/Input 绑定。paired_comparisons.csv 的配对数为 0，CI、损害/纠正数留空，明确 PARTIAL。没有执行新的 Full/SVH 生成。',
        '- MME 的 landmark/03d5e3bfc958be38.jpg（2 题）为历史授权的本机字节版本；保留全部 2,374 题，不宣称历史图片 SHA 一致或严格配对可比。预期 SHA 为 ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a；实际为 61d010ea9fabdc0bf930362d3cc1e4dc554789036f219feba927551effecd510。', '',
        '## 物理与数值验证',
        f"- 历史 CPU 测试通过（{'11' if method == 'fastv' else '20'} 项）；真实 GPU Smoke {'3' if method == 'fastv' else '6'} 张通过。当前发布只读取、校验记录，没有调用模型或 GPU。原始证据位于 evidence/。",
        '- Native/No-op 有效 Logits 与关闭剪枝后的 SVH Logits 使用严格 torch.equal，保留 logits_to_keep=1。没有放宽冻结门槛。',
        '- 全部 35,256 条重新核对 SQLite 完整性、逐记录 SHA256、题号/问题/Gold/Prompt/图片绑定、冻结协议、独立 Cache、有限 Logits 标志和逐层物理视觉行数。持久原始视觉 KV 均为 0。',
        '- 原始图片没有重新跑 Processor；历史逐输入 Pixel/ID Hash 仅保留并绑定原记录，未将其描述为当前重算的验证。']
    if method == 'sparse':
        max_error = max(s['physical_dense_logits_max_abs'] for s in smoke['samples'])
        max_rmse = max(h['relative_rmse'] for s in smoke['samples'] for h in s['hidden_audits'])
        lines += [f'- 历史物理/dense 参考：Token 一致，Logits 最大绝对误差 {max_error:.8f}，最大 Hidden relative RMSE {max_rmse:.8f}；按原先 allclose(atol=0.10, rtol=0.003)、RMSE≤0.01、cosine distance≤1e-4 验收，不冒充逐位相等。',
            '- 视觉 Hidden 为 576→309→204→113→0，L20 起退出，32 层平均 164.125。官方 192 Top-k、回收与聚类 AST 冻结；文本 RoPE 不压紧，合并视觉 Token 使用已声明锚点。这不是独立 SparseVLM compact-position 实现的完全等价版本。',
            '- 曾有官方参考函数 CPU/CUDA bias 设备不一致的失败 Smoke；失败记录保留。随后仅限定工厂分配设备，生产数学与门槛不变，重新冻结并通过六图门槛。']
    else:
        lines += ['- 视觉 Hidden 为 L0–L2:576、L3–L19:192、L20–L31:0。三图 Smoke 核对官方选择器、位置、MLP 和答案旁路残差；没有保存融合开启时完整 physical/dense Logits 对照，不能声称这项额外检查已通过。',
            '- 原 FastV 汇总曾因 MMBench:1737 评分器原地小写化选项而失败；本次精确重放其副作用，原记录 SHA 仍严格通过，未改 Prompt、输入、预测或原检查点。']
    lines += ['', '## 负面结果',
        '- 对历史 SVH 参考，五项分数下降；MMBench 为轻微上升或持平。该观察不是配对显著性结论，不证明安全无损。',
        f"- TextVQA 有 {summaries[-1]['empty']} 个空答案、{summaries[-1]['truncated']} 个截断；全部保留于分母。各项解析失败和重复统计见 negative_results.csv。",
        '- 未进行同机配对速度实验，不宣称端到端加速；物理视觉行减少不等于已证明实际加速。', '',
        '## 冻结、复现与原始数据',
        '- protocol.json 与 mme_protocol.json 是原始冻结字节副本；其中 PENDING、no_push 等为当时状态，不能替代 completion.json、validation.json 与 publication.json 的当前发布结论。',
        '- mme_execution_source/ 保存 MME 补跑的精确冻结源码与小型运行时依赖；与后来 SparseVLM 正式源码不同的 4 个文件从原 Commit 9ba5d63ba00d74c14393cc81dee70ba3071f0bec 恢复并逐个核对 SHA256。未把较新代码冒充 MME 执行版本。',
        '- 源码按原文件复制，附必要本地 import 依赖。模型 revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager，torch 2.6.0+cu124，Transformers 4.49.0；SparseVLM 额外固定 einops 0.6.1，其运行时依赖 SHA 保存在 source_sha256.json。',
        '- 逐条 SQLite、完整 CSV/JSONL、样本 Manifest、图片和权重留在服务器原路径，raw_inventory.json 给出位置与校验值；不上传完整数据集或巨量日志。源码保留原服务器路径，跨机器复现需恢复这些冻结资源，不可自动改协议。',
        '- 发布审计曾使用默认 Python 3.13，在 TextVQA:34875 的 0.6 与历史 0.5999999999999999 上被严格检查阻止。恢复原 Python 3.11 后原样复核；未放宽相等条件或更改原记录。Python 3.12 起内置浮点 sum 的算法变化不可混入冻结评分。',
        '- CPU 审计复现：/data/lk/深度学习/.envs/ate-vlm-research/bin/python vlm_research/fusion_results_delivery/audit.py --method ' + method + ' --destination <新的独立交付worktree>。必须使用 Python 3.11。此入口只做已保存预测的审计/词法重评分，不生成答案。',
        '- SHA256SUMS 覆盖本交付核心文件和源码；文件摘要不自包含。与 Git Commit 一起使用。未创建或合并 PR，未修改 main。', '']
    (target / 'report_zh.md').write_text('\n'.join(lines))
    paths = sorted(p for p in target.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    paths += [destination / p for p in sorted(source_closure(source, protocol['sources']))]
    paths += list((destination / 'vlm_research/fusion_results_delivery').glob('*.py'))
    assert all(p.is_file() for p in paths)
    (target / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(destination)}\n' for p in sorted(set(paths))))
    print(method, 'DONE', len(records), 'preserved all raw files', target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['fastv', 'sparse'], required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    audit(args.method, args.destination.resolve())
