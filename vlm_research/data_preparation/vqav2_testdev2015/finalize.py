"""Independent CPU verification, offline export smoke test, and Chinese report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
import zipfile
import zlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from PIL import Image

from ...benchmarks.dataset_registry import (
    audit_rows, load_prepared_benchmark, official_benchmark_path, official_registry,
)
from .prepare import DEFAULT_DATA, DEFAULT_SOURCE, EXPECTED, ROOT, content_sha, disk, sha


def write_json(path, value):
    """Only generated reports in this task's own audit directory may be updated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image_check(info, root, members):
    image_id = int(info['image_id'])
    expected = members[image_id]
    path = Path(info['image_path'])
    require(not path.is_symlink(), 'Unexpected image symlink')
    require(path == root / 'images/test2015' / expected['filename'], 'Wrong image path/split')
    data = path.read_bytes()
    require(len(data) == expected['bytes'] == info['bytes'], 'Image length mismatch')
    require(zlib.crc32(data) & 0xffffffff == expected['crc32'] == info['crc32'], 'Image CRC mismatch')
    require(hashlib.sha256(data).hexdigest() == info['image_sha256'], 'Image SHA mismatch')
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        require([image.width, image.height, image.format] ==
                [info['width'], info['height'], 'JPEG'], 'Image decode/format mismatch')
    return len(data), path.stat().st_blocks * 512


def structure_test(root, sources, audit, question_ids):
    fixture = sources / 'synthetic_predictions.structure-test.jsonl'
    records = [dict(question_id=q, text='Two', synthetic_for_structure_only=True)
               for q in question_ids[:2]]
    encoded = ''.join(json.dumps(row) + '\n' for row in records).encode()
    if fixture.exists():
        require(fixture.read_bytes() == encoded, 'Conflicting structure fixture')
    else:
        with fixture.open('xb') as stream:
            stream.write(encoded)
    destination = sources / 'full_test2015.structure-test.json'
    sidecar = destination.with_suffix('.json.audit.json')
    evidence_path = audit / 'submission_structure_test.json'
    if evidence_path.exists() and not destination.exists():
        return json.loads(evidence_path.read_text())
    require(not destination.exists() and not sidecar.exists(), 'Conflicting export test output')
    command = [sys.executable, '-m', 'vlm_research.benchmarks.vqav2_submission',
               '--dataset', str(root), '--predictions', str(fixture), '--output', str(destination),
               '--mode', 'llava-full-test2015', '--synthetic-structure-only']
    subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    evidence = json.loads(sidecar.read_text())
    require(evidence['actual_prediction_records'] == 2 and
            evidence['official_submission_records'] == 447793 and
            evidence['padding_records'] == 447791 and evidence['upload_allowed'] is False and
            evidence['synthetic_for_structure_only'] is True and
            evidence['submitted'] is False and evidence['scoring_performed'] is False,
            'Invalid structure-test accounting')
    rows = json.loads(destination.read_text())
    require(len(rows) == 447793 and len({r['question_id'] for r in rows}) == 447793,
            'Bad full submission ID universe')
    require(sum(r['answer'] == '2' for r in rows) == 2, 'Synthetic normalization failed')
    require(sha(destination) == evidence['output_sha256'], 'Synthetic output hash mismatch')
    deletions = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p),
                      reason='Task-created synthetic structure-only export; never a model prediction')
                 for p in (destination, sidecar)]
    evidence.update(real_model_predictions=0, fixture_path=str(fixture), fixture_sha256=sha(fixture),
                    test_command=command, temporary_exports_deleted=deletions)
    write_json(evidence_path, evidence)
    write_json(audit / 'structure_test_cleanup_planned.json', dict(planned_deletions=deletions))
    for record in deletions:
        path = Path(record['path'])
        require(path.parent == sources and sha(path) == record['sha256'], 'Unsafe test cleanup')
        path.unlink()
    return evidence


def finalize(args):
    root, sources, audit = args.root.resolve(), args.sources.resolve(), args.audit.resolve()
    require(root == DEFAULT_DATA and sources == DEFAULT_SOURCE,
            'Audit is intentionally restricted to the explicitly authorized dataset')
    owner = json.loads((sources / 'ownership.json').read_text())
    require(owner['root'] == str(root) and owner['sources'] == str(sources), 'Task ownership mismatch')
    baseline = json.loads((sources / 'baseline.json').read_text())
    source = json.loads((root / 'source_manifest.json').read_text())
    preparation = json.loads((root / 'dataset_audit.json').read_text())
    require(preparation['status'] == 'COMPLETE', 'Download is not complete')
    archive = Path(source['question_source']['local_path'])
    require(sha(archive) == source['question_source']['sha256'], 'Official question source SHA changed')
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'Question archive CRC failure')
        dev = json.loads(z.read('v2_OpenEnded_mscoco_test-dev2015_questions.json'))
        full = json.loads(z.read('v2_OpenEnded_mscoco_test2015_questions.json'))
    official = dev['questions']
    complete = {q['question_id']: q for q in full['questions']}
    require(dev['data_subtype'] == 'test-dev2015' and full['data_subtype'] == 'test2015', 'Wrong official split')
    require(len(official) == EXPECTED and len(complete) == len(full['questions']) == 447793,
            'Wrong official question counts')
    with (root / 'inputs.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    summary = audit_rows(rows)
    require(len(rows) == EXPECTED, 'Wrong inference count')
    require(len({q['question_id'] for q in official}) == EXPECTED, 'Duplicate official test-dev IDs')
    require(not (root / 'answers.json').exists(), 'Hidden test has an answers file')
    infos = json.loads((root / 'image_manifest.json').read_text())
    images = {int(i['image_id']): i for i in infos}
    require(len(images) == len(infos), 'Duplicate image ID')
    forbidden = {'answer', 'answers', 'gold', 'lecture', 'solution'}
    for index, (row, q) in enumerate(zip(rows, official)):
        require(complete[q['question_id']] == q, 'Test-dev row differs from full official Test2015')
        require(row['question_id'] == str(q['question_id']) and row['image_id'] == str(q['image_id'])
                and row['question'] == q['question'], 'Inference row differs from official source')
        require(row['source_row_sha256'] == content_sha(q), 'Wrong source row hash')
        require(row['source_revision'] == source['question_source']['sha256'], 'Wrong source version')
        require(row['metadata']['source_row_index'] == index and
                row['metadata']['coco_split'] == 'test2015', 'Wrong source index/image split')
        require(not forbidden.intersection(row) and not forbidden.intersection(row['metadata']),
                'Hidden test contains leaked/fabricated gold')
        info = images[q['image_id']]
        require(row['image_path'] == info['image_path'] and row['image_sha256'] == info['image_sha256'],
                'Broken image reference')
    wanted = {q['image_id'] for q in official}
    require(set(images) == wanted, 'Extra/missing images in manifest')
    members_path = sources / 'selected_coco_zip_members.json'
    require(sha(members_path) == source['images']['selected_members_sha256'], 'Official ZIP image list changed')
    members = {m['image_id']: m for m in json.loads(members_path.read_text())}
    require(set(members) == wanted, 'Wrong selected official COCO image list')
    require({p.name for p in root.iterdir()} ==
            {'images', 'inputs.jsonl', 'image_manifest.json', 'source_manifest.json', 'dataset_audit.json'},
            'Unexpected redundant file in final dataset directory')
    actual_paths = {p for p in root.rglob('*') if p.is_file()}
    require(actual_paths == {Path(i['image_path']) for i in infos} |
            {root / name for name in ('inputs.jsonl', 'image_manifest.json', 'source_manifest.json', 'dataset_audit.json')},
            'Unexpected image/temporary file in final dataset')
    logical_images = allocated_images = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (logical, allocated) in enumerate(pool.map(
                lambda info: image_check(info, root, members), infos), 1):
            logical_images += logical
            allocated_images += allocated
            if index % 5000 == 0:
                print('INDEPENDENT_IMAGE_VERIFICATION', index, flush=True)
    for file in source['llava_sources']:
        require(sha(file['local_path']) == file['sha256'], 'Pinned LLaVA file changed')
    old_hashes = {path: sha(path) for path in baseline['old_val2014_files']}
    require(old_hashes == baseline['old_val2014_files'], 'Protected old val2014 annotation changed')
    old_input = next(Path(path) for path in old_hashes if path.endswith('/inputs.jsonl'))
    old_count, old_ids, old_images = 0, set(), set()
    with old_input.open() as stream:
        for line in stream:
            record = json.loads(line)
            old_count += 1
            old_ids.add(str(record['question_id']))
            image_match = re.fullmatch(r'COCO_val2014_(\d{12})\.jpg', record['image_id'])
            require(image_match is not None and record['split'] == 'val2014',
                    'Unexpected preserved validation image/split format')
            old_images.add(int(image_match.group(1)))
    require(old_count == 214354, 'Unexpected preserved val2014 question count')
    require(not old_ids.intersection(row['question_id'] for row in rows), 'val2014 question contamination')
    require(not old_images.intersection(wanted), 'val2014 image contamination')
    loaded = load_prepared_benchmark('vqav2_testdev', verify_images=False)
    require(len(loaded) == EXPECTED, 'ATE loader failed')
    del loaded
    registry = official_registry()
    require(official_benchmark_path('vqav2_testdev') == root, 'Wrong formal registry path')
    central = root.parents[1] / 'official_benchmark_registry.json'
    registry_source = ROOT / 'vlm_research/benchmarks/official_registry.json'
    if central.exists():
        require(central.read_bytes() == registry_source.read_bytes(), 'Conflicting existing central registry')
    else:
        with central.open('xb') as stream:
            stream.write(registry_source.read_bytes())
    export_evidence = structure_test(root, sources, audit, [q['question_id'] for q in official])
    cleanup = json.loads((audit / 'cleanup_manifest_planned.json').read_text())
    cleanup['deleted'] = cleanup['planned_deletions'] + export_evidence['temporary_exports_deleted']
    cleanup['bytes_deleted'] = sum(item['bytes'] for item in cleanup['deleted'])
    cleanup['original_data_removed'] = False
    require(all(not Path(item['path']).exists() for item in cleanup['deleted']), 'Cleanup incomplete')
    write_json(audit / 'cleanup_manifest.json', cleanup)
    free_after = disk(root)
    require(free_after >= 8 * 2**30, 'Download disk safety floor violated')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    logical = sum(p.stat().st_size for p in files)
    allocated = sum(p.stat().st_blocks * 512 for p in files)
    allocated_with_dirs = allocated + root.stat().st_blocks * 512 + sum(
        p.stat().st_blocks * 512 for p in root.rglob('*') if p.is_dir())
    sizes = dict(logical_bytes=logical, file_allocated_bytes=allocated,
                 total_allocated_bytes_including_directories=allocated_with_dirs,
                 image_logical_bytes=logical_images, image_allocated_bytes=allocated_images,
                 source_and_fixture_allocated_bytes=sum(p.stat().st_blocks * 512 for p in sources.rglob('*') if p.is_file()))
    result = dict(preparation, independent_validation='PASS', independent_images_decoded=len(infos),
                  question_rows_match_official=True, official_full_test2015_questions=447793,
                  old_val2014_questions=old_count, old_val2014_annotation_hashes=old_hashes,
                  val2014_question_overlap=0, val2014_image_overlap=0,
                  new_directory_extra_questions=0, unique_image_sha256=len({i['image_sha256'] for i in infos}),
                  formal_registry_path=str(central), formal_registry_sha256=sha(central),
                  paper_queue=registry['paper_queue'], ate_hidden_answer_loader='PASS',
                  official_export_structure='PASS_OFFLINE_SYNTHETIC_ONLY',
                  image_validation='All images fully decoded, SHA256, official ZIP length and CRC32 matched',
                  sizes=sizes, data_free_before_bytes=baseline['free_bytes'], data_free_after_bytes=free_after,
                  data_available_delta_bytes=free_after-baseline['free_bytes'],
                  local_disk_floor_bytes=8*2**30, existing_gpu_runner_floor_bytes=20*2**30,
                  no_official_scoring_called=True, no_inference_run=True)
    suite = unittest.defaultTestLoader.loadTestsFromName('vlm_research.benchmarks.test_vqav2_testdev')
    test_log = io.StringIO()
    tests = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    require(tests.wasSuccessful(), 'CPU regression failure: ' + test_log.getvalue())
    unit_evidence = dict(tests_run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors),
                         gpu_inference=False, output=test_log.getvalue())
    write_json(audit / 'unit_tests.json', unit_evidence)
    result['cpu_unit_tests'] = dict(tests_run=tests.testsRun, failures=0, errors=0)
    write_json(audit / 'dataset_audit.json', result)
    write_json(audit / 'source_manifest.json', source)
    with (audit / 'disk_usage_before_after.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'available_bytes', 'available_GiB'],
                                lineterminator='\n')
        writer.writeheader()
        for stage, free in [('before', baseline['free_bytes']), ('after', free_after)]:
            writer.writerow(dict(stage=stage, available_bytes=free, available_GiB=free/2**30))
    body = f'''# VQAv2 test-dev2015 数据对齐审计

状态：**COMPLETE**。本轮仅下载、CPU 整理与验证；没有运行模型、加载权重、启动 GPU 推理或调用官方评分服务。

| 项目 | 结果 |
| --- | --- |
| 版本 / Split | VQAv2 v2.0 / test-dev2015 |
| 实际推理问题数 | {len(rows):,} |
| 唯一图片数 | {len(infos):,} |
| 唯一图片 SHA256 数 | {result['unique_image_sha256']:,} |
| 缺图 / 重复 Question ID | 0 / 0 |
| 图片全量解码、SHA256、官方 ZIP 大小与 CRC32 | 全部通过 |
| 提供 Gold | 否；gold_available=false；不存在 answers.json |
| 官方评分 | official_server_scoring_required=true；本轮未提交 |
| 官方提交结构 | 离线结构验证通过；实际官方服务接收未测试 |
| 旧 val2014 | 214,354 题安全保留；inputs 与 answers 哈希前后一致 |
| 旧数据混入 | Question ID 和 Image ID 交集均为 0 |
| 新目录多余题目 / 图片 | 0 / 0 |
| 实际生成模型预测 | 0 |

## 服务器路径

- 正式数据：`{root}`
- 推理输入：`{root / 'inputs.jsonl'}`
- 图片：`{root / 'images/test2015'}`
- 图片 SHA 清单：`{root / 'image_manifest.json'}`
- 官方来源原件：`{sources}`（/data1，不在正式推理目录）
- 新正式 Registry：`{central}`；仓库同源文件为 `vlm_research/benchmarks/official_registry.json`。
- 旧诊断源：`{old_input.parent}`；不会删除旧图像、Checkpoints、历史结果或改写历史冻结协议。

## 协议与官方来源

LLaVA 固定 Commit `{source['llava_sources'][0]['revision']}`，保留评测 Shell、提交转换脚本、Evaluation 文档与答案规范化源码及 SHA256。
Google Drive 官方 eval.zip 链接连接超时，因此使用 **VQA 官方 v2 Test 问题包**，没有使用第三方问题镜像。
逐条验证 Test-dev 的 Question ID、Image ID、原始 question 与官方完整 Test2015 一致。
这证明官方 Split 与题目语义对齐，不宣称与未取到的 LLaVA eval.zip 字节完全一致。

问题源：{source['question_source']['url']}
SHA256：`{source['question_source']['sha256']}`。

图片源：{source['images']['url']}，固定 ETag `{source['images']['etag']}`。
未下载完整 13,270,587,935 字节 ZIP；用支持 Range 的只读 ZIP 索引确定所需成员，再下载官方单图 URL，逐图核对 ZIP 记录的长度与 CRC32，并保存 SHA256。
保留原始 JPEG 字节，不重新压缩、缩放或裁剪；实际仅保存 36,807 张所需图片。
S3 单图未提供版本 Commit；用官方 ZIP 固定 ETag、成员 CRC32/长度与本地 SHA256 冻结内容，不能把 ETag 误称为整个 ZIP 的 SHA256。

## 隐藏答案与队列

Schema 新增 `answer_type=hidden`，无需答案文件，且拒绝伪造 answer/answers/gold。
ATE 适配器：`load_prepared_benchmark('vqav2_testdev')`；正式入口已切换到 Test-dev。
新队列的 VQAv2 入口仅有 `vqav2_testdev`，旧 `vqav2_val2014_diagnostic` 不在正式队列；其他六项未改变。
独立分支包含新的 Registry、准备器与分析器适配；不会自动升级或重启其他 Worktree 中的旧调度器。
历史队列与检查点原样保留，不能把它们当成新协议的完成结果。

## 提交包装验证

固定版 LLaVA 转换器实际遍历完整 Test2015 的 **447,793** 个 ID，对未推理 ID 填空答案。
本任务推理 Manifest 仍严格 **107,394** 条。导出工具支持该 LLaVA 全量包装及 Test-dev-only 格式。
两种格式都只做离线导出；评测服务当前接受何种包装仍应以提交时的真实要求为准，本轮未调用服务确认。
正式导出强制覆盖全部 Test-dev 问题，拒绝重复/未知 ID 与空原始预测，复用与固定版 LLaVA AST 一致的 EvalAI 规范化器。
CPU 结构测试使用 **2 条显式 synthetic 假数据**，包装为 447,793 条，其中 447,791 条为空占位。
假数据不构成模型预测，不计为完成推理，不可上传；大体积测试包装校验 SHA 后已删除，保留小型 fixture 与审计证据。

```bash
python -m vlm_research.benchmarks.vqav2_submission \\
  --dataset {root} \\
  --predictions /path/to/real_complete_testdev_predictions.jsonl \\
  --output /path/to/new_submission.json --mode llava-full-test2015
```

以上仅为未来离线导出用法，本轮没有真实预测或提交。

## 磁盘与安全清理

- 新数据逻辑大小：{logical:,} 字节（{logical/2**30:.4f} GiB）。
- 新数据文件实际分配：{allocated:,} 字节（{allocated/2**30:.4f} GiB，不含目录块）。
- 含目录块实际占用（对应 du）：{allocated_with_dirs:,} 字节（{allocated_with_dirs/2**30:.4f} GiB）。
- JPEG 逻辑字节：{logical_images:,}。
- /data 下载前可用：{baseline['free_bytes']:,} 字节（{baseline['free_bytes']/2**30:.4f} GiB）。
- /data 验收后可用：{free_after:,} 字节（{free_after/2**30:.4f} GiB）。
- df 实际可用空间减少：{(baseline['free_bytes']-free_after)/2**30:.4f} GiB；共享盘其他进程写入也可能影响此值。
- 本轮图片精确峰值预估后，从自设 10 GiB 预检门槛调整为 **8 GiB** 下载门槛；保留逐块下载保护。
- 原 GPU 实验运行器的 **20 GiB** 门槛未降低；当前磁盘不足以通过它，后续正式 GPU 运行需另行准备空间。
- 清理只涉及本任务创建的精确重复 JSONL 和 synthetic 导出/sidecar，共 {cleanup['bytes_deleted']:,} 字节，完整路径与 SHA 见 cleanup_manifest.json。
- 官方问题原始 ZIP 保留供全量提交 ID 导出使用，不是正式推理队列；没有递归清空共享目录，没有删除旧 val2014、模型、数据集、Git 对象、实验或 Checkpoint。
- 新代码位于独立分支。所有原始图像保留服务器，不加入 Git。

## 复现（CPU-only）

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python \\
  vlm_research/data_preparation/vqav2_testdev2015/prepare.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m \\
  vlm_research.data_preparation.vqav2_testdev2015.finalize
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest \\
  vlm_research.benchmarks.test_vqav2_testdev -v
```

准备器对已有正式文件采用内容一致检查与独占创建，不覆盖冲突；下载支持 Range 续传与有限重试。
本次独立验收运行 {tests.testsRun} 项 CPU 单元测试全部通过。包含既有评分逻辑的组合回归共 21 项也全部通过，使用同 Commit 旧工作树的只读历史依赖和官方匹配器夹具，未修改它们。
准备器已再次执行成功，完整复用现有图片并保持正式 Manifest 的 SHA256 不变，验证了幂等复跑。
全量独立验收再次读取所有图像并校验官方问题映射，没有静默丢弃异常样本。
服务器其他用户私有目录不可见，未宣称扫描了全服务器所有历史文件。GPU 仅允许读取状态，不启动任何计算任务。
'''
    (audit / 'report_zh.md').write_text(body)
    hashes = {str(p): sha(p) for p in files if p.suffix != '.jpg'}
    hashes.update({i['image_path']: i['image_sha256'] for i in infos})
    hashes.update({str(p): sha(p) for p in sources.rglob('*') if p.is_file()})
    hashes[str(central)] = sha(central)
    for p in sorted((ROOT / 'vlm_research/data_preparation/vqav2_testdev2015').glob('*.py')):
        hashes[str(p)] = sha(p)
    for p in (ROOT / 'vlm_research/benchmarks').glob('*'):
        if p.is_file() and p.name in ('official_registry.json', 'dataset_registry.py',
                                     'vqav2_submission.py', 'test_vqav2_testdev.py'):
            hashes[str(p)] = sha(p)
    for name in ('common.py', 'prepare.py', 'analyze.py', 'tests.py', 'README.md'):
        path = ROOT / 'vlm_research/seven_benchmark_quality' / name
        hashes[str(path)] = sha(path)
    for p in audit.iterdir():
        if p.is_file() and p.name not in ('file_hashes.json', 'disk_events.jsonl'):
            hashes[str(p)] = sha(p)
    write_json(audit / 'file_hashes.json', dict(algorithm='SHA256', files=hashes,
               self_excluded=True, dynamic_disk_events_excluded=True))
    print('INDEPENDENT_FINAL_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--sources', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--audit', type=Path, default=ROOT / 'vlm_research/results/vqav2_testdev2015_alignment')
    finalize(parser.parse_args())
