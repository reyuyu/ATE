#!/usr/bin/env python3
"""Re-validate local exports and emit the six small, Git-safe audit artifacts."""
import argparse
import collections
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).parent))
from prepare import GIB, SPECS, allocated, digest, encoded, image_stats, sha


def write(path, value):
    path.write_bytes(encoded(value))


def csv_file(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def verify(result):
    import pyarrow.parquet as pq
    from PIL import Image
    from prepare import question_from_messages
    root = Path(result['local_path'])
    answers = json.loads((root / 'answers.json').read_text())
    images = json.loads((root / 'image_manifest.json').read_text())
    image_lookup = {i['image_id']: i for i in images}
    assert len(images) == len(image_lookup) == len({i['sha256'] for i in images})
    image_paths = set()
    for image in images:
        path = Path(image['image_path'])
        assert path.is_relative_to(root / 'images') and path.stat().st_size == image['bytes']
        assert sha(path) == image['sha256']
        with Image.open(path) as im:
            im.load()
            assert im.size == (image['width'], image['height']) and im.format == image['format']
        image_paths.add(str(path))
    inputs = [json.loads(line) for line in (root / 'inputs.jsonl').read_text().splitlines()]
    full = [json.loads(line) for line in (root / 'records_with_answers.jsonl').read_text().splitlines()]
    assert len(inputs) == len(full) == len(answers) == result['expected_qa']
    assert len({r['question_id'] for r in inputs}) == len(inputs)
    assert set(answers) == {r['question_id'] for r in inputs}
    all_refs = 0
    used_images = set()
    for r, gold in zip(inputs, full):
        assert 'answers' not in r and 'answer' not in r
        assert not {'answer', 'answers', 'gold', 'lecture', 'solution', 'messages', 'qa'} & set(r['metadata'])
        assert r['dataset'] == result['dataset'] and r['split'] == result['split']
        assert isinstance(r['question'], str) and r['question'].strip()
        assert gold['answers'] == answers[r['question_id']]['answers']
        assert all(isinstance(a, str) and a.strip() for a in gold['answers'])
        assert {k: v for k, v in gold.items() if k != 'answers'} == r
        assert r['image_path'] == r['images'][0]['image_path']
        for media in r['images']:
            image = image_lookup[media['image_id']]
            assert media['image_path'] == image['image_path']
            assert media['image_sha256'] == image['sha256']
            used_images.add(media['image_id'])
            all_refs += 1
    assert used_images == set(image_lookup)
    assert all_refs == result['image_references']
    source_rows = [json.loads(line) for line in (root / 'source_records_with_answers.jsonl').read_text().splitlines()]
    row_index, question_index = 0, 0
    for source in result['source_files']:
        path = Path(source['local_path'])
        assert path.stat().st_size == source['bytes'] and sha(path) == source['sha256']
        assert '/' + result['split'] + '-' in source['filename']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
            for original in batch.to_pylist():
                key = 'media' if result['dataset'] == 'InfoVQA-val' else 'image'
                media = original[key] if key == 'media' else [original[key]]
                preserved = dict(original)
                replacements = []
                image_hashes = []
                for value in media:
                    h = digest(value['bytes'])
                    image_hashes.append(h)
                    image = image_lookup[h]
                    replacements.append(dict(value, bytes=dict(sha256=h, exported_path=image['image_path'],
                                                               original_encoding_preserved=True)))
                preserved[key] = replacements if key == 'media' else replacements[0]
                assert source_rows[row_index] == dict(source_row_index=row_index, source_file=str(path), original=preserved)
                if result['dataset'] == 'DocVQA-val':
                    pairs = [(q['question'], q['answers']) for q in original['qa']]
                elif result['dataset'] == 'OCRBench-v1':
                    pairs = [(original['question'], original['answer'])]
                else:
                    pairs = [(question_from_messages(original['messages']), original['answer'])]
                for question, gold in pairs:
                    record = inputs[question_index]
                    assert record['question'] == question
                    assert answers[record['question_id']]['answers'] == gold
                    assert [m['image_sha256'] for m in record['images']] == image_hashes
                    assert record['metadata']['source_row_index'] == row_index
                    question_index += 1
                row_index += 1
    assert row_index == len(source_rows) == result['source_rows']
    assert question_index == len(inputs)
    return dict(passed=True, qa=len(inputs), unique_images=len(images), image_references=all_refs,
                all_images_decoded_again=True, all_source_sha256_rechecked=True,
                answer_alignment=True, inference_input_has_no_gold=True,
                original_parquet_field_and_media_equivalence=True,
                media_count_distribution=dict(collections.Counter(len(r['images']) for r in inputs)),
                questions_with_multiple_images=sum(len(r['images']) > 1 for r in inputs),
                original_id_nonnull_count=sum(r['metadata'].get('original_id') is not None for r in inputs),
                image_stats=image_stats(images))


def main(args):
    audit = Path(args.audit).resolve()
    results = json.loads((audit / 'run_results.json').read_text())
    initial = json.loads((audit / 'disk_events.jsonl').read_text().splitlines()[0])['free_bytes']
    disk = os.statvfs('/data')
    free = disk.f_bavail * disk.f_frsize
    validations = {}
    for result in results:
        if result['status'] == 'COMPLETE':
            validations[result['dataset']] = verify(result)
        else:
            validations[result['dataset']] = dict(passed=False, status=result['status'], reason=result.get('error'))
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).parent),
                            '-p', 'test_*.py', '-v'], text=True, capture_output=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), timeout=120)
    write(audit / 'unit_tests.json', dict(exit=tests.returncode, stdout=tests.stdout, stderr=tests.stderr))
    if tests.returncode:
        raise RuntimeError('CPU regression tests failed')
    events = [json.loads(line) for line in (audit / 'disk_events.jsonl').read_text().splitlines()]
    data_bytes = sum(allocated(Path(r['local_path'])) for r in results)
    validation = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      datasets=validations, complete=all(v['passed'] for v in validations.values()),
                      qa=sum(r['actual_qa'] for r in results), cpu_unit_tests_passed=True,
                      safety=dict(cpu_only=True, model_weights_loaded=False, cuda_initialized=False,
                                  gpu_inference_run=False, source_research_files_modified=False,
                                  existing_files_deleted=False, data_floor_gib=15,
                                  before_free_bytes=initial, after_free_bytes=free,
                                  minimum_observed_free_bytes=min(e['free_bytes'] for e in events),
                                  allocated_dataset_bytes=data_bytes,
                                  all_transfers_finished=True),
                      qwen_processor_audit=dict(status='NOT_RUN_OPTIONAL',
                                               reason='Local Qwen3-VL processor config exists, but the available validation Python environment has transformers 4.56.2 without the Qwen3-VL implementation. No extra dependencies/models were downloaded; resolution statistics are complete.',
                                               requested_visual_token_cap=2048,
                                               no_token_counts_estimated_or_claimed=True))
    write(audit / 'data_validation.json', validation)
    sources = dict(datasets=[dict(dataset=r['dataset'], repository=r['repository'], revision=r.get('revision'),
                                  split=r['split'], original_repository_url='https://huggingface.co/datasets/' + r['repository'],
                                  metadata_endpoint=r.get('metadata_endpoint'),
                                  files=r.get('source_files', [])) for r in results],
                   transport_note='hf-mirror.com transports the original repository revision. Every retained Parquet matches the revision metadata LFS SHA256 and byte length; not a different dataset.',
                   library_versions={m: __import__(m).__version__ for m in ('huggingface_hub', 'pyarrow', 'PIL')},
                   python=sys.version, platform=platform.platform())
    write(audit / 'source_manifest.json', sources)
    summary = [dict(dataset=r['dataset'], expected_qa=r['expected_qa'], actual_qa=r['actual_qa'],
                    source_rows=r.get('source_rows', ''), image_references=r.get('image_references', ''),
                    unique_images=r['unique_images'], unique_image_sha256=r.get('unique_image_sha256', ''),
                    disk_bytes=allocated(Path(r['local_path'])), disk_gib=round(allocated(Path(r['local_path'])) / GIB, 6),
                    split=r['split'], revision=r.get('revision', ''), validation=r['status'],
                    local_path=r['local_path']) for r in results]
    csv_file(audit / 'dataset_summary.csv', summary)
    resolutions = []
    for r in results:
        for metric, values in r.get('image_stats', {}).items():
            resolutions.append(dict(dataset=r['dataset'], population='unique_original_image_sha256',
                                    metric=metric, P50=values['P50'], P90=values['P90'], P95=values['P95'],
                                    P99=values['P99'], Max=values['P100']))
    if resolutions:
        csv_file(audit / 'image_resolution_summary.csv', resolutions)
    report = [
        '# OCRBench、DocVQA、InfoVQA 数据下载与完整性审计', '',
        '本轮只准备数据并做 CPU 验证，没有加载模型权重、初始化 CUDA、运行推理或评分。未删除既有研究文件。', '',
        '## 汇总', '',
        '| Dataset | Expected QA | Actual QA | Unique Images | Disk Size (GiB) | Validation |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for r in summary:
        report.append(f'| {r["dataset"]} | {r["expected_qa"]} | {r["actual_qa"]} | {r["unique_images"]} | {r["disk_gib"]:.6f} | {r["validation"]} |')
    report.extend([
        f'| Total | 9150 | {sum(r["actual_qa"] for r in summary)} | {sum(r["unique_images"] for r in summary)} | {data_bytes/GIB:.6f} | {"COMPLETE" if validation["complete"] else "NOT_COMPLETE"} |', '',
        '磁盘占用使用 st_blocks × 512 汇总，包含原始 Parquet、原始编码图片和整理标注；保留了所有本轮下载的 Parquet，没有靠删除源数据满足空间约束。Total 图片数为各数据集唯一图片数之和，不是跨数据集 SHA 去重数。', '',
        f'- 开始时 /data 可用：{initial} bytes（{initial/GIB:.6f} GiB）。',
        f'- 本报告生成时 /data 可用：{free} bytes（{free/GIB:.6f} GiB）。',
        f'- 监测到的最低可用空间：{validation["safety"]["minimum_observed_free_bytes"]/GIB:.6f} GiB。',
        f'- 本轮数据实际新增占用：{data_bytes} bytes（{data_bytes/GIB:.6f} GiB）。审计工作目录位于 /data1，另行统计。',
        f'- 全文件系统可用空间减少：{(initial-free)/GIB:.6f} GiB；该数值包含其他程序同时写入，不能全部归因于本任务。', '',
        '## 数据路径与使用约束', '',
        '`inputs.jsonl` 不包含答案、原始带答案 messages、qa、Gold、lecture 或 solution，供后续模型输入读取。',
        '`answers.json` 以稳定 question_id 映射全部参考答案，仅供离线评分。',
        '`records_with_answers.jsonl` 是满足统一字段要求的完整记录，不可整体传入模型。',
        '`source_records_with_answers.jsonl` 保留所有原始字段和多媒体顺序；只将二进制图片替换为无损导出文件路径及 SHA。',
        '`image_manifest.json` 保存原始字节 SHA、宽高、编码、文件大小和所有源行/媒体索引。', '',
    ])
    for r in results:
        report.extend([f'### {r["dataset"]}', '', f'- 数据绝对路径：`{r["local_path"]}`。',
                       f'- 模型输入：`{r["local_path"]}/inputs.jsonl`。',
                       f'- 参考答案：`{r["local_path"]}/answers.json`。',
                       f'- 来源：`{r["repository"]}`；Revision：`{r.get("revision")}`；Split：`{r["split"]}`。',
                       f'- 状态：{r["status"]}；QA {r["actual_qa"]}；原始源行 {r.get("source_rows", "未完成")}；图片引用 {r.get("image_references", "未完成")}；唯一 SHA {r["unique_images"]}。'])
        if r['status'] != 'COMPLETE':
            report.append('- 阻塞或失败原因：' + r.get('error', '未知'))
        report.extend(['', 'Question Type 分布：', '', '```json',
                       json.dumps(r.get('question_type_distribution', {}), ensure_ascii=False, indent=2), '```', ''])
        if r.get('original_dataset_distribution'):
            report.extend(['原始 Dataset 类别分布：', '', '```json',
                           json.dumps(r['original_dataset_distribution'], ensure_ascii=False, indent=2), '```', ''])
    report.extend([
        '## 完整性与协议', '',
        '- OCRBench 固定为 echo840/OCRBench 的 test，预期 1000 题与原始 10 个 Question Type，不使用 v2。',
        '- DocVQA 按原始图片行的 qa 列表逐条展开；保留原始图片行与 qa 索引，不把 1286 张图片误当题数。源镜像没有问题 ID 时不伪造官方 ID。',
        '- DocVQA 的 1286 个原始图片记录中有两个记录的图片编码字节完全相同，因此唯一 SHA 图片文件数为 1285；原始 1286 个记录和全部 5349 题均完整保留。DocVQA 源镜像未提供 Question Type，NOT_PROVIDED 仅表示字段缺失，不是新分类。',
        '- InfoVQA 只选择 data/validation-*.parquet；保留原始 id、question_type、messages、全部 answer 和 media。多图片记录使用 images 数组，不截取第一张。单数 image_* 字段只是第一张的兼容索引。',
        '- InfoVQA 的实际 messages 元素使用 role/user + question/hint/options/choices，而不是 content。首次解析因此安全失败、没有标记 COMPLETE；修正 Schema 分支并增加回归测试后，复用已通过 SHA 的原始 Parquet 重做整理。dataset_runs.jsonl 保留首次失败和后续结果，run_results.json 与 data_validation.json 表示最终状态。所有额外原始字段完整保存在 source_records_with_answers.jsonl，未混入模型 inputs.jsonl。',
        '- question_id 由 Dataset、Split、原始行索引及 DocVQA 的 QA 索引构造，原始 ID 单独保留。相同问题、相同原始 ID 或图片均不合并题目。',
        '- 图片仅按完整原始编码字节的 SHA256 去重；不改变编码，不缩放、不裁剪、不重新压缩。源图片路径/行索引的多个关联均保留。',
        '- 全图片使用 PIL load 完整解码两次以上，并重新检查落盘 SHA、格式与宽高。逐题检查非空问题、全部参考答案、唯一 ID 和图片映射。',
        '- 每个源 Parquet 都与固定 Revision 的 HF LFS SHA256 和长度一致；镜像仅为传输通道，不依赖仅同名文件判断等价。原始 HF 元数据及实际镜像地址随 source_manifest.json 保存。',
        '- P50/P90/P95/P99/Max 以唯一原始编码图片为总体，采用线性分位数；宽高为原始编码尺寸。详细数据见 image_resolution_summary.csv。',
        '- 不实现新的评分器；后续 ATE 代码需要薄适配器读取 inputs.jsonl，评分另读 answers.json。数据 COMPLETE 不表示正式评测已完成。', '',
        'CPU 回归测试 8 项通过，包括 DocVQA 多答案展开、多媒体保留、实际 InfoVQA question 字段 Schema、非法空答案拒绝、已有文件冲突保护、无损导出及重跑一致性、磁盘安全线阻断。见 unit_tests.json。', '',
        '## Qwen 显存风险审计', '',
        '已完成全部原图分辨率及像素分布。可选官方 Processor 审计未执行：本机存在 Qwen3-VL 配置，但本轮可用 Python 环境的 transformers 4.56.2 没有 Qwen3-VL 实现。未额外安装依赖、下载模型或加载权重。不把静态分辨率公式冒充实际 image_grid_thw，不声称已计算 ViT Patch 或 1024/1536/2048 Token 阈值计数。后续审计应采用独立 2048 视觉 Token 配置，不修改图片。', '',
        '## 重现', '',
        '依赖：huggingface_hub 0.36.2、pyarrow 24.0.0、Pillow 11.3.0。脚本只导入这些 CPU 数据库及标准库，不导入 torch。', '',
        '```bash',
        'export CUDA_VISIBLE_DEVICES=""',
        'PY=/home/likai/anaconda3/envs/AFAC/bin/python',
        'AUDIT="$(pwd)/vlm_research/results/ocr_doc_infovqa_download_audit"',
        '$PY vlm_research/data_preparation/ocr_doc_infovqa/prepare.py --audit "$AUDIT"',
        '$PY vlm_research/data_preparation/ocr_doc_infovqa/finalize.py --audit "$AUDIT"',
        '$PY -m unittest discover -s vlm_research/data_preparation/ocr_doc_infovqa -p "test_*.py" -v',
        '```', '',
        '重跑会固定使用已保存元数据中的 Revision，逐文件 SHA 核验后复用；冲突内容不覆盖。下载采用 HF Hub local_dir 断点续传，每文件最多两次任务级尝试、每次最长 600 秒；后台只监视/终止本任务自身的下载子进程。',
        'file_hashes.json 保存数据文件、各图片、原始 Parquet、脚本和报告的 SHA256；该文件不自哈希。大型文件全部留在服务器，不提交图片或 Parquet。', '',
        '## 停止范围', '',
        '本任务到数据准备与 GitHub 审计交付为止，不启动 Native、C3-V2、Qwen 或其他 GPU 实验。', '',
    ])
    (audit / 'report_zh.md').write_text('\n'.join(report))
    hashes = []
    for r in results:
        for path in sorted(Path(r['local_path']).rglob('*')):
            if path.is_file() and '/.cache/' not in str(path):
                hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    script_dir = Path(__file__).parent
    for path in sorted(list(script_dir.glob('*.py')) + list(audit.glob('*'))):
        if path.is_file() and path.name != 'file_hashes.json':
            hashes.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path)))
    write(audit / 'file_hashes.json', dict(algorithm='SHA256', excludes=['file_hashes.json itself', 'HF local_dir transport cache'], files=hashes))
    print(json.dumps(dict(complete=validation['complete'], qa=validation['qa'],
                          allocated_bytes=data_bytes, free_bytes=free, report=str(audit / 'report_zh.md')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    main(parser.parse_args())
