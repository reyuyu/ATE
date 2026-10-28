"""CPU acceptance, including actual Processor inputs, without loading model weights."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import subprocess
import unittest

from PIL import Image

from .dataset_registry import OUTPUT, freeze_json, load_manifest, sha256
from .protocols import MODEL_PATH, prompt_text
from .validate_datasets import validate


def model_file_audit():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/model.json').read_text())
    rows = []
    for record in config['files']:
        p = Path(MODEL_PATH)/record['name']
        actual = sha256(p)
        if p.stat().st_size != record['bytes'] or (record['lfs_sha256'] and actual != record['lfs_sha256']):
            raise ValueError(f'Pinned model resource changed: {p}')
        rows.append(dict(name=p.name, bytes=p.stat().st_size, sha256=actual, lfs_sha256=record['lfs_sha256']))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    output = args.output/'tests'
    if (output/'validation.json').exists():
        raise ValueError('Acceptance exists; inspect/revalidate, do not overwrite')
    import torch
    import transformers
    import vlm_common
    vlm_common.default_model = lambda: MODEL_PATH
    if torch.cuda.is_available():
        raise ValueError('CPU acceptance must hide all GPUs')
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
        focused = unittest.TextTestRunner(stream=logs, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_benchmark_suite'))
        complete = unittest.TextTestRunner(stream=logs, verbosity=2).run(unittest.defaultTestLoader.discover('.', pattern='test_*.py'))
    output.mkdir(parents=True, exist_ok=True)
    (output/'cpu_console.txt').write_text(logs.getvalue())
    if not focused.wasSuccessful() or not complete.wasSuccessful() or focused.skipped or complete.skipped:
        freeze_json(output/'validation.json', {'passed': False, 'reason': 'CPU regression failure/skip',
            'focused_tests': focused.testsRun, 'complete_tests': complete.testsRun})
        raise AssertionError('CPU regression failure; inspect cpu_console.txt')
    data = validate(args.output)
    processor = transformers.AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.patch_size = 14
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = 'default'
    inputs_checked = []
    from physical_exit_pilot import candidate_inputs
    from .smoke import freeze_plan
    smoke_plan = freeze_plan(args.output)
    for manifest_path in sorted((args.output/'manifests').glob('*.json')):
        manifest = load_manifest(manifest_path, verify_images=False)
        if manifest['status'] != 'ready_native_full':
            continue
        lookup = {r['question_id']: r for r in manifest['samples']}
        for qid in smoke_plan['datasets'].get(manifest_path.stem, []):
            sample = lookup[qid]
            mapping = sample['metadata']['input_mapping']
            with Image.open(sample['image_path']) as im:
                image = im.convert('RGB')
            inputs = processor(images=image, text=mapping['rendered_prompt'], return_tensors='pt')
            if inputs['input_ids'][0].tolist() != mapping['prompt_ids']:
                raise AssertionError('Actual image Processor IDs differ from CPU freeze')
            if inputs['pixel_values'].shape != (1, 3, 336, 336) or not torch.isfinite(inputs['pixel_values']).all():
                raise AssertionError('Invalid native image processing')
            lengths = {}
            for label, (candidate, records) in zip(sample['candidates'] or [], candidate_inputs(processor, inputs, sample['candidates']) if sample['candidates'] else []):
                if records != mapping['answer_mapping'][label]:
                    raise AssertionError('Multi-token mapping differs from existing scorer')
                if not torch.equal(candidate['input_ids'][:, :inputs['input_ids'].shape[1]], inputs['input_ids']):
                    raise AssertionError('Candidate prompt prefix changed')
                lengths[label] = len(records)
            inputs_checked.append({'dataset': manifest_path.stem, 'question_id': qid,
                                   'actual_processor_ids_exact': True, 'visual_tokens': 576, 'candidate_lengths': lengths})
    freeze_json(output/'validation.json', {'passed': True, 'focused_tests': focused.testsRun,
        'complete_tests': complete.testsRun, 'skipped': 0, 'dataset_validation': data, 'actual_processor_inputs': inputs_checked,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                        'python_executable': __import__('sys').executable},
        'model_files': model_file_audit(),
        'console_sha256': sha256(output/'cpu_console.txt'),
        'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                         for p in Path(__file__).resolve().parents[1].rglob('*.py') if 'results' not in p.parts}})
    print(json.dumps({'passed': True, 'focused_tests': focused.testsRun, 'complete_tests': complete.testsRun,
                      'actual_processor_inputs': len(inputs_checked)}, indent=2))


if __name__ == '__main__':
    main()
