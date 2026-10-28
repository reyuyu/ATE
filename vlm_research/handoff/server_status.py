"""One-shot recovery status; no polling, training, GPU forward or historical edits."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--venv-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new snapshot path, never overwrite an old audit')
    manifest = load(ROOT / 'results/phase3_2_fixed_k_exit/manifest.json')
    roi = load(ROOT / 'results/phase3_4_roi_sink_diagnostic/manifest.json')
    image_root = args.resources / 'data/image-tasks/MMEB/GQA'
    unique = {s['img_id']: s for s in manifest['samples'] + manifest['validation_samples']}
    cache_paths = sorted((ROOT / 'results/phase3_4_roi_sink_diagnostic/attention_rows').glob('*_task.npz'))
    shapes = {}
    for path in cache_paths:
        with np.load(path, allow_pickle=False) as cache:
            shapes[path.stem.removesuffix('_task')] = list(cache['attention'].shape)
    assert len(unique) == 252 and len(cache_paths) == 53
    assert set(shapes) == set(roi['main_question_ids'] + roi['case_question_ids'])
    assert all(shape == [32, 32, 2, 576] for shape in shapes.values())
    rows32 = list(csv.DictReader((ROOT / 'results/phase3_2_fixed_k_exit/per_sample_results.csv').open()))
    rows33 = list(csv.DictReader((ROOT / 'results/phase3_3_directed_suffix/per_sample_results.csv').open()))
    source = list(csv.DictReader((ROOT / 'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv').open()))
    assert len(rows32) == 1920 and len(rows33) == 3120 and len(source) == 964
    packages = subprocess.check_output([str(args.venv_python), '-m', 'pip', 'list', '--format=json'], text=True)
    config = load(ROOT / 'configs/model.json')
    model = args.resources / 'models/llava-1.5-7b-hf' / config['revision']
    image_verified = any(load(path).get('passed') and load(path).get('images_checked') == 252
                         and load(path).get('source_images_checked') == 785
                         for path in (args.resources / 'audit').glob('recovery*.json'))
    cpu_completed = any(all((attempt / f'cpu_{suite}/validation.json').is_file()
                            and load(attempt / f'cpu_{suite}/validation.json').get('passed')
                            for suite in ['inheritance', 'all'])
                        for attempt in (args.resources / 'audit').glob('pinned_env_20261028*'))
    condition_counts = [len(load(path)) for path in
                        (ROOT / 'results/attention_inheritance_15133706').glob('remote_*/per_condition_results.json')]
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                  status='recovery_in_progress_no_new_gpu_result', status_script_python=platform.python_version(),
                  baseline_commit='7970be981e3462560cb66d83e1014eaa1b1ebdae',
                  current_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  historical_manifest_sha256=hashlib.sha256((ROOT / 'results/phase3_2_fixed_k_exit/manifest.json').read_bytes()).hexdigest(),
                  phase32_images=len(manifest['samples']), pilot_images=len(manifest['validation_samples']),
                  phase32_labels=dict(Counter(s['answer'] for s in manifest['samples'])),
                  phase32_rows=len(rows32), phase33_rows=len(rows33),
                  choose_rel_questions=len(source), choose_rel_unique_images=len({s['img_id'] for s in source}),
                  cached_task_attention_shapes=shapes, attention_cache_images=53,
                  main_images=len(roi['main_question_ids']), isolated_cases=len(roi['case_question_ids']),
                  frozen_images_required=len(unique), frozen_images_present=sum((image_root / f'image_{img_id}.jpg').is_file() for img_id in unique),
                  full_image_hash_audit_completed=image_verified, image_root=str(image_root), model_root=str(model),
                  fixed_model_files={f['name']: dict(exists=(model / f['name']).is_file(), expected_bytes=f['bytes']) for f in config['files']},
                  fixed_weight_hash_audit_completed=(args.resources / 'audit/model_files_verified.json').is_file(),
                  venv_python=str(args.venv_python), venv_packages=json.loads(packages),
                  original_7_and_101_tests_on_fixed_env_completed=cpu_completed,
                  new_single_case_gpu_conditions_completed=max(condition_counts, default=0),
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free,memory.used,utilization.gpu', '--format=csv'], text=True),
                  note='One-time snapshot, not a watcher. Presence is not image/model hash certification. Historical Full/L23 lambda0 are not new server results. No automatic GPU launch.')
    if condition_counts:
        result['status'] = 'new_single_case_records_present_inspect_run_validation'
    result['new_code_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['attention_inheritance_remote.py', 'cpu_tests_remote.py', 'cpu_remote_tools_tests.py',
                     'handoff/restore_checkout.py', 'handoff/restore_server_resources.py',
                     'handoff/reuse_cuda_wheels.py', 'handoff/install_server_env.sh',
                     'handoff/server_status.py', 'handoff/verify_history_remote.py']}
    pin = args.resources / 'data/mmeb_download_source.json'
    if pin.exists():
        result['data_recovery_pin'] = load(pin)
    result['units'] = {}
    for unit in ['ate-restore-data-range-20261028', 'ate-restore-model-range-20261028', 'ate-environment-cpu-paths-20261028']:
        state = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result'], capture_output=True, text=True)
        result['units'][unit] = state.stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Snapshot written:', args.output)


if __name__ == '__main__':
    main()
