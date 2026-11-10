"""CPU-only source, full-cohort, resource and image-byte acceptance."""
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import shutil
import subprocess
import sys

from .common import *


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def partition(samples):
    # Union shared image bytes and circular groups before balancing predicted work.
    parent, circular = {}, {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for s in samples:
        image = s['image_sha256']
        find(image)
        if s['benchmark'] == 'MMBench':
            group = s['circular_group']
            if group in circular:
                left, right = find(image), find(circular[group])
                parent[max(left, right)] = min(left, right)
            circular[group] = image
    units = defaultdict(list)
    for s in samples:
        units[find(s['image_sha256'])].append(s)
    weight = lambda group: sum(576 + min(s['max_new_tokens'], 128)*8 for s in group)
    costs, counts, assignment = [0, 0], [0, 0], {}
    for _, group in sorted(units.items(), key=lambda x: (-weight(x[1]), x[0])):
        rank = min(range(2), key=lambda r: (costs[r], counts[r], r))
        costs[rank] += weight(group)
        counts[rank] += len(group)
        for s in group:
            assignment[s['key']] = rank
    for group in units.values():
        assert len({assignment[s['key']] for s in group}) == 1
    return dict(workers=2, assignment=assignment, question_counts=counts,
                estimated_work=costs, estimator='576 + 8*min(cap,128); no model-output information')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Already frozen; do not overwrite'
    now = datetime.now().astimezone().isoformat()
    hw = hardware()
    atomic(OUT / 'resource_precheck.json', dict(timestamp=now, hardware=hw,
        idle_gpu_uuids=idle_uuids(hw), gpu_status='READY' if len(idle_uuids(hw))>=2 else 'BLOCKED_OTHER_USERS',
        worktree=str(ROOT), disk_free_bytes=shutil.disk_usage(ROOT).free,
        no_external_process_terminated=True, old_ate_gpu_jobs_running=False,
        old_results_preserved=True, github_push_authorized=False))
    refs = []
    registry_path = 'vlm_research/results/featurize_dataset_backup_3090/dataset_registry.json'
    registry = {}
    for commit in (BASE, FASTV_ARCHIVE, GQA_REF, PAPER_REF):
        probe = subprocess.run(['git', 'cat-file', '-e', f'{commit}:{registry_path}'], cwd=ROOT,
                               capture_output=True)
        registry[commit] = 'PRESENT' if probe.returncode == 0 else 'ABSENT'
    atomic(OUT / 'dataset_registry_lookup.json', dict(requested_path=registry_path, commits=registry,
        fallback='Exact frozen manifests and current SHA256 audit; no invented registry'))
    for commit, rel in [
        (GQA_REF, 'vlm_research/results/featurize_4090_gqa_official_aligned_threeway/protocol.json'),
        (PAPER_REF, 'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench/paper_protocol.json'),
        (PAPER_REF, 'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench/score_summary.json'),
        (PAPER_REF, 'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench/model_acceptance.json')]:
        blob = git_bytes(commit, rel)
        path = OUT / 'references' / commit / Path(rel).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        refs.append(dict(commit=commit, git_path=rel, local_path=str(path), sha256=sha(path)))
    assert refs[0]['sha256'] == '1a7db2cacd807105fe6ccd1bf9c91652032050c242c2a49ddd5d7f490dcdeec4'
    manifest_path = OLD / 'sample_manifest.jsonl.gz'
    assert sha(manifest_path) == '2a83d76a622418c9925b3c4d3cd274c0ae8e320bb4065a85665ad586a0390943'
    samples = rows(manifest_path)
    assert Counter(s['benchmark'] for s in samples) == Counter(COUNTS)
    assert len(samples) == 35256 and len({s['key'] for s in samples}) == 35256
    assert all(s['max_new_tokens'] == CAPS[s['benchmark']] for s in samples)
    # Read-only source manifest reuse; no silent MME exclusion or substitution.
    shutil.copyfile(manifest_path, OUT / 'sample_manifest.jsonl.gz')
    images, conflicts = {}, []
    for s in samples:
        key = s['image_path']
        if key not in images:
            images[key] = dict(actual=sha(key), manifest_sha256=s['image_sha256'])
        assert images[key]['manifest_sha256'] == s['image_sha256']
        assert images[key]['actual'] == s['image_sha256'], (key, images[key])
        if (s['benchmark'] == 'MME' and s['image_id'] == 'landmark/03d5e3bfc958be38.jpg' and
                images[key]['actual'] != 'ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a'):
            conflicts.append(dict(key=s['key'], path=key, actual=images[key]['actual'],
                                  expected='ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a'))
    atomic(OUT / 'image_integrity.json', dict(images=images, verified_paths=len(images)))
    atomic(OUT / 'data_acceptance.json', dict(expected=COUNTS, restored=Counter(s['benchmark'] for s in samples),
        status='BLOCKED_MME_REFERENCE_BYTES' if conflicts else 'READY', blockers=conflicts,
        manifest_rows=35256, historical_control_records='NOT_RESTORED; aggregate references only',
        previous_binding_recheck=dict(path=str(OLD / 'protocol_recheck.json'),
                                      sha256=sha(OLD / 'protocol_recheck.json'))))
    acceptance = read(OUT / 'references' / PAPER_REF / 'model_acceptance.json')
    files = {}
    for name, expected in acceptance['files'].items():
        path = MODEL / name
        actual = dict(bytes=path.stat().st_size, sha256=sha(path))
        assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256'], name
        files[name] = actual
    for path in sorted(MODEL.iterdir()):
        if path.is_file() and not path.name.endswith('.safetensors'):
            files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    atomic(OUT / 'model_files_sha256.json', dict(path=str(MODEL), revision=MODEL.name, files=files))
    part = partition(samples)
    atomic(OUT / 'partition.json', part)
    test = subprocess.run([sys.executable, '-m', 'vlm_research.fastv_svh.tests'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
    (OUT / 'cpu_tests.log').write_text(test.stdout+test.stderr)
    atomic(OUT / 'cpu_tests.json', dict(passed=test.returncode==0, returncode=test.returncode,
                                      log_sha256=sha(OUT / 'cpu_tests.log')))
    assert test.returncode == 0, test.stderr
    protocol = dict(status='CPU_FROZEN_GPU_UNVERIFIED', created_at=now, algorithm='FastV-K3 + llava-SVH',
        base_commit=BASE, fastv_archive_commit=FASTV_ARCHIVE, fastv_official_commit=FASTV_OFFICIAL,
        model_revision=MODEL.name, model_path=str(MODEL), dtype='float16', attention='eager',
        torch='2.6.0+cu124', transformers='4.49.0', logits_to_keep=1, batch_size=1, greedy=True,
        condition_count=1, expected_counts=COUNTS, caps=CAPS, total=35256,
        visual_rows=dict(L0_L2=576, L3_L19=192, L20_L31=0),
        image_exit_before=20, original_rope_positions=True, answer_bypass_layers=[2,3,4,5],
        answer_image_access=False, R_writer_layers=list(range(20)), persistent_original_visual_kv=0,
        selector=dict(layer=2, original_last_query=True, independent_same_svh_input=True,
                      residual_injection=False, extra_full_q_k_and_qk_counted=True,
                      top_k=192, R='2/3', algorithm='unchanged official AST'),
        numerical_gate=GATE, sources=sources(), references=refs,
        manifest_sha256=sha(OUT / 'sample_manifest.jsonl.gz'), partition_sha256=sha(OUT / 'partition.json'),
        model_files_sha256=sha(OUT / 'model_files_sha256.json'),
        data_acceptance_sha256=sha(OUT / 'data_acceptance.json'),
        all_six_ready=not bool(conflicts), cpu_tests_passed=True, gpu_smoke='NOT_RUN',
        no_push=True, paired_controls='Require exact row-level input and protocol bindings; missing remains missing')
    atomic(OUT / 'protocol.json', protocol)
    atomic(OUT / 'completion.json', dict(status='BLOCKED', completed=0, requested=35256,
        cpu_tests='PASS', gpu_smoke='NOT_RUN', reasons=(
        (['NO_TWO_IDLE_GPUS'] if len(idle_uuids(hw)) < 2 else []) +
        (['MME_REFERENCE_IMAGE_SHA_MISMATCH'] if conflicts else []))))
    print('CPU_READY', 'ROWS', len(samples), 'PARTITIONS', part['question_counts'],
          'IDLE_GPUS', len(idle_uuids(hw)), 'MME_CONFLICT_ROWS', len(conflicts), flush=True)


if __name__ == '__main__':
    main()
