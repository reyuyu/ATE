"""CPU-only freeze of the one authorized experiment and the MME supplement."""
from collections import Counter, defaultdict
from datetime import datetime
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

from .common import *


def refreeze_failed_smoke():
    """Archive an unlaunched GPU gate; never rebind completed MME predictions."""
    old = read(OUT / 'protocol.json')
    assert not (OUT / 'gpu_smoke.json').exists()
    assert not list(OUT.glob('worker*/results.sqlite'))
    assert (OUT / 'gpu_smoke/STOP.json').exists()
    failure = read(OUT / 'gpu_smoke/STOP.json')
    assert 'Expected all tensors to be on the same device' in failure['traceback']
    assert read(PRIOR / 'complete_with_mme_authorized/completion.json')['completed'] == 35256
    check_idle(BASE_GPUS)
    assert old['manifest_sha256'] == sha(MANIFEST) and old['partition_sha256'] == sha(PARTITION)
    for name, expected in old['model_file_stats'].items():
        stat = (MODEL / name).stat()
        assert [stat.st_size, stat.st_mtime_ns, stat.st_ino] == expected
    current = sources()
    changed = [name for name in current if current[name] != old['sources'].get(name)]
    permitted = {'vlm_research/sparsevlm_svh/' + name for name in
                 ('official.py', 'engine.py', 'tests.py', 'prepare.py')}
    assert set(current) == set(old['sources']) and set(changed) <= permitted, changed
    cpu = subprocess.run([sys.executable, '-m', 'vlm_research.sparsevlm_svh.tests'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), capture_output=True, text=True)
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    if cpu.returncode:
        (OUT / f'prepare_cpu_failure_{stamp}.log').write_text(cpu.stdout + cpu.stderr)
        raise RuntimeError(cpu.stdout + cpu.stderr)
    archive = OUT / 'revisions' / stamp
    archive.mkdir(parents=True)
    for name in ('protocol.json', 'validation.json', 'completion.json', 'cpu_tests.json',
                 'cpu_tests.log', 'smoke0.log', 'gpu_smoke', 'launch.json',
                 'gpu_release_audit.json', 'manual_resume_20261110.json'):
        path = OUT / name
        if path.exists():
            path.rename(archive / name)
    (OUT / 'cpu_tests.log').write_text(cpu.stdout + cpu.stderr)
    atomic(OUT / 'cpu_tests.json', dict(passed=cpu.returncode == 0, returncode=cpu.returncode,
        log_sha256=sha(OUT / 'cpu_tests.log'), gpu_loaded=False))
    recovery = dict(previous_protocol_sha256=digest(old), archived_gate=str(archive),
        preserved_failure=failure, changed_sources=changed, production_math_changed=False,
        compatibility_fix='Official reference-only bias factory allocated on query.device; original AST unchanged',
        thresholds_unchanged=True, old_MME_protocol_and_predictions_unchanged=True,
        no_formal_sparse_predictions_existed=True)
    atomic(OUT / 'reference_device_recovery.json', recovery)
    protocol = dict(old, sources=current, cpu_tests_sha256=sha(OUT / 'cpu_tests.json'),
        reference_device_recovery_sha256=sha(OUT / 'reference_device_recovery.json'),
        created_at=datetime.now().astimezone().isoformat())
    atomic(OUT / 'protocol.json', protocol)
    atomic(OUT / 'validation.json', dict(status='CPU_PASS_GPU_GATE_PENDING', cpu_tests_passed=True,
        actual_gpu_images=0, prior_smoke_failure_archived=True, protocol_sha256=digest(protocol),
        no_threshold_change=True))
    atomic(OUT / 'completion.json', dict(status='GPU_SMOKE_PENDING', completed=0, requested=35256,
        gpu_smoke='RETRY_AFTER_REFERENCE_DEVICE_FIX', no_push=True))
    print('REFROZEN_REFERENCE_DEVICE_ONLY', digest(protocol), flush=True)


def main(refreeze_unlaunched=False, recover_failed_unlaunched=False):
    recovery = None
    if (OUT / 'protocol.json').exists() and (refreeze_unlaunched or recover_failed_unlaunched):
        assert not (OUT / 'gpu_smoke.json').exists()
        assert not list(MME_OUT.glob('worker*/results.sqlite')) and not list(OUT.glob('worker*/results.sqlite'))
        assert not (MME_OUT / 'launch.json').exists() and not (OUT / 'launch.json').exists()
        if recover_failed_unlaunched:
            from .handoff import prior_state, predecessor_complete
            status = read(OUT / 'handoff_status.json')
            assert status['status'] == 'STOPPED_WITH_EVIDENCE'
            state = prior_state()
            assert state['ActiveState'] == 'failed' and state['ExecMainStatus'] == '1', state
            assert predecessor_complete() == 32882
            recovery = dict(old_protocol_sha256=digest(read(OUT / 'protocol.json')),
                failed_handoff_sha256=sha(OUT / 'handoff_status.json'), predecessor_state=state,
                reason='Post-generation official scorer input-mutation assertion; no new GPU run existed',
                no_algorithm_or_threshold_change=True, no_existing_predictions_rewritten=True)
        else:
            assert not (OUT / 'handoff_status.json').exists()
        archive = OUT / 'revisions' / datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
        archive.mkdir(parents=True)
        for path in list(OUT.iterdir()):
            if path.is_file():
                path.rename(archive / path.name)
        if recovery:
            recovery['archived_receipts'] = str(archive)
    assert not (OUT / 'protocol.json').exists(), 'Never overwrite a frozen run'
    OUT.mkdir(parents=True, exist_ok=True)
    parent = read(PRIOR / 'protocol.json')
    assert parent['sources'] == inherited_sources(), 'Read-only SVH/FastV source does not match predecessor'
    assert sha(MANIFEST) == parent['manifest_sha256'] and sha(PARTITION) == parent['partition_sha256']
    samples = rows(MANIFEST)
    assert Counter(s['benchmark'] for s in samples) == COUNTS
    assert len({s['key'] for s in samples}) == 35256
    assignment = read(PARTITION)['assignment']
    clusters = defaultdict(set)
    circular = defaultdict(set)
    pairs = defaultdict(set)
    for s in samples:
        clusters[s['image_sha256']].add(assignment[s['key']])
        if s['benchmark'] == 'MMBench':
            circular[s['circular_group']].add(assignment[s['key']])
        if s['benchmark'] == 'MME':
            pairs[s['pair_id']].add(assignment[s['key']])
    assert all(len(v) == 1 for v in (*clusters.values(), *circular.values(), *pairs.values()))
    acceptance = read(PRIOR / 'data_acceptance.json')
    assert len(acceptance['blockers']) == 2
    exceptions = []
    from PIL import Image
    for b in acceptance['blockers']:
        assert b['key'].startswith('MME:')
        assert sha(b['path']) == b['actual']
        sample = next(s for s in samples if s['key'] == b['key'])
        assert sample['image_sha256'] == b['actual']
        with Image.open(b['path']) as image:
            image.load()
            exceptions.append(dict(**b, mode=image.mode, size=image.size,
                accepted_by_current_user=True, strict_historical_byte_match=False))
    authorization = dict(status='USER_AUTHORIZED_MME_CURRENT_BYTES',
        instruction='允许评测MME，此外下一个任务提前告诉你，同步完上一个问题立马跑这个实验',
        frozen_manifest_unchanged=True, dropped_questions=0, mme_questions=2374,
        historical_byte_exceptions=exceptions, affects_unique_images=1, affects_questions=2,
        no_claim_of_strict_paired_historical_MME=True,
        recorded_at=datetime.now().astimezone().isoformat())
    atomic(OUT / 'mme_authorization.json', authorization)
    cpu = subprocess.run([sys.executable, '-m', 'vlm_research.sparsevlm_svh.tests'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1'), capture_output=True, text=True)
    (OUT / 'cpu_tests.log').write_text(cpu.stdout + cpu.stderr)
    atomic(OUT / 'cpu_tests.json', dict(passed=cpu.returncode == 0, returncode=cpu.returncode,
        log_sha256=sha(OUT / 'cpu_tests.log'), gpu_loaded=False,
        development_failures=[dict(kind='missing_einops', resolution='Official pin 0.6.1 installed only in isolated .runtime-deps'),
            dict(kind='inherited_test_runner_grad_enabled', resolution='Explicit inference-only CPU test entry; no engine/tolerance change')]))
    assert cpu.returncode == 0, cpu.stdout + cpu.stderr
    models = read(PRIOR / 'model_files_sha256.json')
    model_stats = {}
    for name, expected in models['files'].items():
        path = MODEL / name
        before = path.stat()
        assert before.st_size == expected['bytes'] and sha(path) == expected['sha256'], name
        after = path.stat()
        assert before.st_mtime_ns == after.st_mtime_ns and before.st_ino == after.st_ino
        model_stats[name] = [after.st_size, after.st_mtime_ns, after.st_ino]
    atomic(OUT / 'model_files_sha256.json', models)
    import einops
    import torch
    assert einops.__version__ == '0.6.1'
    assert torch.__version__ == '2.6.0+cu124' and importlib.metadata.version('transformers') == '4.49.0'
    assert not torch.cuda.is_initialized()
    artifact = 'vlm_research/results/unified_retain128_e2e32_4090/configuration_audit.json'
    retain_commit = '3f51c84bd469dd6a240fd2bedaec9ca1af361d55'
    audit = subprocess.check_output(['git', 'show', f'{retain_commit}:{artifact}'], cwd=ROOT)
    atomic(OUT / 'retain128_reference_audit.json', dict(commit=retain_commit, path=artifact,
        content=json.loads(audit), note='Reference audit only: this run uses official192, never128'))
    atomic(OUT / 'environment.json', dict(python=sys.executable, torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'), einops=einops.__version__,
        einops_path=einops.__file__, no_shared_environment_change=True, hardware=hardware(),
        data_free_bytes=shutil.disk_usage(OUT).free, cuda_initialized=False))
    base = dict(sources=sources(), manifest_path=str(MANIFEST), manifest_sha256=sha(MANIFEST),
        partition_path=str(PARTITION), partition_sha256=sha(PARTITION),
        parent_protocol_sha256=digest(parent), model_revision=parent['model_revision'],
        model_path=str(MODEL), model_files_sha256=sha(OUT / 'model_files_sha256.json'),
        model_file_stats=model_stats, dtype='float16', attention='eager',
        torch=torch.__version__, transformers=importlib.metadata.version('transformers'),
        logits_to_keep=1, batch_size=1, greedy=True, caps=CAPS,
        numerical_gate=GATE, cpu_tests_sha256=sha(OUT / 'cpu_tests.json'),
        mme_authorization_sha256=sha(OUT / 'mme_authorization.json'),
        created_at=datetime.now().astimezone().isoformat(), no_push=True)
    if recovery:
        atomic(OUT / 'prelaunch_recovery.json', recovery)
        base['prelaunch_recovery_sha256'] = sha(OUT / 'prelaunch_recovery.json')
    mme = dict(base, algorithm='FastV-K3 + llava-SVH', stage='authorized_complete_MME',
        questions=2374, numerical_engine_unmodified=True,
        parent_engine_sha256=sha(ROOT / 'vlm_research/fastv_svh/engine.py'),
        use_existing_parent_gpu_gate=True, existing_32882_predictions_unchanged=True)
    protocol = dict(base, algorithm='SparseVLM-192-prefix + llava-SVH',
        questions=35256, expected_counts=COUNTS, condition_count=1,
        official_commit=OFFICIAL_COMMIT, official_archive_commit=ARCHIVE_COMMIT,
        retained_tokens=192, topk_schedule=[300, 200, 110], prune_after_layers=[2, 6, 15],
        recycling_clustering='Unmodified official AST including 30 percent recycling and density noise',
        R_writer_layers=list(range(20)), answer_bypass_layers=[2, 3, 4, 5],
        answer_image_access=False, image_exit_before=20, persistent_original_visual_kv=0,
        score_queries='Unmodified official embedding-based text-rater selection, possibly Pminus1',
        score_only_mask='Actual current-stage SVH causal/source mask; Pminus1->Image remains zero',
        score_only_no_residual=True, positions='Original text absolute positions at every stage',
        merged_positions='Stage-local merged-token ordinal anchored at original image_start+ordinal; '
            'synthetic identity retained, not claimed original patch center; sequence-causal order unchanged',
        independent_sparsevlm_difference='Does not compact text RoPE, uses SVH source mask, exits at L20',
        recycling_randomness='Forked per-input-tensor-hash and layer seed; official density-noise code unchanged',
        expected_visual_by_layer=[576]*3+[309]*4+[204]*9+[113]*4+[0]*12,
        mean_visual_rows='Computed from actual per-layer physical shapes, not standalone206.5',
        smoke_images=6, gpu_gate_status='PENDING_AFTER_PREDECESSOR_RELEASE',
        optional_efficiency='Not scheduled: six-benchmark quality is priority')
    atomic(OUT / 'mme_protocol.json', mme)
    atomic(OUT / 'protocol.json', protocol)
    atomic(OUT / 'sample_manifest_reference.json', dict(path=str(MANIFEST), sha256=sha(MANIFEST),
        questions=35256, counts=COUNTS, partition_path=str(PARTITION), partition_sha256=sha(PARTITION),
        rank_counts=dict(Counter(assignment[s['key']] for s in samples)),
        image_clusters=len(clusters), circular_groups=len(circular), mme_image_pairs=len(pairs)))
    atomic(OUT / 'validation.json', dict(status='CPU_PASS_GPU_GATE_PENDING', cpu_tests_passed=True,
        actual_gpu_images=0, protocol_sha256=digest(protocol), no_threshold_change=True))
    atomic(OUT / 'completion.json', dict(status='QUEUED_AFTER_FASTV_AND_MME', completed=0, requested=35256,
        gpu_smoke='NOT_YET_RUN', paired_controls='RAW_HISTORY_MISSING', no_push=True))
    (OUT / 'report_zh.md').write_text('# SparseVLM-prefix + SVH\n\n'
        '状态：CPU回归通过，等待前一轮FastV+SVH及获准的MME补跑完成；本轮GPU Smoke尚未执行，正式预测0/35256。\n\n'
        '只有六张真实图片GPU门禁全部通过后才启动两卡完整评测。数值或语义失败停止留证，不调参、不放宽门槛。\n\n'
        'MME保留全部2374题，1张图片涉及2题的历史字节差异已获用户授权，仍不能声称与旧版本严格配对一致。\n\n'
        '历史SVH/Native逐题预测与Pixel绑定尚未恢复，最终不能凭历史总分伪造配对损害/纠正或CI。\n')
    print('FROZEN_CPU_ONLY', digest(protocol), 'MME', digest(mme), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refreeze-unlaunched', action='store_true')
    parser.add_argument('--recover-failed-unlaunched', action='store_true')
    parser.add_argument('--refreeze-failed-smoke', action='store_true')
    args = parser.parse_args()
    if args.refreeze_failed_smoke:
        refreeze_failed_smoke()
    else:
        main(args.refreeze_unlaunched, args.recover_failed_unlaunched)
