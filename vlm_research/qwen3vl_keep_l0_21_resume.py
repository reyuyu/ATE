"""Explicit restart helper; never alters the frozen worker or scientific records.

Run with the pinned Qwen Python only after an interrupted collection. This is
not a GPU monitor and does not wait for resources. Existing metadata are moved
to an attempt directory, allowing the original immutable worker to create fresh
Smoke/barrier records while retaining every validated question checkpoint.
"""
import os
from pathlib import Path
import sys
import time

from .qwen3vl_keep_l0_21.common import (
    OUT, ROOT, read, verify, assert_idle, digest, key, sha256, seal)


def archive_metadata(out, stamp):
    names = ['launch.json', 'gpu_release_audit.json']
    names += [f'shard{i}/{name}.json' for i in (0, 1)
        for name in ('preflight', 'execution', 'smoke', 'complete', 'release')]
    paths = [out/name for name in names]
    paths += list(out.glob('shard*/failure*.json')) + list(out.glob('launcher_failure*.json'))
    for path in paths:
        if path.exists():
            destination = out/'resume_attempts'/stamp/path.relative_to(out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists()
            path.rename(destination)


def main():
    p = verify()
    assert sys.executable == read(OUT/'environment.json')['executable'], 'Use the pinned Qwen Python'
    if (OUT/'completion.json').exists():
        assert read(OUT/'completion.json')['status'] == 'COMPLETE'
        print('Already COMPLETE; no model loading or generation repeated.')
        return
    if (OUT/'launch.json').exists():
        launch = read(OUT/'launch.json')
        for pid in [launch['pid'], *launch['worker_pids']]:
            command = Path(f'/proc/{pid}/cmdline')
            if command.exists():
                assert b'vlm_research.qwen3vl_keep_l0_21.run' not in command.read_bytes(), 'Experiment is still running'
    specs = read(OUT/'input_bindings.json')
    samples = {key(s): s for s in read(OUT/'sample_manifest.json')['samples']}
    shards = read(OUT/'shards.json')['keys']
    counts = []
    for shard in (0, 1):
        paths = list((OUT/f'shard{shard}/checkpoints').glob('*.json'))
        for path in paths:
            r = read(path)
            identifier = key(r)
            assert identifier in shards[shard] and path.stem == identifier
            assert r['binding'] == digest(p) and r['sample_sha256'] == digest(samples[identifier])
            assert r['input_binding'] == specs[identifier]['binding'] and r['generation_sha256'] == digest(r['generation'])
        counts.append(len(paths))
    if sum(counts) == 3000:
        raise RuntimeError('All generation checkpoints exist; do not repeat GPU work. Resume CPU analysis/delivery only.')
    for gpu in p['gpus']:
        assert_idle(gpu['uuid'])
    stamp = str(time.time_ns())
    archive_metadata(OUT, stamp)
    seal(OUT/'resume_attempts'/stamp/'request.json', dict(existing_checkpoints=counts,
        frozen_protocol_binding=digest(p), helper_sha256=sha256(__file__), no_checkpoints_modified=True))
    from .qwen3vl_keep_l0_21.run import launch
    launch()


if __name__ == '__main__':
    main()
