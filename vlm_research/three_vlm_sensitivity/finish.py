"""CPU-only post-process exit audit; never terminate or reset any GPU process."""
import os
import base64
import hashlib
import subprocess
import time

from .common import OUT, read, write, csv_write


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi'])
    decoded = raw.decode('utf-8', errors='replace')
    return dict(snapshot=decoded, snapshot_raw_base64=base64.b64encode(raw).decode('ascii'),
        snapshot_raw_sha256=hashlib.sha256(raw).hexdigest(),
        utf8_replacement_count=decoded.count('\ufffd'),
        snapshot_decoding='UTF-8 with replacement; raw bytes retained because nvidia-smi can truncate multibyte process names')


def main():
    executions = {n: read(OUT / f'{n}_run_execution.json') for n in ['qwen', 'internvl']}
    assert all(value['passed'] for value in executions.values())
    for name in executions:
        assert read(OUT / f'{name}_completion.json')['complete']
    listing = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    present = {int(row.split(',')[0]) for row in listing.splitlines()}
    own = [value['pid'] for value in executions.values()]
    assert not set(own) & present
    assert all(not os.path.exists(f'/proc/{pid}') for pid in own)
    uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    write(OUT / 'gpu1_release.json', dict(released=True, own_worker_pids=own, own_model_processes_absent=True,
        physical_gpu=1, uuid=uuid, remaining_compute_processes=listing, device_idle=not bool(listing),
        external_processes_terminated=False, timestamp=time.time(),
        **gpu_snapshot()))
    rows = []
    for name in executions:
        memory = read(OUT / f'{name}_run_memory.json')
        rows.extend(dict(model=name, **row) for row in memory['rows'])
    csv_write(OUT / 'gpu1_memory_samples.csv', rows)
    prior = read(OUT / 'execution_summary.json')
    write(OUT / 'sequential_initial_execution.json', prior)
    write(OUT / 'execution_summary.json', dict(status='gpu_completed_analysis_pending',
        initial_sequential_attempt=prior, internvl_native_head_addendum=True,
        final_executions=executions, final_smokes={n: read(OUT / f'{n}_gpu_smoke.json') for n in executions},
        formal_prefills={n: read(OUT / f'{n}_completion.json') for n in executions},
        physical_gpu=1, all_model_processes_exited=True, final_gpu_release='gpu1_release.json'))
    print('GPU1 OWN MODEL PROCESSES EXITED; all 192000 new single-layer prefills complete', flush=True)


if __name__ == '__main__':
    main()
