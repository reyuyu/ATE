"""Freeze after the user's explicit approval of physical GPU0 and GPU1."""
import subprocess
import sys

from .common import *
from ..fastp2_bypass_writer.common import hardware


def main():
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    devices = [dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'),
               dict(index=1, uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417')]
    resource = hardware()
    assert all(d['uuid'] not in resource['processes'] for d in devices)
    result = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_recovery.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    atomic(OUT / 'gpu_authorization.json', dict(status='USER_AUTHORIZED', at=now(), devices=devices,
        request='Assistant explicitly requested physical GPU0 and GPU1 for this experiment.',
        user_reply='可以使用', scope='This fixed E1-E5 experiment only; no historical authorization assumed.'))
    protocol = read(OUT / 'protocol.json')
    protocol.update(gpu_authorization='USER_AUTHORIZED', devices=devices,
                    candidate_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic(OUT / 'protocol.json', protocol)
    atomic(OUT / 'execution_freeze.json', dict(at=now(), sources=sources(), devices=devices,
        weights=read(OUT / 'source_manifest.json')['weights'],
        code_commit=protocol['candidate_source_commit'], cpu_tests_sha256=sha(OUT / 'cpu_tests.log'),
        **{name+'_sha256': sha(OUT / (name+'.json')) for name in
           ('sample_manifest', 'partition', 'protocol', 'gpu_authorization')}))
    atomic(OUT / 'authorized_preflight.json', dict(at=now(), **resource))
    print('FROZEN: GPU0/GPU1, unchanged500 questions, five recovery conditions; CPU19 PASS')


if __name__ == '__main__':
    main()
