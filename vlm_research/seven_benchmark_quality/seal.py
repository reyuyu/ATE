"""Finalize CPU prelaunch code/official-scorer binding; never after GPU launch."""
import shutil
from .common import *


def main():
    assert not (OUT / 'running.json').exists()
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['artifacts'].items():
        assert sha(OUT / name) == expected, name
    archive = OUT / 'attempts/initial_cpu_protocol.json'
    assert not archive.exists(), 'Do not replace a prior seal'
    atomic(archive, protocol)
    official = OUT / 'official'
    official.mkdir(exist_ok=True)
    resource = Path('/data/lk/ATE-resources/experiments/seven_benchmark_native_physical/official')
    for name in ('matching_util_api.json', 'multiple_choice_api.json'):
        shutil.copy2(resource / name, official / name)
        protocol['artifacts']['official/' + name] = sha(official / name)
    protocol.update(sources=sources(), sealed_at=now(),
        mmbench_status='Official VLMEvalKit exact_matching plus circular all-rotation hit; no paid judge',
        mmbench_official_commit='3c192143007e4eb3f5c1a36558a4f8c1451980cb',
        cpu_protocol_before_final_code=digest(read(archive)))
    assert official_mmbench_matcher()('B', dict(A='red', B='blue')) == 'B'
    atomic(OUT / 'protocol.json', protocol)
    print('FINAL_SEAL', digest(protocol), flush=True)


if __name__ == '__main__':
    main()
