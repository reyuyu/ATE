"""Archive pre-execution CPU snapshots; never rewrite a GPU-running snapshot."""
import shutil
import subprocess
from common import *


def main():
    assert not list(OUT.glob('worker*.sqlite')) and not list(OUT.glob('smoke*.json'))
    p=json.loads((OUT/'protocol.json').read_text())
    target=OUT/'protocol_unified_pre_gpu_cpu_snapshot.json'
    assert not target.exists()
    shutil.copyfile(OUT/'protocol.json',target)
    p['sources']={**{k:sha(k) for k in p['sources'] if not k.startswith(str(ROOT))},
                  **{str(path):sha(path) for path in ROOT.glob('*.py')}}
    p['execution_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    p['pre_execution_cpu_tests']='8 tests, including per-layer capacity, independent cache and disjoint manifest'
    write(OUT/'protocol.json',p)
    check()
    print('EXECUTION_FROZEN',p['execution_source_commit'],sha(OUT/'protocol.json'))


if __name__=='__main__':
    main()
