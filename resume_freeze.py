"""Archive a stopped attempt before rebinding an implementation-only repair."""
import shutil
import subprocess
from common import *


def main():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(uuid in processes for uuid in UUIDS)
    folder=OUT/'attempts'/'0001_writer_duplicate_field'
    assert not folder.exists()
    folder.mkdir(parents=True)
    for path in [OUT/'protocol.json',OUT/'pipeline_failure.json',OUT/'STOP',OUT/'gpu_release_audit.json']+list(OUT.glob('failure*.json')):
        if path.exists():
            shutil.copyfile(path,folder/path.name)
            if path.name!='protocol.json':
                path.unlink()
    counts={}
    for i in [0,1]:
        db=readonly(OUT/f'worker{i}.sqlite')
        counts[i]=db.execute('SELECT count(*) FROM records').fetchone()[0]
        db.close()
    assert sum(counts.values())==0,'Rebind only this pre-record serializer failure; never rewrite completed records'
    p=json.loads((OUT/'protocol.json').read_text())
    p['sources']={**{k:sha(k) for k in p['sources'] if not k.startswith(str(ROOT))},
                  **{str(path):sha(path) for path in ROOT.glob('*.py')}}
    p['serializer_repair']='Avoid duplicate prefill_layers keyword when merging model result; algorithm and input unchanged'
    p['prior_attempt_records']=counts
    p['execution_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(OUT/'protocol.json',p)
    check()
    print('SERIALIZER_FIX_FROZEN',counts,sha(OUT/'protocol.json'))


if __name__=='__main__':
    main()
