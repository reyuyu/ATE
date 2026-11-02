"""Independent CPU verification of persisted results, no additional GPU forwards."""
import time
import subprocess
import numpy as np
import torch

from benchmarks.dataset_registry import sha256
from temporal_visual_gap.common import OUT, CONDITIONS, read, write, verify, gate
from temporal_visual_gap.engine import compare
from temporal_visual_gap.run import checkpoint
from vlm_common import write_json


def release():
    done=read(OUT/'completion.json')
    table=subprocess.check_output(['nvidia-smi','-i','1',
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
    apps=subprocess.check_output(['nvidia-smi','-i','1',
        '--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not any(str(done['pid'])==line.split(',')[1].strip() for line in apps.splitlines())
    write(OUT/'gpu_release.json',dict(released=True,worker_pid=done['pid'],table=table,
        compute_processes=apps,physical_gpu=1,checked_epoch=time.time(),no_new_GPU_experiments=True))


def main():
    tick=time.time(); p=verify(); m=read(OUT/'sample_manifest.json')
    assert read(OUT/'completion.json')['passed'] and read(OUT/'validation.json')['passed']
    assert read(OUT/'analysis_cpu_validation.json')['passed']
    tokens=0; branch_rows=0; gate_rows=0
    for s in m['COCO']+m['GQA']:
        path=OUT/f"checkpoints/A/{s['question_id'].replace(':','_')}.json"
        r=checkpoint(path,p,s,'A','Native-ImageOFF')
        assert r['weights_inputs_unchanged'] and r['hooks_clean']
        keys=[i for i,t in enumerate(s['input_mapping']['prompt_ids']) if t==32000]
        assert len(keys)==576 and r['image_keys']==keys
        n=len(s['input_mapping']['prompt_ids'])
        with np.load(r['raw_file']) as raw:
            fl,off=raw['full_logits'],raw['imageoff_logits']
        ids=r['generation']['token_ids']
        assert len(fl)==len(off)==len(ids)==len(r['token_metrics'])==len(r['probability_audits'])
        assert np.isfinite(fl).all() and np.isfinite(off).all()
        for t,(f,o,row,audit) in enumerate(zip(fl,off,r['token_metrics'],r['probability_audits'])):
            fresh=compare(torch.from_numpy(f),torch.from_numpy(o))
            assert all(fresh[k]==row[k] for k in fresh), 'Persisted metric differs from raw logits'
            assert row['t']==t and row['full_token']==ids[t] and row['absolute_query']==n+t-1
            assert row['consumed_token']==(ids[t-1] if t else None)
            assert row['is_eos']==(ids[t]==2) and row['cache_after']==n+t
            assert not t or row['cache_before']==n+t-1
            for branch in ['Full','ImageOFF']:
                rr=audit[branch]; assert [x['layer'] for x in rr]==list(range(32))
                for x in rr:
                    assert x['blocked']==(branch=='ImageOFF')
                    assert x['image_probability_max']==0. if branch=='ImageOFF' else x['image_probability_max']>0.
                branch_rows+=32
            tokens+=1
    for s in m['COCO']:
        for c in CONDITIONS:
            path=OUT/f"checkpoints/B/{s['question_id'].replace(':','_')}/{c}.json"
            r=checkpoint(path,p,s,'B',c); g=r['generation']
            assert r['weights_inputs_unchanged'] and r['hooks_clean'] and r['first_logits_exact']
            assert len(r['gates'])==len(g['token_ids'])-1
            for t,row in enumerate(r['gates'],start=1):
                assert row['t']==t and row['on']==gate(c,s['image_sha256'],t)
                if not r['historical_binding']:
                    assert [x['layer'] for x in row['probabilities']]==list(range(32))
                    assert row['consumed_token']==g['token_ids'][t-1] and row['predicted_token']==g['token_ids'][t]
                    for x in row['probabilities']:
                        assert x['blocked']==(not row['on'])
                        assert x['image_probability_max']>0. if row['on'] else x['image_probability_max']==0.
                gate_rows+=1
    files={str(q.relative_to(OUT)):sha256(q) for q in sorted(OUT.rglob('*'))
           if q.is_file() and q.suffix in ['.json','.csv','.pdf','.png','.md','.log']
           and q.name not in ['delivery_validation.json','artifact_hashes.json']}
    write_json(OUT/'artifact_hashes.json',files)
    write_json(OUT/'delivery_validation.json',dict(passed=True,source_unchanged=True,
        raw_full_vocabulary_metrics_exactly_recomputed=tokens, all32_single_step_probability_rows=branch_rows,
        temporal_gate_rows_verified=gate_rows,artifacts=len(files),seconds=time.time()-tick,
        code_sha256=sha256(__file__)))
    print('Delivery verification passed:',tokens,'native positions;',gate_rows,'temporal gates;',len(files),'artifacts',flush=True)


if __name__=='__main__': main()
