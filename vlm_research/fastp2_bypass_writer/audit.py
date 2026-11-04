"""Independent integer-shape FLOPs, cohort and checkpoint audit, outside inference."""
import collections
from pathlib import Path

from .common import OUT, ROOT, atomic, read, sha, digest, verify, key
from .engine import CONDITIONS


def prefill_flops(rows):
    return sum(r['estimated_flops'] for r in rows if r['phase']=='prefill')


def theoretical_savings(n,images,d=4096,layers=4):
    all_attention=layers*(8*n*d*d+4*n*n*d)
    ia=layers*(4*(images+1)*d*d+4*(images+1)*n*d)
    return dict(A1=all_attention,A2=ia,B1=0)


def main():
    p=verify()
    samples=read(OUT/'sample_manifest.json')
    specs=read(OUT/'input_specs.json')
    rows=[]
    missing=[]
    counts=collections.Counter()
    for k,s in samples.items():
        rank=p['assignment'][s['image_sha256']]
        path=OUT/f'shard{rank}/checkpoints'/(k+'.json')
        if not path.exists():
            missing.append(k)
            continue
        value=read(path)
        v=value['payload']
        assert digest(v)==value['sha256'] and v['protocol_sha256']==digest(p)
        assert v['sample_sha256']==digest(s) and v['input_spec_sha256']==digest(specs[k])
        n=v['layout']['length'];I=len(v['layout']['image']);R=len(v['layout']['relay'])
        expected=theoretical_savings(n,I)
        expected.update(B3=20*4*4096*(R-specs[k]['selection']['k']+1)*I,
                        **{'B2-'+str(seed):0 for seed in (20261103,20261107,20261111)})
        baseline=prefill_flops(v['conditions']['A0']['flops'])
        assert set(v['conditions'])==set(CONDITIONS)
        for c in CONDITIONS:
            entry=v['conditions'][c]
            got=baseline-prefill_flops(entry['flops'])
            if c!='A0': assert got==expected[c],(k,c,got,expected[c])
            mlp=sum(r['estimated_flops'] for r in entry['flops'] if r['phase']=='prefill' and r['component']=='mlp')
            baseline_mlp=sum(r['estimated_flops'] for r in v['conditions']['A0']['flops'] if r['phase']=='prefill' and r['component']=='mlp')
            assert mlp==baseline_mlp
            counts[s['dataset'],c]+=1
            assert len(entry['measurements'])==(20 if s['dataset']=='pilot60' else 0)
        rows.append(dict(sample=k,prefill_A1_saved=expected['A1'],prefill_A2_saved=expected['A2'],
                         prefill_B3_saved=expected['B3'],prefill_baseline=baseline,R=R,
                         semantic_k=specs[k]['selection']['k'],actual_shape_equal_closed_form=True))
    fixed=collections.Counter()
    for length in (16,64):
        for path in OUT.glob(f'shard*/fixed{length}/*.json'):
            blob=read(path);v=blob['payload']
            assert digest(v)==blob['sha256'] and v['protocol_sha256']==digest(p)
            k='pilot60/'+v['sample']
            s=samples[k]
            assert path.parent.parent.name=='shard'+str(p['assignment'][s['image_sha256']])
            for c in CONDITIONS:
                cs=v['conditions'][c]
                assert len(cs['measurements'])==20 and len(cs['quality']['token_ids'])==length
                assert all(m['token_ids']==cs['quality']['token_ids'] for m in cs['measurements'])
            fixed[length]+=1
    all_hashes={str(path.relative_to(OUT)):sha(path) for path in OUT.rglob('*.json')
                if path.name not in ('independent_audit.json','artifact_hashes.json')}
    overlap={}
    for left,right in (('pilot60','gqa240'),('pilot60','textvqa300'),('gqa240','textvqa300')):
        a={s['image_sha256'] for s in samples.values() if s['dataset']==left}
        b={s['image_sha256'] for s in samples.values() if s['dataset']==right}
        overlap[left+'__'+right]=len(a&b)
    atomic(OUT/'independent_audit.json',dict(status='PASS' if not missing and fixed[16]==fixed[64]==60 else 'PARTIAL',
        missing=missing,condition_counts={str(k):v for k,v in counts.items()},fixed_inputs=dict(fixed),
        shape_census_equals_independent_integer_formula=rows,all_MLP_flops_preserved=True,
        image_cluster_overlap=overlap,source_verified=True,checkpoint_sha256=all_hashes))
    print('INDEPENDENT_AUDIT',len(rows),'/600','fixed',dict(fixed),'overlap',overlap,flush=True)


if __name__=='__main__':main()
