"""Independent CPU completeness and immutable-source audit after GPU exit."""
from collections import Counter, defaultdict

from prompt_common import *


def main():
    p=check_sources()
    old=json.loads((FIRST/'protocol.json').read_text())
    for path,state in old['model_files'].items():
        s=Path(path).stat()
        assert [s.st_size,s.st_mtime_ns]==state['stat'],path
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    index={r['key']:r for r in rows}
    assert len(index)==len(rows)
    assert not {r['image_sha256'] for r in rows if r['stage']=='Discovery'} & {r['image_sha256'] for r in rows if r['stage']=='Validation'}
    all_records=[]
    for shard in [0,1]:
        with readonly(OUT/f'worker{shard}.sqlite') as db, readonly(OUT/f'bindings{shard}.sqlite') as bd:
            saved={}
            for key,blob,expected in bd.execute('SELECT id,payload,sha256 FROM records'):
                raw=zlib.decompress(blob)
                assert hashlib.sha256(raw).hexdigest()==expected
                saved[key]=json.loads(raw)
            for key,blob,expected in db.execute('SELECT id,payload,sha256 FROM records'):
                raw=zlib.decompress(blob)
                assert hashlib.sha256(raw).hexdigest()==expected
                r=json.loads(raw)
                source=index[r['key']]
                assert source['shard']==shard and source['stage']==r['stage']
                assert source['gold']==r['gold'] and source['image_sha256']==r['image_sha256']
                assert r['gpu_uuid']==UUIDS[shard] and r['protocol_sha256']==sha(OUT/'protocol.json')
                assert r['generated_tokens']==len(r['token_ids'])<=source['max_new_tokens']
                assert r['input_binding_sha256']==digest(saved[r['key']+'|'+r['prompt']])
                assert r['structural']['status']=='PASS'
                all_records.append(r)
    grouped=defaultdict(list)
    for r in all_records:
        grouped[r['stage'],r['key'],r['prompt']].append(r)
    for group in grouped.values():
        assert {r['condition'] for r in group}=={'Native','SVH'}
        assert len({r['input_binding_sha256'] for r in group})==1
    selection=json.loads((OUT/'selection_frozen.json').read_text())['selected']
    expected=0
    for row in rows:
        prompts=conditions(row['benchmark'])[1:] if row['stage']=='Discovery' else [selection[row['benchmark']]]
        for prompt in prompts:
            if prompt!='P0':
                assert (row['stage'],row['key'],prompt) in grouped
                expected+=2
    assert expected==len(all_records)==4568
    reuse=json.loads((OUT/'p0_fresh_sequence_regression.json').read_text())
    assert reuse['all_token_ids_identical'] and reuse['all_input_bindings_identical']
    release=json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['own_workers_terminated'] and not release['own_gpu_processes_remain']
    write(OUT/'validation.json',dict(status='PASS_PROMPT_EXPERIMENT_INTEGRITY',fresh_quality_records=len(all_records),
        p0_reused_quality_records=len(rows)*2, measured_performance_records=192,
        complete_native_svh_input_pairs=len(grouped),source_hashes_unchanged=True,model_file_stats_match_historical=True,
        p0_fresh_sequences_and_bindings_match=True,global_image_disjoint=True,gpu_release_pass=True,
        inherited_strict_dense_physical_status='FAIL_RETAINED',not_author_official_reproduction=True,
        counts=dict(Counter(r['stage']+'|'+r['benchmark']+'|'+r['prompt']+'|'+r['condition'] for r in all_records))))
    write(OUT/'artifact_sha256.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact_sha256.json'})
    print('FINAL_INTEGRITY_PASS',expected,'new quality records',flush=True)


if __name__=='__main__':
    main()
