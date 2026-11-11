"""Preserve the pre-GPU superseded protocol and freeze the user's revision."""
import shutil
from collections import Counter,defaultdict
from common import *


def main():
    old=json.loads((OUT/'protocol.json').read_text())
    assert old['main_view']=='EXACT_FROZEN_SVH_PREFIX_DECODE_ONLY_REACCESS'
    assert not list(OUT.glob('smoke*.json')) and not list(OUT.glob('worker*.sqlite'))
    shutil.copyfile(OUT/'protocol.json',OUT/'superseded_decode_only_protocol_cpu_only.json')
    shutil.copyfile(OUT/'candidate_grid.json',OUT/'superseded_candidate_grid_cpu_only.json')
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    existing={r['image_sha256']:r['split'] for r in rows}
    # Final protection cohorts: preserve complete MME pairs, no overlap with search images.
    extra=[]
    planned=inventory()
    for task in ['POPE','MME']:
        pool=[r for r in planned if r['benchmark']==task and r['image_sha256'] not in existing]
        labels=['random','popular','adversarial'] if task=='POPE' else sorted({r['task'] for r in pool})
        for label in labels:
            group=[r for r in pool if r['strategy' if task=='POPE' else 'task']==label]
            units=defaultdict(list)
            for r in group:
                units[r['key'] if task=='POPE' else r['pair_id']].append(r)
            count=32 if task=='POPE' else 8
            for key in sorted(units,key=lambda k:digest([SEED,'PROTECTION',task,label,k]))[:count]:
                for r in units[key]:
                    cluster='I:'+r['image_sha256']
                    shard=int(digest([SEED,'SHARD',cluster])[:8],16)%2
                    extra.append(dict(r,analysis_task=task,split='Held-out',cluster=cluster,shard=shard,prompt_protocol='FROZEN_PLAN_B_ORIGINAL'))
    hist=historical({r['key'] for r in extra})
    store=Store(OUT/'historical.sqlite')
    for r in extra:
        for cond in ['Native','SVH']:
            rec=hist[r['key'],cond]
            assert rec['image_sha256']==r['image_sha256'] and sha(r['image_path'])==r['image_sha256']
            if rec.get('status','COMPLETE')!='COMPLETE':
                r['historical_blocked_status']=rec['status']
            store.put(r['key']+'|'+cond,rec)
    store.conn.close()
    rows+=extra
    placement={}
    for r in rows:
        r['shard']=placement.setdefault(r['image_sha256'],r['shard'])
    write(OUT/'sample_manifest.json',rows)
    grid={'K0':[],'R14':list(range(11,25)),'R11':list(range(11,22)),
          'R8':list(range(14,22)),'R6':list(range(16,22)),'R4':list(range(18,22)),
          'R2':[20,21],'R1':[21]}
    for lo in [8,10,12,14]:
        grid[f'W{lo}_{lo+13}']=list(range(lo,lo+14))
    for lo,hi in [(17,24),(19,24),(21,24),(23,24)]:
        grid[f'Deep{hi-lo+1}']=list(range(lo,hi+1))
    grid.update(Prior4=[21,23,24,26],Prior2=[24,26],Prior1=[26])
    write(OUT/'candidate_grid.json',grid)
    old.update(revision_request='USER_UNIFIED_P_MINUS_1_AND_REAL_DECODE_SAME_SET',
        main_view='UNIFIED_P1_AND_DECODE_SAME_LAYER_SET',main_candidates_max=40,
        auxiliary_view='P1_ONLY_AND_DECODE_ONLY_FOR_FINAL_1_TO_2_SETS_NOT_RANKED',
        auxiliary_prefill_layers=None,superseded_protocol='superseded_decode_only_protocol_cpu_only.json',
        no_gpu_executed_under_old_protocol=True,manifest_sha256=sha(OUT/'sample_manifest.json'),
        sources={**{str(p):sha(p) for p in ROOT.glob('*.py')},
                 **{k:sha(k) for k in old['sources'] if not k.startswith(str(ROOT))}},
        main_candidate_definitions=grid,selection_constraints_unchanged=True,
        additional_protection_counts=dict(Counter(r['benchmark'] for r in extra)))
    write(OUT/'protocol.json',old)
    print(json.dumps(dict(grid=grid,counts=dict(Counter(r['analysis_task'] for r in rows)),
        status='REVISED_BEFORE_ANY_GPU_EXECUTION'),indent=2))


if __name__=='__main__':
    main()
