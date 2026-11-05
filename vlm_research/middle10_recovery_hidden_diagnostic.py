"""After timing exits, audit E4 P-1 propagation on the three frozen Smoke inputs.

This reuses the unchanged E3/E4 algorithms; it never selects a candidate or scores
additional questions. No concurrent GPU workload is permitted while it runs.
"""
import gc
import os

import torch

from .middle10_recovery.common import OUT, read, atomic, write_csv, sha, now
from .middle10_recovery.engine import generate, prepare
from .middle10_recovery.run import verify_freeze
from .fastp2_bypass_writer.common import hardware
from .p1_layerwise_accuracy.engine import Adapter


def main():
    freeze = verify_freeze()
    assert read(OUT / 'driver_exit.json')['returncodes'] == [0, 0]
    assert all(read(OUT / f'worker{r}/completion.json')['status'] == 'COMPLETE' for r in range(2))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    device = next(d for d in freeze['devices'] if d['uuid'] == visible)
    resource = hardware()
    assert device['uuid'] not in resource['processes'], 'Diagnostic GPU must be idle'
    samples, partition = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
    adapter = None
    rows = []
    try:
        adapter = Adapter('llava')
        for key in partition['smoke']:
            layout, _, _ = prepare(adapter, samples[key])
            e3 = generate(adapter, layout, 'E3', 3, fixed=True, capture='all', audit=True)
            e4 = generate(adapter, layout, 'E4', 3, fixed=True, capture='all', audit=True)
            assert len(e3[3]) == len(e4[3]) == 32
            for layer, (before, after) in enumerate(zip(e3[3], e4[3])):
                assert torch.isfinite(before).all() and torch.isfinite(after).all()
                x, y = before[0, -1].float(), after[0, -1].float()
                rows.append(dict(key=key, layer=layer, p1_input_position=layout.length-1,
                    p1_hidden_norm_e3=float(x.norm()), p1_hidden_norm_e4=float(y.norm()),
                    p1_change_norm=float((y-x).norm()),
                    p1_cosine_distance=float(1-torch.nn.functional.cosine_similarity(x, y, dim=0)),
                    non_p1_prefix_exact=torch.equal(before[:, :-1], after[:, :-1]),
                    non_p1_prefix_max_abs=float((after[:, :-1]-before[:, :-1]).abs().max()),
                    first_token_changed=e3[0]['token_ids'][0] != e4[0]['token_ids'][0]))
            assert all(r['p1_change_norm']==0 for r in rows if r['key']==key and r['layer']<2)
            assert any(r['p1_change_norm']>0 for r in rows if r['key']==key and r['layer'] in (2,3,4,5))
            del e3, e4
            adapter.verify()
        write_csv(OUT / 'e4_p1_hidden_propagation.csv', rows)
        atomic(OUT / 'hidden_diagnostic.json', dict(status='PASS', at=now(), pid=os.getpid(),
            samples=partition['smoke'], layers_per_sample=32, scope='Same three frozen Smoke inputs; E3 vs E4, no added conditions or quality scoring.',
            source_sha256=sha(__file__), execution_sources=freeze['sources'],
            prefix_all_layers_exact=all(r['non_p1_prefix_exact'] for r in rows),
            prefix_max_abs=max(r['non_p1_prefix_max_abs'] for r in rows)))
        print('PASS: E4 first changes P-1 atL2; captured propagation through all32 layers on3 frozen inputs', flush=True)
    finally:
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        atomic(OUT / 'hidden_diagnostic_release.json', dict(at=now(), pid=os.getpid(),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            process_exit_releases_remaining=True))


if __name__ == '__main__':
    main()
