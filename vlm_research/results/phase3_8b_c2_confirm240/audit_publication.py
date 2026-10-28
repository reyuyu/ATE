"""Post-run verification only: never executes a model or changes scientific results.

Run with the fixed interpreter from this directory. This independently verifies
published CSV/JSON statistics, exact Pilot logits, provenance and file coverage.
The 1e-14 bound below applies only to CPU statistical arithmetic, not model
logits, Hook invariants, Pilot reproduction or scorer equality (all exact).
"""
import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from fixed_k_exit import atomic_json
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256


def read(path):
    return json.loads(path.read_text())


def exact_p(harm, rescue):
    n = harm+rescue
    if not n:
        return 1.
    with localcontext() as ctx:
        ctx.prec = 110
        term = Decimal(1)/(Decimal(2)**n)
        tail = term
        for k in range(min(harm, rescue)):
            term = term*Decimal(n-k)/Decimal(k+1)
            tail += term
        return float(min(Decimal(1), 2*tail))


def verify_stats(rows, summary):
    n = len(rows)
    assert summary['n'] == n
    arrays = {name: np.array([r[name] for r in rows], dtype=float)
              for name in ['full_correct', 'c2_correct', 'bias_correct', 'gain', 'delta_d']}
    indices = np.random.default_rng(382402026).integers(0, n, size=(10000, n))
    def close(a, b):
        assert np.max(np.abs(np.asarray(a)-np.asarray(b))) <= 1e-14
    def mean_ci(values):
        return np.percentile(np.asarray(values)[indices].mean(axis=1), [2.5, 97.5])
    close(summary['gain']['mean'], arrays['gain'].mean())
    close(summary['gain']['median'], np.median(arrays['gain']))
    close(summary['gain']['std'], np.std(arrays['gain'], ddof=1))
    close([summary['gain']['p5'], summary['gain']['p95']], np.percentile(arrays['gain'], [5,95]))
    assert summary['gain']['minimum'] == min(arrays['gain'])
    assert summary['gain']['maximum'] == max(arrays['gain'])
    assert summary['gain']['positive'] == int((arrays['gain']>0).sum())
    close(summary['ci95']['mean_gain'], mean_ci(arrays['gain']))
    close(summary['ci95']['c2_minus_full_accuracy'], mean_ci(arrays['c2_correct']-arrays['full_correct']))
    close(summary['ci95']['c2_minus_bias_accuracy'], mean_ci(arrays['c2_correct']-arrays['bias_correct']))
    close(summary['ci95']['mean_delta_d'], mean_ci(arrays['delta_d']))
    for label, name in [('C0','full_correct'), ('C2','c2_correct'), ('B1_offline','bias_correct')]:
        assert summary['accuracy'][label]['correct'] == int(arrays[name].sum())
        assert summary['accuracy'][label]['total'] == n
    for label, a, b in [('c2_vs_full','full_correct','c2_correct'),
                        ('c2_vs_bias','bias_correct','c2_correct'),
                        ('bias_vs_full','full_correct','bias_correct')]:
        table = [[0,0],[0,0]]
        for x,y in zip(arrays[a], arrays[b]):
            table[int(x)][int(y)] += 1
        assert summary[label]['table'] == table
        assert summary[label]['exact_mcnemar_two_sided_p'] == exact_p(table[1][0], table[0][1])
    for answer in ['gold','wrong','left','right']:
        changes = np.array([r['delta_'+answer] for r in rows])
        observed = summary['absolute_logp'][answer]
        close(observed['mean'], changes.mean())
        close(observed['median'], np.median(changes))
        close(observed['mean_ci95'], mean_ci(changes))
        assert observed['minimum'] == min(changes) and observed['maximum'] == max(changes)
        assert observed['positive'] == int((changes>0).sum())
    assert summary['both_candidates_absolute_logp_increase']['count'] == sum(r['delta_left']>0 and r['delta_right']>0 for r in rows)


def main():
    config, manifest = read(OUT/'experiment_config.json'), read(OUT/'manifest.json')
    assert digest(dict(config=config, samples=manifest['samples'])) == manifest['binding']
    assert config['plan']['conditions'] == ['C0','C2'] and config['plan']['layers'] == [2,3,4]
    assert config['plan']['offline_right_bias'] == 1.
    assert config['plan']['bootstrap_draws'] == 10000 and config['plan']['bootstrap_seed'] == 382402026
    assert all(sha256(ROOT/p)==h for p,h in config['sources'].items())
    assert all(sha256(ROOT/p)==h for p,h in config['inputs'].items())
    assert all(sha256(Path(config['model_path'])/p)==h for p,h in config['model_files_sha256'].items())
    original = read(OUT/'publication_validation.json')
    assert original['passed'] and all(sha256(OUT/p)==h for p,h in original['files'].items())
    gates = {name:read(OUT/name/'validation.json') for name in ['tests','smoke','run']}
    assert all(g['passed'] and g['binding']==manifest['binding'] for g in gates.values())
    assert gates['tests']['tests']==132 and all(gates['tests'][k]==0 for k in ['failures','errors','skips'])
    assert gates['smoke']['count']==2 and gates['run']['count']==240
    assert gates['smoke']['execution']==gates['run']['execution']
    hashes=[gates[stage][name] for stage in ['smoke','run'] for name in ['weights_before_sha256','weights_after_sha256']]
    pilot = ROOT/'results/phase3_8_sink_stage'
    hashes += [read(pilot/'run/validation.json')['weights_after_sha256']]
    assert len(set(hashes))==1
    with (OUT/'per_sample_scores.csv').open() as stream:
        csv_rows=list(csv.DictReader(stream))
    with (OUT/'offline_bias_baseline.csv').open() as stream:
        bias_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==480 and len(bias_rows)==240
    gpu={(r['question_id'],r['condition']):r for r in csv_rows}
    bias={r['question_id']:r for r in bias_rows}
    assert len(gpu)==480 and len(bias)==240
    records=[]; positions=[]; regressions=0
    for sample in manifest['samples']:
        q=sample['question_id']; folder=OUT/'run'/('sample_'+q)
        data=read(folder/'sample.json'); ck=read(folder/'checkpoint.json')
        assert ck['binding']==manifest['binding'] and ck['execution']==gates['run']['execution']
        assert ck['sample_sha256']==digest(sample) and ck['sample_json_sha256']==sha256(folder/'sample.json')
        assert gates['run']['samples'][q]==sha256(folder/'sample.json')
        assert data['raw_sha256']==sha256(folder/'logits.npz') and data['native_restoration_exact']
        for name in ['C0','C2']:
            stored=next(r for r in data['scores'] if r['condition']==name)
            csv_row=gpu[(q,name)]
            for key,value in stored.items():
                assert csv_row[key]==str(value)
        f,t=gpu[(q,'C0')],gpu[(q,'C2')]
        assert f['cohort']==t['cohort']==sample['cohort'] and f['gold']==sample['answer']
        d=float(f['right_logp'])-float(f['left_logp'])
        mb=(d+1)*(1 if sample['answer']=='right' else -1)
        assert float(bias[q]['margin'])==mb and (bias[q]['correct']=='True')==(mb>0)
        assert not any(k in bias[q] for k in ['gold_logp','right_logp','wrong_logp'])
        record=dict(question_id=q,gold=sample['answer'],cohort=sample['cohort'],
            full_correct=float(f['margin'])>0,c2_correct=float(t['margin'])>0,bias_correct=mb>0,
            gain=float(t['margin'])-float(f['margin']),
            delta_d=(float(t['right_logp'])-float(t['left_logp']))-d)
        for answer in ['gold','wrong','left','right']:
            record['delta_'+answer]=float(t[answer+'_logp'])-float(f[answer+'_logp'])
        records.append(record);positions.append(data['mapping']['answer_query'])
        if sample['cohort']=='Discovery16':
            with np.load(folder/'logits.npz') as new, np.load(pilot/'run'/('sample_'+q)/'raw_vectors_logits_attention.npz') as old:
                assert all(np.array_equal(new[k],old[k]) for k in new.files)
            assert all(r['bitwise_exact'] and r['logits_max_abs_delta']==0 and r['margin_delta']==0 for r in data['pilot_regression'].values())
            regressions+=1
    assert len({s['img_id'] for s in manifest['samples']})==240 and regressions==16
    for key,cohort in [('224','Confirmation224'),('16','Discovery16'),('240',None)]:
        selected=[r for r in records if cohort is None or r['cohort']==cohort]
        summary=read(OUT/('summary_'+key+'.json'))
        verify_stats(selected,summary)
        for answer in ['left','right']:
            verify_stats([r for r in selected if r['gold']==answer],summary['groups'][answer])
    def finite(value):
        if isinstance(value,float): assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values(): finite(v)
        elif isinstance(value,list):
            for v in value: finite(v)
    for path in OUT.rglob('*.json'):
        finite(read(path))
    assert not list(OUT.rglob('failure_*.json'))
    files={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='final_publication_audit.json'}
    result=dict(passed=True,scope='240 completed / 224 independent confirmation / 16 discovery',
        code_commit=config['code_commit'],binding=manifest['binding'],images=240,
        independently_rechecked_statistical_groups=9,paired_bootstrap_draws=10000,
        independent_mcnemar_reference='110-digit Decimal PMF recurrence',
        cpu_statistical_arithmetic_comparison_bound=1e-14,model_checks='bitwise exact; no tolerance relaxation',
        discovery_logit_regressions_bitwise_exact=regressions,full_model_state_sha256=hashes[0],
        smoke_formal_and_original_pilot_weights_equal=True,csv_rows=480,offline_bias_rows=240,
        dynamic_P_minus_1_range=[min(positions),max(positions)],plots_visually_reviewed=3,
        failed_attempts=0,original_summary_manifest_verified=True,
        live_console_self_hash=False,scientific_aggregations=1,
        publication_stage='closed artifacts; final commit is reported separately to avoid self-reference',
        files=files)
    atomic_json(OUT/'final_publication_audit.json',result)
    print('FINAL PUBLICATION AUDIT PASS',len(files),'closed files; 9 statistical groups; all exact regressions')


if __name__=='__main__':
    main()
