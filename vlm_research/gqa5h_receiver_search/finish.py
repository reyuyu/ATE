"""Finalize reviewable artifacts and their nonrecursive SHA index, before Git."""
import ast
from .common import *

def main():
    guard(reserve=420);v=read(P/'validation.json');assert v['passed'];assert read(P/'expanded3000/analysis_completion.json')['n']==3000
    stat=read(P/'expanded3000/cluster_statistics_validation.json');assert stat['passed'] and stat['final_paired_csv_sha256']==sha(P/'expanded3000/paired_comparisons.csv')
    from .expanded_statistics import cluster_p
    toy=[dict(image_sha256=im,correct=1,S_correct=0) for im in ['imageA','imageA','imageB']];pvalue=cluster_p(toy,'S');assert abs(pvalue-.5)<.02
    atomic_json(P/'cluster_test_validation.json',dict(passed=True,fixture='Three positive question deltas, two from the same image. Exact image-cluster sign-flip p=0.5; independently flipping three questions would incorrectly give 0.25.',Monte_Carlo_p=pvalue,expected_exact_cluster_p=.5))
    groups={}
    for row in csv.DictReader((P/'per_question.csv').open()):groups.setdefault((row['split'],row['job_id'],row['candidate_id']),[]).append(row)
    for row in csv.DictReader((P/'accuracy_summary.csv').open()):
        rr=groups[(row['split'],row['job_id'],row['candidate_id'])];assert len(rr)==int(row['n']) and sum(int(r['correct']) for r in rr)==int(row['correct']) and abs(np.mean([float(r['R_ratio']) for r in rr])-float(row['mean_R_ratio']))<1e-12
    required=['report_zh.md','experiment_timeline.csv','candidate_registry.csv','selection_rules.json','budget_accuracy_frontier.csv','dev_results.csv','tune_results.csv','final_holdout_results.csv','per_question.csv','token_selection_audit.csv','harm_cases.csv','recovered_cases.csv','validation.json','resource_audit.json','checkpoint_sha_index.csv','final_writer_manifest.json','winner_freeze.json','deployment_validation.json','statistics_completion.json','expanded3000/per_question.csv','expanded3000/token_manifest.json','expanded3000/gpu_release.json']
    assert all((P/f).is_file() and (P/f).stat().st_size>0 for f in required)
    for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    source={str(f.relative_to(ROOT)):sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))};atomic_json(P/'final_source_index.json',source)
    index=[]
    for file in sorted(P.rglob('*')):
        if file.is_file() and not (file.parent==P and file.name in ['artifact_sha_index.csv','completion.json','delivery.json']) and file.suffix not in ['.lock']:
            assert file.stat().st_size<90*1024**2;index.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=sha(file)))
    writecsv(P/'artifact_sha_index.csv',index)
    for row in index:assert sha(ROOT/row['path'])==row['sha256']
    elapsed=time.monotonic()-read(P/'clock.json')['start_monotonic'];assert elapsed<18000
    atomic_json(P/'completion.json',dict(status='COMPLETE',UTC=utc(),elapsed_seconds_conservative=elapsed,hard_deadline_seconds=18000,strict_under_5_hours=True,Winner=read(P/'winner_freeze.json')['candidate_id'],FINAL600_complete=True,expanded_frozen_Winner_3000_complete=True,unique_new_generations=v['unique_new_generations'],all_checkpoints_and_compact_SHA_verified=True,raw_archives_on_server_only=True,required_artifacts=required,artifact_count=len(index),artifact_index_sha256=sha(P/'artifact_sha_index.csv'),all_GPU_workers_exited=True,no_main_changes=True,followup_experiment_executed=False,note='Git commit/push and remote verification are recorded by the delivery step.'))
    print(json.dumps(read(P/'completion.json')),flush=True)

if __name__=='__main__':main()
