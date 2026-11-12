"""Local aggregate-only commits. Never push, stage raw predictions or model data."""
import json
import subprocess

from .common import ROOT,write,sha


def commit(label,folder):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='codex/qwen3vl-svh-quality-then-svhr-search-2x3090'
    patterns=['**/accuracy_summary.csv','**/subgroup_summary.csv','**/output_anomalies.csv',
        '**/ranking.json','**/completion.json','**/gpu_release.json','**/validation_selection.json',
        '**/final_selection.json','**/report_zh.md','**/protocol.json','**/candidate_grid.json',
        '**/sample_counts.json','**/numerical_summary.json','**/cpu_scoring_audit.json',
        '**/pope_8910_binding_audit.json','**/pope_8910_auxiliary.csv','**/timing_summary.csv',
        '**/physical_accuracy.csv','**/pareto_frontier.csv','**/history_evidence.json']
    patterns += ['**/all_stage_accuracy.csv','**/physical_status.json','**/validation.json']
    patterns += ['**/cpu_validation.json','**/figures/*.png','**/figures/*.pdf','**/full_physical_decision.json']
    chosen={path for pattern in patterns for path in folder.glob(pattern) if path.is_file()}
    if (ROOT/'overnight_report_zh.md').exists():
        chosen.add(ROOT/'overnight_report_zh.md')
    paths=sorted(str(p.relative_to(ROOT)) for p in chosen)
    if paths:
        subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True)
        dirty=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
        if dirty:
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','-c','gc.auto=0',
                'commit','-m',f'Archive {label}: local aggregates, raw predictions remain private'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(folder/'local_commit.json',dict(commit=head,branch=branch,label=label,pushed=False,
        files={str(p.relative_to(ROOT)):sha(p) for p in chosen},raw_predictions_committed=False))
    return head
