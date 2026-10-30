"""Delivery integrity or explicitly partial resource status; never fake effects."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from benchmarks.dataset_registry import freeze_json,sha256
from .protocol import ROOT,OUT,PYTHON,GPUS
from .run import verify,gate,verify_targets
from .engine import metrics


def gpu_snapshot():
    return dict(observed_at=datetime.now().astimezone().isoformat(),
        gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
        compute_process_csv=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True))


def readme():
    text=f'''# Late-Layer Attention Source Decomposition

See report.md for the actual current completion status; never infer completion
from the presence of runnable code. All source, sample and protocol hashes bind
the unchanged collector commit `{verify()['code_commit']}`. CPU-only analysis is
separate; it cannot alter frozen worker sources or Observation2 artifacts.

## Resume

From `{ROOT}` on the same local resources, only when allfour GPUs are idle:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY={PYTHON}
$PY -m unittest -v test_late_attention_sources test_token_role_decode test_late_source_statistics
$PY -u -m paper_rq1_late_attention_sources.launch
$PY -u -m paper_rq1_late_attention_sources.analyze
$PY -u -m paper_rq1_late_attention_sources.finalize complete
```

Do not rerun data selection/CPU freeze to replace the already frozen samples.
Valid completed smoke shards/checkpoints are reused. Allfour Smoke completions
are required before Full; all Full captions are required before central target
selection; the exact target SHA is required before source interventions.
An occupied GPU stops that stage without terminating any other process. See
resource_event.json for the initial refused GPU0 process, and original logs.

## Artifacts

- sample_manifest.json, data_manifest.json: independent200+1 smoke image binding.
- cpu_validation.json, execution_manifest.json: tested collector/protocol/model.
- smoke/, full/, ablation/: actual per-GPU environments/logs/checkpoints only.
- selected_token_manifest.json: created only after native Full200 completes.
- source_ablation_per_token.csv: all same-prefix Full and source interventions.
- native_source_contributions.csv: actual native mass/projected norm/cosines.
- source_layer_summary.csv, native_source_summary.csv: signed/absolute/tail/flip
  summaries, with image-cluster confidence intervals.
- planned_primary_contrasts.csv: frozen41-comparison family; role, source, deep
  continuity and recent/older comparisons. Full controls/rank audit retained.
- visual_function_paired_comparison.csv, deep_layer_peak_analysis.csv,
  recent_vs_older_history.csv: user-requested primary breakdowns.
- numerical_validation.json: complete only after all formal shards pass.
- figures/: five PNG/PDF real-data figures, created only after complete gates.
- partial/: earlier resource-blocked snapshots, never formal result evidence.

Absent artifacts mean not yet executed, not zero effects. Raw full-vocabulary
FP32 NPZ files stay at `/data/lk/ATE-resources/late_attention_sources/raw/`, with
SHA256 in checkpoints and completed raw_logits_manifest.csv. Model weights and
raw images are not Git blobs. Single-step LogP loss is not answer accuracy,
source losses are not additive, and no actual speedup is measured.
'''
    (OUT/'README.md').write_text(text)


def partial():
    config=verify()
    if (OUT/'analysis_completion.json').exists():raise ValueError('Do not downgrade an existing completed analysis')
    stages={}
    validations=[]
    for stage in ['smoke','full','ablation']:
        stages[stage]=[]
        for rank in range(4):
            path=OUT/f'{stage}/shard{rank}/completion.json'
            if path.exists():
                value=json.loads(path.read_text())
                if value['rank']!=rank:raise ValueError('Shard completion rank mismatch')
                stages[stage].append(value)
                if stage=='smoke':
                    cp=next((path.parent/'checkpoints').glob('*.json'));r=json.loads(cp.read_text())
                    if not r['passed'] or sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Invalid completed smoke')
                    for check in r['checks']:validations.extend(check.get('audits',[]))
    if len(stages['ablation'])==4:raise ValueError('All formal shards complete: run real analysis, not partial report')
    snapshot=gpu_snapshot()
    if not any(line.startswith(uuid) for line in snapshot['compute_process_csv'].splitlines() for uuid in GPUS):
        raise ValueError('Allfour are now idle: resume frozen collection instead of claiming resource blockage')
    stamp=str(time.time_ns());folder=OUT/'partial'/stamp;folder.mkdir(parents=True)
    status=dict(status='partial_resource_blocked',source_commit=config['code_commit'],snapshot=snapshot,
        images_frozen=200,smoke_unique_images=1,smoke_complete_physical_gpus=[r['rank'] for r in stages['smoke']],
        full_completed_images=sum(r['completed'] for r in stages['full']),
        formal_completed_images=sum(r['completed'] for r in stages['ablation']),
        targets_frozen=(OUT/'selected_token_manifest.json').exists(),stages=stages,
        actual_smoke_decomposition_checks=len(validations),
        max_smoke_error_to_bound=max(v for r in validations for k,v in r.items() if k.endswith('_error_to_bound')) if validations else None,
        formal_effect_estimates_available=False,formal_figures_generated=False,
        other_processes_untouched=True,reason='Four-physical-GPU requirement and allfour smoke gate are retained. GPU0 occupied by another user, so Full and formal ablations not launched.')
    freeze_json(folder/'status.json',status)
    readme()
    lines=['# Late-Layer Attention Source Decomposition','', '**状态：partial，尚无正式实验结果。**','',
        f"冻结200张新图片；采集CPU测试20项通过，独立统计测试见statistics_tests.txt。冻结采集Commit：`{config['code_commit']}`。",
        f"已完成真实GPU Smoke的物理卡：{status['smoke_complete_physical_gpus']}；这是一张独立Smoke图在各卡的重复验证，不是正式200图结果。",
        f"Native Full完成图片数：{status['full_completed_images']}；正式消融完成图片数：{status['formal_completed_images']}；目标词位置尚未选择/冻结。",
        '', '## 阻塞原因','',
        'GPU0被其他用户的计算任务占用，初次预检拒绝加载。没有终止、修改或抢占该任务。GPU1–3的Smoke进程已退出。',
        '保留四卡要求和全部Smoke通过后才开始Full的门槛；没有改成三卡、混用旧结果或把Smoke冒充机制结论。',
        '', '## 已验证和未执行','',
        f"本轮已完成Smoke的来源分解检查{len(validations)}项；具体输出、No-op/Cache/Hook审计和原始Logits哈希均保留。",
        '正式来源损失、Visual−Function对比、L25–L31连续性、Recent−Older效应均尚未测得；不填零、不预测方向、不绘制正式论文图。',
        '五张主图、CSV汇总和中文研究报告的生成代码已实现，但受全部200图完成校验约束。',
        '', '## 恢复','', 'GPU0及其余三卡空闲后，按[README.md](README.md)原命令恢复；有效Smoke记录会复用，冻结数据/采集源码不变。',
        f"本次快照：`partial/{stamp}/status.json`；实际GPU状态记录于该快照，不能把以后资源变化当作已有实验结果。"]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'research_summary.md').write_text('# 中文研究总结：partial\n\n尚未执行正式200图采集，八项研究问题均待真实结果回答。\n\n'+
        '\n'.join(f'{i}. {q}：尚无正式证据。' for i,q in enumerate([
            'Visual在L26/L31依赖哪种来源','L31的Whole/Image差异能否由History/Prompt解释','Function来源差异',
            'Relation中层与后层依赖','L25–L31连续深层敏感区','Attention Mass与输出贡献是否一致',
            '生成历史是否承担重要预测作用','哪些因果解释可被支持'],1))+'\n\n不复用上一轮数值冒充本轮独立确认；详见report.md与恢复命令。\n')
    for name in ['report.md','research_summary.md','README.md']:
        shutil.copyfile(OUT/name,folder/name)
    freeze_json(folder/'inventory.json',dict(status='partial',source_commit=config['code_commit'],
        files_sha256={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='inventory.json' and p.suffix not in ['.lock','.tmp']}))
    print(json.dumps({k:status[k] for k in ['status','smoke_complete_physical_gpus','full_completed_images','formal_completed_images']},indent=2))


def complete():
    config=verify();target=verify_targets();gate('ablation')
    initial=json.loads((OUT/'analysis_completion.json').read_text())
    if initial['status']!='complete_collected_analysis':raise ValueError('Statistical analysis not complete')
    for name,h in initial['files_sha256'].items():
        if sha256(OUT/name)!=h:raise ValueError('Changed collected analysis artifact: '+name)
    for name,h in initial['analysis_source_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Changed completed analysis source: '+name)
    count=0
    for rank in range(4):
        path=sorted((OUT/f'ablation/shard{rank}/checkpoints').glob('*.json'))[0]
        record=json.loads(path.read_text());native=json.loads((OUT/f'full/shard{rank}/checkpoints/{path.name}').read_text())
        with np.load(native['raw_path']) as arrays:base=arrays['full_logits']
        with np.load(record['raw_path']) as arrays:altered=arrays['altered_logits']
        rows=[r for r in record['rows'] if r['mode']!='Full']
        for index in [0,len(rows)//2,len(rows)-1]:
            row=rows[index];calc=metrics(torch.from_numpy(base[row['generation_step']]),altered[index])
            for key in ['loss','abs_delta_logp','js_divergence','relative_probability_change','full_token_entropy','top1_flip']:
                if calc[key]!=row[key]:raise ValueError('Raw metric check failed: '+key)
            count+=1
    readme();snapshot=gpu_snapshot()
    freeze_json(OUT/'delivery_review.json',dict(passed=True,raw_metrics_exact=count,snapshot=snapshot,
        original_analysis_inventory_sha256=sha256(OUT/'analysis_completion.json'),frozen_source_commit=config['code_commit']))
    freeze_json(OUT/'final_completion.json',dict(status='complete',images=200,positions=target['positions'],
        intervention_forwards=46*target['positions'],collector_commit=config['code_commit'],selected_sha256=target['selected_sha256'],
        no_accuracy_or_speedup_claim=True,analysis_sources={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')
            if str(p.relative_to(ROOT)) not in config['source_sha256']},
        files_sha256={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_completion.json' and p.suffix not in ['.lock','.tmp']}))
    print('Complete delivery inventory and12 independent raw metrics validated.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['partial','complete']);args=parser.parse_args()
    partial() if args.stage=='partial' else complete()
