"""Post-run delivery audit; never alter frozen worker code or checkpoints."""
import subprocess
import time

from benchmarks.dataset_registry import freeze_json, sha256
from vlm_common import write_json
from .run import OUT, ROOT, GPUS, read, verify


def main():
    verify()
    prior=read(OUT/'artifact_hashes.json')
    for name,h in prior.items():
        assert sha256(OUT/name)==h
    smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(3)]
    fields=['python','torch','transformers','cuda_runtime','cudnn','gpu',
            'model_id','model_revision','parameter_dtypes']
    assert all({k:s['environment'][k] for k in fields} ==
               {k:smokes[0]['environment'][k] for k in fields} for s in smokes)
    assert all(s['checks']==smokes[0]['checks'] for s in smokes)
    table=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader'],text=True)
    assert not any(uuid in apps for uuid in GPUS)
    freeze_json(OUT/'release_validation.json',dict(passed=True,checked_epoch=time.time(),
        gpu_indices=[1,2,3],gpu_uuids=GPUS,compute_processes_absent=True,
        gpu_snapshot=table,compute_apps=apps,gpu0_not_touched=True,
        three_gpu_environment_and_all_smoke_checks_exact=True))
    report=OUT/'report_zh.md'
    freeze_json(OUT/'delivery_binding.json',dict(
        finalizer_source_sha256=sha256(ROOT/'rq2_visual_sufficiency/finalize.py'),
        aggregation_manifest_before_delivery_sha256=sha256(OUT/'artifact_hashes.json'),
        aggregation_report_before_delivery_sha256=sha256(report),
        reason='Post-run interpretation and GPU release audit only. Frozen execution and raw results unchanged; refresh final artifact hashes after adding delivery notes.'))
    extra=['','## 最终研究判断与下一步建议','',
        '- 两层保留在本240题的完整答案Accuracy点估计接近Full（同为213/240），但并非逐题相同：ChooseAttr纠正1题，CompareAttr损害1题。普通配对95%CI为[-1.25,+1.25]pp，六比较同时区间为[-1.674,+1.674]pp，尚不能排除实质性损害。',
        '- 在只保留L14的基础上，增加L19恢复2题，未额外损害题目；相对全关恢复4题、未额外损害题目。这是可观察的探索性恢复效果，但两个恢复比较的校正区间下界均为0，不能报告为校正后显著收益。',
        '- 删除L14+L19出现3题损害、0题纠正，净下降1.25pp；同时区间[-3.376,0]pp，不能据此声称统计上确证了损害大小。',
        '- 其余30层联合Image OFF在本轮未出现灾难性额外损害；平均首Token Loss为0.001431 nat，匹配历史单层Loss之和为0.001325，差0.000106。全32层关闭的额外非加性Loss为0.003816 nat。不能把单层低敏感直接推广为任何多层组合都安全；这里的联合效应由实际生成而非单层相加验证。',
        '- 建议把Keep L14+L19作为值得独立确认的候选，而不是已验证的充分/等价方案。后续如获批准，应预先固定可接受损害界限、以尚未参与筛层的剩余2760题独立验证，并保留任务分组和逐题损害。本轮已经停止，未运行该扩展。',
        '', '## 资源释放与执行说明','',
        'GPU1、GPU2、GPU3的正式Worker均已正常退出。释放审计时三卡均为27MiB、0%利用率，均无CUDA计算进程；GPU0原有任务保持运行。原始nvidia-smi快照见release_validation.json。',
        '正式三卡并行墙钟约2.30分钟，累计Worker时间409.0秒；Smoke累计131.5秒，合计约9.01 GPU分钟（均包含相应模型加载，不含CPU编程、统计和Git上传）。',
        '首轮CPU检查发现旧Mask禁止Cache。仅新增空缓存首次Prefill适配，保留原knockout_mask算法；已有缓存及Decode仍被拒绝。修正后15项CPU检查及三卡真实Smoke均通过，未放宽数值容差；初始失败记录保留在failures/，正式Worker无失败。',
        '最终artifact_hashes.json覆盖报告、图表、日志、原始分布和检查点；delivery_binding.json记录执行结束后的纯交付补充源码及此前聚合快照哈希。','']
    report.write_text(report.read_text()+'\n'.join(extra))
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json' and p.suffix not in ('.lock','.tmp')}
    write_json(OUT/'artifact_hashes.json',hashes)
    print('Delivery report, identical three-GPU smoke audit, final hashes and GPU release verified',flush=True)


if __name__=='__main__':main()
