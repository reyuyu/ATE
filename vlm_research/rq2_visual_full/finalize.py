"""Final delivery and release audit after both immutable workers have exited."""
import subprocess
import time

from benchmarks.dataset_registry import freeze_json, sha256
from .run import OUT, ROOT, GPUS, GPU_INDICES, read, verify


def main():
    c=verify();v=read(OUT/'validation.json');assert v['status']=='complete' and v['new_generations']==11040
    table=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader'],text=True)
    assert not any(uuid in apps for uuid in GPUS)
    freeze_json(OUT/'release_validation.json',dict(passed=True,checked_epoch=time.time(),
        gpu_indices=GPU_INDICES,gpu_uuids=GPUS,selected_gpus_no_compute_process=True,
        gpu_snapshot=table,all_compute_apps_snapshot=apps,other_gpu_processes_never_signalled=True))
    freeze_json(OUT/'delivery_binding.json',dict(finalizer_sha256=sha256(ROOT/'rq2_visual_full/finalize.py'),
        execution_protocol_sha256=sha256(OUT/'protocol.json'),postprocessing_binding_sha256=sha256(OUT/'postprocessing_binding.json')))
    main_smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(len(GPUS))]
    prior_smokes=list((OUT/'startup_three_gpu_attempt').glob('shard*/smoke.json'))
    startup_seconds=sum(read(p)['seconds'] for p in prior_smokes)
    smoke_seconds=sum(s['seconds'] for s in main_smokes)
    gpu_seconds=sum(v['worker_seconds'].values())+startup_seconds+smoke_seconds
    text=['','## 解释上的额外限制','',
        'All Image OFF和Keep L14的Accuracy点估计也接近Full，不能据此认定L14/L19是唯一必要的视觉层。Keep L14+L19相对All Image OFF的恢复比较经四项校正后仍包含0；本轮证据支持预声明的条件性充分性/非劣效结论，而不是两层独特性、必要性或显著优越性。不能把未预声明为主检验的其他条件事后改为新的确认性非劣效发现。',
        '', '## 最终资源释放','',
        '正式GPU1、GPU3的Worker均正常退出；交付审计时两卡无CUDA计算进程。GPU2启动预检冲突后未分配模型，也未向其他GPU进程发送终止信号。原始快照见release_validation.json。',
        f"正式Worker累计{sum(v['worker_seconds'].values())/60:.2f}分钟；正式Smoke累计{smoke_seconds/60:.2f}分钟；三卡启动尝试中成功Smoke累计{startup_seconds/60:.2f}分钟。合计GPU任务时长{gpu_seconds/60:.2f}分钟，含相应模型加载，不是CUDA核执行时间或加速比。",
        '执行源码、统计/绘图源码、CPU验证、所有复用来源和最终结果均保存哈希；artifact_hashes.json覆盖全部输出（不包含临时锁文件和清单自身）。本轮到此停止。','']
    report=OUT/'report_zh.md';report.write_text(report.read_text()+'\n'.join(text))
    files={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file()
           and p.name!='artifact_hashes.json' and p.suffix not in ('.lock','.tmp')}
    freeze_json(OUT/'artifact_hashes.json',files)
    print('Final delivery:',len(files),'hashed artifacts; selected GPUs released; experiment stopped',flush=True)


if __name__=='__main__':main()
