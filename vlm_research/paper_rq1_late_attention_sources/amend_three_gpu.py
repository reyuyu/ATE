"""Archive the unstarted four-GPU attempt without replacing frozen samples."""
import json
from pathlib import Path
import shutil
from benchmarks.dataset_registry import freeze_json,sha256
from .protocol import OUT
from .scheduling import configuration
from .finalize import gpu_snapshot


def main():
    amendment=OUT/'three_gpu_amendment.json'
    if amendment.exists():raise ValueError('Amendment already recorded; do not archive current workers')
    if any((OUT/name).exists() for name in ['full','ablation','selected_token_manifest.json','target_freeze.json']):
        raise ValueError('Only amend before formal Full/target selection')
    old=json.loads((OUT/'execution_manifest.json').read_text())
    for key,name in [('sample_sha256','sample_manifest.json'),('data_sha256','data_manifest.json')]:
        if sha256(OUT/name)!=old[key]:raise ValueError('Frozen images changed')
    samples=json.loads((OUT/'sample_manifest.json').read_text())['samples']
    if len(samples)!=200 or any(sum(r['rank']==i for r in samples)!=50 for i in range(4)):
        raise ValueError('Original four logical shards must remain intact')
    archive=OUT/'attempts/four_gpu_resource_blocked';archive.mkdir(parents=True,exist_ok=False)
    inventory={}
    for name in ['execution_manifest.json','cpu_validation.json','cpu_tests.txt','final_cpu_tests.txt',
                 'statistics_tests.txt','smoke','report.md','research_summary.md','README.md']:
        path=OUT/name
        if not path.exists():continue
        files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        inventory.update({str(p.relative_to(OUT)):sha256(p) for p in files})
        shutil.move(str(path),str(archive/name))
    freeze_json(amendment,dict(status='operational_amendment_before_formal_gpu',
        user_authorization='User explicitly authorized three GPUs instead of four.',
        old_collector_commit=old['code_commit'],sample_sha256=sha256(OUT/'sample_manifest.json'),
        data_sha256=sha256(OUT/'data_manifest.json'),scheduling=configuration(),
        full_images_before_amendment=0,formal_images_before_amendment=0,
        scientific_conditions_unchanged=True,engine_unchanged=True,labels_unchanged=True,
        old_smoke_not_reused_for_new_binding=True,
        archived_paths_sha256=inventory,snapshot=gpu_snapshot()))
    (OUT/'report.md').write_text('# Late-Layer Attention Source Decomposition\n\n'
        '状态：三卡调度已获授权，正式结果尚未完成。\n\n'
        '冻结200图及四个50图逻辑分片不变，GPU1负责分片0/3，GPU2负责分片1，GPU3负责分片2。'
        'Full与干预固定同图同卡；GPU0完全不参与。旧四卡尝试及Smoke已归档，不混入新绑定。'
        '重新完成CPU验证、冻结代码和三卡Smoke后才运行正式采集。详见three_gpu_amendment.json。\n')
    print('Archived initial attempt; all200 frozen images and shard assignments unchanged.',flush=True)


if __name__=='__main__':main()
