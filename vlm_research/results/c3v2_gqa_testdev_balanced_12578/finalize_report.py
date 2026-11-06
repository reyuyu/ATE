"""Finalize the report only after complete inference and independent scoring."""
import csv,gzip,hashlib,json,sqlite3,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from vlm_research.gqa_testdev_c3.common import OUT,LABELS,atomic,digest,freeze_check,now,read,samples,score,sha,unpack

def main():
    freeze=freeze_check();protocol=read(OUT/'protocol.json');manifest=samples();by_id={s['question_id']:s for s in manifest}
    reference=read(ROOT/'vlm_research/results/c3v2_gqa_val_balanced_full/execution_freeze.json')
    assert all(freeze['weights'].get(p)==h for p,h in reference['weights'].items())
    assert sha(ROOT/'vlm_research/gqa_testdev_c3/engine.py')==sha(ROOT/'vlm_research/gqa_balanced_c3/engine.py')
    assert len(by_id)==len(manifest)==12578
    for path,h in protocol['annotations'].items():assert sha(path)==h,path
    image_audit=read(OUT/'source_dataset_audit.json')
    for row in image_audit['images']:assert sha(row['path'])==row['sha256']
    records={};input_bindings={}
    for rank in range(2):
        path=OUT/f'worker{rank}/results.sqlite'
        done=read(OUT/f'worker{rank}/completion.json');assert done['status']=='COMPLETE'
        assert sha(path)==done['database_sha256']
        with gzip.open(path.with_suffix('.sqlite.gz'),'rb') as stream:
            h=hashlib.sha256()
            while b:=stream.read(1024*1024):h.update(b)
        assert h.hexdigest()==sha(path)
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        for row in db.execute('SELECT payload,sha256 FROM results'):
            r=unpack(row);key=r['question_id'],r['condition'];assert key not in records
            s=by_id[key[0]];g=r['generation'];assert key[1] in LABELS
            assert r['binding']==digest(freeze) and r['sample_sha256']==digest(s)
            assert r['score']==score(s,g['text']) and r['image_sha256']==s['image_sha256']
            assert g['all_effective_logits_finite'] and g['independent_cache'] and g['logits_to_keep']==1
            if key[0] in input_bindings:assert r['input_binding']==input_bindings[key[0]]
            else:input_bindings[key[0]]=r['input_binding']
            records[key]=r
        db.close()
    assert len(records)==37734 and len(input_bindings)==12578
    assert all((s['question_id'],c) in records for s in manifest for c in LABELS)
    assert read(OUT/'resource_release.json')['all_workers_exited']
    summary=list(csv.DictReader((OUT/'accuracy_summary.csv').open()))
    pairs=list(csv.DictReader((OUT/'paired_comparisons.csv').open()))
    official=list(csv.DictReader((OUT/'official_score_crosscheck.csv').open()))
    rounded_agreement=all(abs(float(x['delta_pp']))<=.00500001 for x in official)
    for row in summary:
        correct=sum(records[s['question_id'],row['condition']]['score'] for s in manifest)
        assert correct==int(row['correct'])
        assert abs(float(row['accuracy_percent'])-100*correct/12578)<1e-10
    report=(OUT/'report_zh.md').read_text()
    report=report.replace('| Native |','| Native Full |').replace('| C3-V2 |','| Physical C3-V2 |').replace('| C3-V2+L11-L24 |','| C3-V2 + L11–L24 |')
    report=report.split('\n<!-- final-audit -->')[0]
    table='\n'.join(f"| {r['comparison']} | {r['delta_pp']} | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_token_flips']} |" for r in pairs)
    task_rows=list(csv.DictReader((OUT/'task_accuracy.csv').open()));task_groups={}
    for r in task_rows:task_groups.setdefault((r['grouping'],r['group']),{})[r['condition']]=r
    task_table=[]
    for (family,group),values in task_groups.items():
        n,c,a=[float(values[x]['accuracy_percent']) for x in ['Native','C3-V2','C3-V2+L11-L24']]
        task_table.append(f"| {family}/{group} | {values['Native']['questions']} | {n:.4f} | {c:.4f} | {a:.4f} | {c-n:+.4f} | {a-n:+.4f} |")
    task_table='\n'.join(task_table)
    q=task_groups['structural','query']
    q_gain=float(q['C3-V2+L11-L24']['accuracy_percent'])-float(q['C3-V2']['accuracy_percent'])
    q_net=round(q_gain*int(q['Native']['questions'])/100)
    negative_groups=', '.join(f"{g}: {float(task_groups['structural',g]['C3-V2+L11-L24']['accuracy_percent'])-float(task_groups['structural',g]['C3-V2']['accuracy_percent']):+.4f} pp" for g in ['verify','logical'])
    anomalies='\n'.join(f"| {r['condition']} | {r['eos']} | {r['truncated']} | {r['empty']} | {r['repeated']} |" for r in summary)
    max_error=max(c['equivalence'].get('max_abs_logits',0) for rank in range(2) for x in read(OUT/f'worker{rank}/smoke.json')['checks'] for c in x['conditions'])
    agreement='官方工具输出精度为两位小数；三条件与主评分在该舍入精度下一致。' if rounded_agreement else '官方评分存在超出两位小数舍入的差异，必须保留两套数值并进一步说明原因。'
    report+=f'''\n<!-- final-audit -->
直接回答：原版Physical C3-V2相对同协议Native的差值为{float(pairs[0]['delta_pp']):+.4f} pp；L11–L24回看相对原版C3-V2为{float(pairs[2]['delta_pp']):+.4f} pp，回看相对Native为{float(pairs[1]['delta_pp']):+.4f} pp。以上全部来自同一testdev_balanced队列，不使用公开62.0%作为配对基线。

| Comparison | ΔACC (pp) | 基线正确→干预错误 | 基线错误→干预正确 | 首Token变化 |
|---|---:|---:|---:|---:|
{table}

| 官方类别 | 题数 | Native ACC (%) | C3 ACC (%) | 回看 ACC (%) | C3−Native (pp) | 回看−Native (pp) |
|---|---:|---:|---:|---:|---:|---:|
{task_table}

分组结果仅作描述：回看相对C3净增加333道正确题，其中Query组净增加{q_net}题（+{q_gain:.4f} pp）。并非所有结构类别都受益，{negative_groups}。没有据此调整层集合或选择问题。

| Condition | EOS结束 | 截断 | 空答案 | 重复标记 |
|---|---:|---:|---:|---:|
{anomalies}

{agreement}原始官方工具stdout、逐题预测和两套分数均保留。

正确性边界：Native/HF/No-op有效Logits严格逐位一致；原版C3与冻结的历史物理参考逐位一致。回看Physical与Dense功能参考使用历史已冻结的FP16验收门槛，Smoke最大Logits绝对差为{max_error:g}，未声称二者逐位等价，也未修改门槛。两张卡分别通过全部三个短/中/长输入Smoke。全部37,734条记录重新核验哈希、独立Cache标记和有限Logits标记；同题三条件的实际输入绑定完全一致。

后处理记录：原报告脚本在构造异常表时重复传入condition字段，首次运行报错；失败日志finish.log与修复差异postprocessing_changes.json均保留。独立副本finish_report.py仅修正该字段重复及脚本导入路径，冻结推理源码、输入和全部答案未修改，也没有重新运行GPU。CPU复算顺序为：先运行vlm_research.gqa_testdev_c3.analyze，再运行本目录finish_report.py，最后运行本目录finalize_report.py。仓库保存了两份results.sqlite.gz，解压可恢复原始SQLite；official_testdev_balanced_questions.json.gz可恢复官方原问题文件，official_gqa_evaluator.py是实际使用评测器的逐字节副本。

![Three-condition quality comparison](quality_comparison.png)

源数据：[Stanford GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)；[LLaVA官方GQA评测入口](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/gqa.sh)。图片传输和字节等价性限制见source_dataset_audit.json。本轮没有进行测速或扩展条件。源码、Manifest和Checkpoint索引见execution_freeze.json、sample_manifest.jsonl.gz及result_hashes.json。
'''
    (OUT/'report_zh.md').write_text(report)
    audit=dict(status='PASS',at=now(),records=37734,questions=12578,image_clusters=398,
        identical_actual_inputs_across_three_conditions=True,all_checkpoint_hashes_verified=True,
        historical_model_weight_shas_match=True,historical_generation_engine_byte_identical=True,
        compressed_checkpoints_match_original_database=True,source_annotations_and_images_rehashed=True,
        scores_independently_recomputed=True,official_agreement_to_printed_precision=rounded_agreement,
        all_task_workers_exited=True,postprocessor_sha256=sha(Path(__file__)))
    atomic(OUT/'final_delivery_audit.json',audit)
    done=read(OUT/'completion.json');done.update(report_sha256=sha(OUT/'report_zh.md'),final_delivery_audit=audit,at=now());atomic(OUT/'completion.json',done)
    atomic(OUT/'result_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='result_hashes.json' and p.suffix!='.sqlite' and not p.name.endswith(('-wal','-shm','.pyc'))})
    print(json.dumps(dict(audit=audit,summary=summary,pairs=pairs),indent=2))

if __name__=='__main__':main()
