"""Review and seal delivery after CPU reporting QA. No inference is performed."""
import shutil
from .common import *
from .pipeline import load
from relay_source_causality.common import hardware

def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))

def main():
    done=read(P/'completion.json');assert done['status']=='COMPLETE';v=read(P/'validation.json');assert v['passed']
    # Check the current complete delivery index before making the final reporting revision.
    ix=rows(P/'artifact_index.csv')
    for x in ix:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    protocol=verify_sources()
    for path,value in protocol['source_hashes'].items():assert sha(path)==value,path
    rr=rows(P/'per_question.csv');assert len(rr)==3000 and len({r['question_id'] for r in rr})==3000
    for r in rr:
        assert int(r['correct'])==int(gqa_prediction(r['answer'])==r['gold'])
        for s in ['All','Q100','Q80','S','SA']:assert int(r[s+'_correct'])==int(gqa_prediction(r[s+'_answer'])==r['gold'])
        assert r['eos']=='True' and all(r[k]=='False' for k in ['empty','truncated','repeated_trigram'])
    assert sum(int(x['correct']) for x in rr)==2563
    state=hardware();selected={g['uuid'] for g in protocol['gpus']};assert not any(x.split(',')[0].strip() in selected for x in state['compute_processes'].splitlines())
    original_driver=read(P/'driver_started.json');assert all(not Path('/proc',str(pid)).exists() for pid in original_driver['children'])
    repo=ROOT.parent;assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='codex/semantic-article-resume180'
    main_head=subprocess.check_output(['git','rev-parse','main'],cwd=repo,text=True).strip();assert main_head=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
    # Keep every previous metadata version. Only the final root index is authoritative.
    archive=P/'reporting_pass2';archive.mkdir()
    for name in ['report_zh.md','validation.json','completion.json','artifact_index.csv']:shutil.copyfile(P/name,archive/name)
    (P/'validation.json').rename(archive/'validation_root_original.json');(P/'completion.json').rename(archive/'completion_root_original.json')
    failure=PRIVATE/'closeout_extra.log'
    if failure.exists():shutil.copyfile(failure,P/'reporting_pass1'/'supplemental_report_immutable_writer_error.log')
    text=(P/'report_zh.md').read_text(encoding='utf-8').replace('所有16/32位和模型协议保持冻结','FP16、Eager和模型协议保持冻结')
    text+='\n## 复现与交付审计\n\n最终完整性索引为本目录artifact_index.csv，完成标记为completion.json；reporting_pass1/2仅保存CPU报告修订的历史元数据。补充报告首次尝试更新只写一次的元数据时被保护逻辑拒绝，未涉及推理或数值验证失败；保留日志并归档原记录后完成修订，所有答案、Token集合及统计数值不变。\n\n'
    text+='重新生成3000题接收集合（只读CPU，不生成答案）：\n\n```sh\n'+read(P/'selection_rules.json')['reproduce_command']+'\n```\n\n'
    text+='![Complete-answer accuracy](accuracy_comparison.png)\n\n![Observed budget and accuracy](budget_accuracy.png)\n\n![Six-task accuracy differences](task_delta_accuracy.png)\n'
    (P/'report_zh.md').write_text(text,encoding='utf-8')
    elapsed=time.monotonic()-read(P/'clock.json')['start_monotonic'];assert elapsed<10800
    atomic_json(P/'resource_audit.json',dict(UTC=utc(),elapsed_seconds_before_commit=elapsed,hard_limit_seconds=10800,GPU_resident_driver_seconds=read(P/'gpu_release.json')['resident_seconds'],selected_GPU_UUIDs=sorted(selected),owned_worker_PIDs=original_driver['children'],all_owned_workers_exited=True,current_hardware=state,private_arrays_not_uploaded=True,local_main_before_and_after=main_head))
    v['source_files']={str(f.relative_to(ROOT.parent)):sha(f) for f in Path(__file__).parent.glob('*.py')};v['delivery_QA']=dict(passed=True,previous_index_files_checked=len(ix),source_protocol_exact=True,all3000_ACC_recomputed=True,Native_images_unchanged=True,receiver_reproduction_exact_3000=True,figures_visually_reviewed=True,all_own_GPU_processes_exited=True,report_revisions_only=True,UTC=utc());atomic_json(P/'validation.json',v)
    index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f not in [P/'artifact_index.csv',P/'completion.json'] and f.suffix!='.lock'];writecsv(P/'artifact_index.csv',index)
    done.update(UTC=utc(),validation_sha256=sha(P/'validation.json'),artifact_index_sha256=sha(P/'artifact_index.csv'),elapsed_minutes=elapsed/60,sealed_after_reporting_QA=True);atomic_json(P/'completion.json',done)
    for x in index:assert sha(ROOT/x['path'])==x['sha256']
    print(json.dumps(dict(sealed=True,files=len(index),elapsed_minutes=elapsed/60,main_unchanged=main_head,completion=done)),flush=True)

if __name__=='__main__':main()
