"""Publish reviewed CPU results; never allocate CUDA or schedule new inference."""
from .common import *
from relay_source_causality.common import hardware

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT.parent,text=True).strip()

def main():
    branch='codex/semantic-single-gpu-search3h';assert git('branch','--show-current')==branch
    assert git('rev-parse','HEAD')=='f1d7d503f4529e1d9820821275c403a87a866290';mainref=git('rev-parse','main');assert mainref=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT.parent).returncode==0
    done=read(P/'completion.json');assert done['status']=='COMPLETE';assert read(P/'validation.json')['passed'];assert read(P/'gpu_release.json')['all_own_children_exited'];assert remaining()>180
    driver=read(P/'driver_started.json');assert all(not Path('/proc',str(pid)).exists() for pid in driver['children'])
    for lock in P.rglob('worker.lock'):
        dest=PRIVATE/('retired_'+str(lock.relative_to(P)).replace('/','_'));assert not dest.exists();lock.rename(dest)
    stats=list(csv.DictReader((P/'paired_comparisons.csv').open()));summary=list(csv.DictReader((P/'accuracy_summary.csv').open()));win=summary[0];sa=next(x for x in summary if x['condition']=='S-assistant');q=next(x for x in summary if x['condition']=='Question-Only');p=next(x for x in stats if x['baseline']=='S-assistant')
    from .pipeline import load
    dd,mm=load('S-assistant');changes=[]
    for r in csv.DictReader((P/'per_question.csv').open()):
        sp=spec(r['question_id']);old=mm.select(sp,dd['config']);new=json.loads(r['writers']);added=set(new)-set(old);removed=set(old)-set(new)
        if old==new:assert r['answer']==r['SA_answer']
        changes.append(dict(question_id=r['question_id'],question=sp['question'],old_k=len(old),new_k=len(new),R=len(sp['R']),changed=old!=new,added_positions=json.dumps(sorted(added)),removed_positions=json.dumps(sorted(removed)),added_groups=json.dumps([dict(words=g['words'],positions=g['positions'],question=g['question']) for g in sp['groups'] if set(g['positions'])<=added],ensure_ascii=False),old_answer=r['SA_answer'],new_answer=r['answer'],old_correct=r['SA_correct'],new_correct=r['correct']))
    writecsv(P/'token_changes.csv',changes)
    events=list(csv.DictReader((P/'checkpoint_index.csv').open()));resident=read(P/'gpu_release.json')['resident_seconds']+read(P/'attempt0_audit/gpu_release.json')['resident_seconds'];atomic_json(P/'resource_audit.json',dict(UTC=utc(),one_GPU_UUID='GPU-b08489af-0a8a-eb48-55ba-8346733af417',model_driver_resident_seconds_including_restarted_attempt=resident,new_unique_generation_CUDA_event_ms=sum(float(r['cuda_event_ms']) for r in events),new_unique_generation_wall_seconds=sum(float(r['seconds']) for r in events),elapsed_wall_seconds=time.monotonic()-read(P/'clock.json')['start_monotonic'],clock_not_reset=True))
    text=(P/'report_zh.md').read_text(encoding='utf-8');text+='\n## 直接比较与复现\n\n'
    text+=f"本轮最终方案相对S＋ASSISTANT净{int(win['correct'])-int(sa['correct']):+d}题，ΔACC={float(p['delta_pp']):+.3f}pp，95%CI [{float(p['CI95_low_pp']):+.3f}, {float(p['CI95_high_pp']):+.3f}]pp；相较Question-Only净{int(win['correct'])-int(q['correct']):+d}题。预算分别为新方案{100*float(win['mean_R_ratio']):.3f}%、S＋ASSISTANT {100*float(sa['mean_R_ratio']):.3f}%、Question-Only {100*float(q['mean_R_ratio']):.3f}%。不能仅以新增正确题忽略新增损害及更高预算。\n\n"
    text+='本轮在修正上轮多一级结果目录的Git索引路径时主动暂停了自己的Worker，保留attempt0_audit及全部Checkpoint，修正的是来源审计路径而非推理/选择器；同一时钟下重做单卡Smoke并精确恢复。未发生放宽容差或精度变更。控制台继承的BOTH_GPU_SMOKES_PASSED标签不代表双卡，本轮protocol.gpus只有GPU1且仅worker0加载模型。\n\n'
    text+='排序middle的冻结公式使用原生绝对位置除以末Question位置的距离，因此不能把它解释为按Question内部相对坐标选择中心Token；公式与所有负面结果照实保留，没有在GPU运行后修订选择规则。\n\n'
    text+='只读CPU复现命令（沿用本文的PYTHONPATH和冻结资源路径）：`python -m semantic_single_gpu_search3h.reproduce`。它逐题重建全部3000个接收集合并与winner_freeze.json严格比较，不加载Gold或模型。\n'
    (P/'report_zh.md').write_text(text,encoding='utf-8')
    state=hardware();assert not any('GPU-b08489af-0a8a-eb48-55ba-8346733af417' in x for x in state['compute_processes'].splitlines());atomic_json(P/'delivery_validation.json',dict(UTC=utc(),passed=True,scientific_validation_sha256=sha(P/'validation.json'),completion_sha256=sha(P/'completion.json'),figures_visually_reviewed=True,all_owned_processes_exited=True,current_resource=state,source_manifest={str(f.relative_to(ROOT.parent)):sha(f) for f in Path(__file__).parent.glob('*.py')},elapsed_seconds=time.monotonic()-read(P/'clock.json')['start_monotonic']))
    index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f!=P/'artifact_index.csv'];writecsv(P/'artifact_index.csv',index)
    for r in index:assert sha(ROOT/r['path'])==r['sha256']
    files=[f for f in Path(__file__).parent.glob('*.py')]+[ROOT/r['path'] for r in index]+[P/'artifact_index.csv'];paths=[str(f.relative_to(ROOT.parent)) for f in files];assert all(x.startswith(('vlm_research/semantic_single_gpu_search3h/','vlm_research/results/semantic_single_gpu_search3h/')) for x in paths)
    specfile=PRIVATE/'git_paths.nul';specfile.write_bytes(b'\0'.join(x.encode() for x in paths)+b'\0');subprocess.run(['git','add','-f','--pathspec-from-file='+str(specfile),'--pathspec-file-nul'],cwd=ROOT.parent,check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT.parent,check=True)
    body=PRIVATE/'commit_message.txt';body.write_text(f"Complete single-GPU S-preserving receiver search\n\nScreen input-only whole-group receivers through 60/180/360/600\nquestions, then freeze one candidate for original GQA3000.\n{done['winner']}: {done['correct']}/3000, mean R {100*done['mean_R_ratio']:.3f}%.\nPreserve strict smoke, checkpoints, all harms, cluster bootstrap,\naudit-path repair and exact resume without resetting the clock.\nAll previously seen data are explicitly exploratory.\n",encoding='utf-8')
    subprocess.run(['git','-c','gc.auto=0','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(body),'--quiet'],cwd=ROOT.parent,check=True);head=git('rev-parse','HEAD');print('COMMITTED',head,flush=True);subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT.parent,check=True);assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==head and git('rev-parse','main')==mainref and not git('status','--porcelain')
    elapsed=time.monotonic()-read(P/'clock.json')['start_monotonic'];atomic_json(PRIVATE/'delivery_receipt.json',dict(UTC=utc(),commit=head,branch=branch,push_verified=True,working_tree_clean=True,main_unchanged=mainref,elapsed_seconds_through_push=elapsed,within_three_hours=elapsed<10800,all_owned_GPU_workers_exited=True,report_url=f'https://github.com/reyuyu/ATE/blob/{head}/vlm_research/results/semantic_single_gpu_search3h/report_zh.md'));print(read(PRIVATE/'delivery_receipt.json'),flush=True)

if __name__=='__main__':main()
