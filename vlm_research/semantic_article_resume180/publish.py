"""Seal reviewed figures, commit only this continuation, and verify its push."""
from .common import *
from relay_source_causality.common import hardware

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT.parent,text=True).strip()

def main():
    branch='codex/semantic-article-resume180';assert git('branch','--show-current')==branch
    main_ref=git('rev-parse','main');assert main_ref=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
    assert git('ls-remote','origin','refs/heads/'+branch).split()[0]=='0e1f9500c98a688b00367ff0bf89f0e8882c704c'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT.parent).returncode==0
    done=read(P/'completion.json');assert done['status']=='COMPLETE' and done['correct']==2563 and done['sealed_after_reporting_QA']
    archive=P/'figure_qa_pass';previous=list(csv.DictReader((P/'artifact_index.csv').open()));changed=[]
    for x in previous:
        if sha(ROOT/x['path'])!=x['sha256']:changed.append(x['path'])
    assert set(Path(x).name for x in changed)=={'task_delta_accuracy.png','task_delta_accuracy.pdf'}
    assert read(P/'figure_QA.json')['numerical_data_unchanged']
    atomic_json(P/'delivery_source_manifest.json',{str(f.relative_to(ROOT.parent)):sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    (P/'completion.json').rename(archive/'completion_root_original.json')
    index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f not in [P/'artifact_index.csv',P/'completion.json'] and f.suffix!='.lock'];writecsv(P/'artifact_index.csv',index)
    done.update(UTC=utc(),artifact_index_sha256=sha(P/'artifact_index.csv'),delivery_source_manifest_sha256=sha(P/'delivery_source_manifest.json'),final_figure_QA_passed=True,elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60);atomic_json(P/'completion.json',done)
    for x in index:assert sha(ROOT/x['path'])==x['sha256']
    assert remaining()>180
    paths=[str(f.relative_to(ROOT.parent)) for f in sorted(Path(__file__).parent.glob('*.py'))]+[str(ROOT/x['path']).removeprefix(str(ROOT.parent)+'/') for x in index]+[str((P/'artifact_index.csv').relative_to(ROOT.parent)),str((P/'completion.json').relative_to(ROOT.parent))]
    assert all(x.startswith(('vlm_research/semantic_article_resume180/','vlm_research/results/semantic_article_resume180/run_20261104/')) for x in paths)
    specfile=PRIVATE/'git_paths.nul';specfile.write_bytes(b'\0'.join(x.encode() for x in paths)+b'\0');subprocess.run(['git','add','-f','--pathspec-from-file='+str(specfile),'--pathspec-file-nul'],cwd=ROOT.parent,check=True)
    staged=git('diff','--cached','--name-only').splitlines();assert len(staged)==len(set(paths)) and all(x in paths for x in staged)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT.parent,check=True)
    message=PRIVATE/'commit_message.txt';message.write_text('Complete frozen Semantic augmentation continuation on GQA3000\n\nEvaluate eight S-preserving input rules through 180/360/600 stages,\nthen expand one frozen candidate to the original 3000 questions.\nS+ASSISTANT+noun+article scores 2563/3000 at 46.336% mean R;\nit adds only three correct answers over S+ASSISTANT and remains\nbelow Question-Only. Preserve all harms and exploratory paired CIs.\n\nValidate five CPU tests, strict three-case smoke on each GPU,\n2948 unique new generations, masks, independent caches and hashes.\nRelease both models immediately after generation. No new samples,\nmodel training, numerical tolerance changes, or main changes.\n',encoding='utf-8')
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(message),'--quiet'],cwd=ROOT.parent,check=True);head=git('rev-parse','HEAD');print('COMMITTED',head,'files',len(staged),flush=True)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT.parent,check=True);assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==head
    assert git('rev-parse','main')==main_ref and not git('status','--porcelain')
    elapsed=time.monotonic()-read(P/'clock.json')['start_monotonic'];receipt=dict(UTC=utc(),commit=head,branch=branch,pushed_and_verified=True,working_tree_clean=True,main_unchanged=main_ref,elapsed_seconds_through_push=elapsed,within_three_hours=elapsed<=10800,artifact_index_sha256=sha(P/'artifact_index.csv'),completion_sha256=sha(P/'completion.json'),resource_state=hardware(),report_url=f'https://github.com/reyuyu/ATE/blob/{head}/vlm_research/results/semantic_article_resume180/run_20261104/report_zh.md')
    atomic_json(PRIVATE/'delivery_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
