"""Package the CPU freeze and the initial resource stop; never queries/uses a GPU."""
import collections,datetime,io,unittest,sys
import numpy as np
import torch
from .common import *
from .tests import Audit

def main():
    assert not torch.cuda.is_initialized();plan=read(OUT/'token_plan.json');cases=plan['cases'];audit=read(OUT/'cpu_input_validation.json')
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit));(OUT/'cpu_tests.log').write_text(stream.getvalue());assert test.wasSuccessful(),stream.getvalue()
    # Exact values from the first nvidia-smi tool call in this task. No polling.
    gpu=dict(status='STOP_NO_IDLE_GPU',observation='Initial read-only nvidia-smi query at task start; not a later availability check',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_definition='No compute processes; memory used<200MiB, free>22000MiB, utilization=0; select up to two UUIDs',selected=[],gpu_workers_started=0,model_loads=0,subsequent_gpu_polls=0,GPUs=[dict(index=0,uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',used_MiB=18611,free_MiB=5545,utilization_percent=38,pid=3211916,process='/data/lk/深度学习/.envs/ate-qwen3vl/bin/python',process_memory_MiB=18578),dict(index=1,uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417',used_MiB=2335,free_MiB=21821,utilization_percent=70,pid=3214981,process='/home/xrl/Project/.envs/stable_spectral_20261016/bin/python',process_memory_MiB=2302),dict(index=2,uuid='GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',used_MiB=2337,free_MiB=21819,utilization_percent=53,pid=3214983,process='/home/xrl/Project/.envs/stable_spectral_20261016/bin/python',process_memory_MiB=2304),dict(index=3,uuid='GPU-96bc853f-ceea-2404-c704-c48a3d1ca185',used_MiB=18167,free_MiB=5989,utilization_percent=41,pid=3211917,process='/data/lk/深度学习/.envs/ate-qwen3vl/bin/python',process_memory_MiB=18134)],all_four_have_active_processes=True,other_processes_killed=False,background_wait_or_monitor_created=False)
    save('gpu_preflight.json',gpu)
    rules=dict(boundary='Actual two native R positions immediately after the last Image and before first Question: IDs29871 and13; original offsets, pieces and decodes audited all3000. No textual SPACE/NL insertion.',semantic='Original frozen S from01c82b1; no reselection.',noun_article='Reuse committed spaCy3.8.7/en_core_web_sm3.8.0 parses from11fe70e; N=NOUN/PROPN; D=a/an/the with dep=det and head NOUN/PROPN; map native offsets then close to original Semantic whole groups, no outcome-based repair.',template_control='Choose earliest two complete one-subword lexical Instruction groups by native token position, after Question, excluding Question, B, Image and P-1. Require both outside S and N+D; no split words; otherwise unavailable and retained. Optional random seeds not added.',protocol='LLaVA frozen revision, FP16/Eager/Transformers4.49.0, Greedy32, logits_to_keep=1. All32 P-1/Decode imageblocked; only selected original Prefill R positions can see original Image keys; hard mask, full native QKV/KV/MLP/Residual; no o_proj ablation.',dedup='Within one original question only: exact receiver-position equality; audit matching original prompt/input/image/revision/protocol and checkpoint bytes before reuse. Preserve every condition label.',image_shards='Sorted unique Image SHA256 assigned alternately to two prospective workers; one-worker plan assigns all images0. Freeze question/condition sets independently of GPU availability.',cohorts='All3000 primary; recomputed image-disjoint2640 excludes historical Semantic240 IDs and all images; pilot120 and original12 harm cases separately. Not an external independent test.')
    save('selection_rules.json',rules)
    stats=dict(primary_endpoint='Complete generated answer GQA accuracy using the original gqa_prediction scorer; no Gold NLL for quality assessment.',comparisons=COMPARISONS,bootstrap_repetitions=20000,bootstrap_seed=20261104,cluster='Image SHA256; resample image clusters with replacement, retain every question and paired condition for each selected image; question-weighted sums divided by resampled total question count.',unadjusted_CI='Exploratory two-sided percentile95% CI, [2.5,97.5] percentiles; accuracy differences in percentage points.',primary_multiple_comparison_control=dict(family='Five predeclared comparisons on the main3000-question cohort',method='Bonferroni two-sided percentile99% intervals per comparison',family_alpha=.05,comparisons=5,per_comparison_alpha=.01,percentiles=[.5,99.5],interpretation='Exploratory bootstrap simultaneous bounds; no exact finite-sample guarantee or noninferiority claim.'),secondary_subsets='2640,120,12 and six tasks: descriptive paired95% intervals, explicitly outside primary corrected family.',metrics=['accuracy','delta_vs_All_pp','correct_to_wrong','wrong_to_correct','answer_text_changed','first_token_flip','EOS/empty/truncated/repeated_trigram','actual_writer_k_over_R'],no_interpretation_of_CI_crossing_zero_as_noninferiority=True,no_NLL_computed=True,no_budget_search=True)
    save('analysis_plan.json',stats)
    hist=list(csv.DictReader((OUT/'audited_historical_baselines.csv').open()));available={(r['question_id'],r['condition']):r for r in hist}
    for q,v in cases.items():
        for label,w in v['writers'].items():
            if w is None or (q,label) in available:continue
            u=v['unique'][v['aliases'][label]]
            if not u['reuse']:continue
            cp=read(ROOT/u['reuse']['path']);assert sha256(ROOT/u['reuse']['path'])==u['reuse']['sha256'];g=cp['generation']
            available[(q,label)]=dict(question_id=q,condition=label,correct=int(g['correct']),answer=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),source_path=u['reuse']['path'],source_commit=u['reuse']['commit'],source_sha256=u['reuse']['sha256'])
    cohorts={'main3000':list(cases),'image_disjoint2640':[q for q,v in cases.items() if v['confirmatory2640']],'pilot120':[q for q,v in cases.items() if v['pilot120']],'historical12':[q for q,v in cases.items() if v['historical12']]}
    summary=[]
    for cohort,qids in cohorts.items():
        for label in CONDITIONS:
            rr=[available[(q,label)] for q in qids if (q,label) in available];complete=len(rr)==len(qids)
            summary.append(dict(cohort=cohort,condition=label,cohort_n=len(qids),audited_history_available=len(rr),pending=len(qids)-len(rr),correct=sum(int(r['correct']) for r in rr) if complete else '',accuracy_percent=100*sum(int(r['correct']) for r in rr)/len(qids) if complete else '',status='AUDITED_HISTORY_ONLY' if complete else 'INCOMPLETE_NO_NEW_GENERATION'))
    csvwrite('accuracy_summary.csv',summary)
    rows=[]
    for q in cohorts['historical12']:
        v=cases[q];s=next(s for s in read(OUT/'sample_manifest.json')['samples'] if s['question_id']==q)
        for label in CONDITIONS:
            w=v['writers'][label];a=available.get((q,label));base=v['ND'] if label.startswith(('Noun','N+D')) else v['S']
            rows.append(dict(question_id=q,question=s['question'],gold=s['answer'],condition=label,answer=a['answer'] if a else 'NOT_GENERATED',correct=a['correct'] if a else '',k=len(w) if w is not None else '',R=len(v['R']),writer_positions=json.dumps(w),added_to_reference=json.dumps(sorted(set(w)-set(base))) if w is not None else '',status='AUDITED_HISTORICAL_ANSWER' if a else 'PENDING_GPU',source_commit=a['source_commit'] if a else '',source_path=a['source_path'] if a else ''))
    csvwrite('historical_failure_cases.csv',rows)
    bindings={n:sha256(OUT/n) for n in ['token_plan.json','sample_manifest.json','cohorts.json','prospective_shards.json','selection_rules.json','analysis_plan.json','source_audit.json','parser_environment.json']}
    package=ROOT/'gqa3000_space_nl_visual_writers';sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted(package.glob('*.py'))}
    for p in [ROOT/'gqa3000_token_visual_writing/engine.py',ROOT/'semantic_visual_writers/engine.py',ROOT/'semantic_visual_writers/groups.py',ROOT/'paper_rq1/scorers.py']:sources[str(p.relative_to(ROOT))]=sha256(p)
    save('protocol.json',dict(status='INPUT_AND_ANALYSIS_DESIGN_FROZEN_RESOURCE_STOP',base_commit=BASE,semantic_commit=SEM,native_attention_reference=ATTENTION,model_revision=REVISION,precision='float16',attention='eager',transformers='4.49.0',greedy=True,max_new_tokens=32,logits_to_keep=1,conditions=CONDITIONS,primary_comparisons=COMPARISONS,bindings=bindings,sources=sources,gpus=[],GPU_execution_entry_not_prepared_or_frozen=True,required_before_future_GPU_run='Recheck actual idle GPUs; prepare and freeze execution source against this immutable input design; pass actual per-GPU HF/No-op torch.equal, forbidden actual probabilities0, non-target native masks, independent KV/MLP/input/weight hashes and hook cleanup smoke, then generate. No worker was launched in this attempt.',smoke_ids=['00150746','11697551','11127723'],cpu_tests=test.testsRun,new_Greedy_generations_expected=plan['planned_new_canonical_generations'],logical_new_labels=21000,resource_status='STOP_NO_IDLE_GPU'))
    save('validation.json',dict(status='CPU_ONLY_GPU_NOT_RUN',CPU_passed=True,CPU_tests=test.testsRun,input_audit=audit,history_checkpoint_bytes_and_scores_verified=True,HF_noop_GPU_logits_equal='NOT_RUN',forbidden_actual_GPU_probabilities='NOT_RUN',GPU_input_weight_cache_hook_audit='NOT_RUN',new_accuracy_tests='NOT_RUN',no_CUDA_initialized=not torch.cuda.is_initialized(),no_GPU_workers_started=True,protocol_bindings=bindings))
    save('gpu_release.json',dict(status='NO_GPU_ALLOCATED',our_GPU_workers_started=0,our_model_loads=0,new_generations=0,other_processes_killed=False,no_background_polling=True,initial_resource_snapshot='gpu_preflight.json',no_claim_about_later_gpu_availability=True))
    save('completion.json',dict(status='STOP_NO_IDLE_GPU',experiment_complete=False,CPU_design_frozen=True,questions=3000,conditions=9,new_Greedy_generations=0,pending_new_canonical_generations=plan['planned_new_canonical_generations'],pending_logical_new_labels=21000,old_baseline_records_audited=6000,cpu_tests_passed=test.testsRun,GPU_smoke_passed=False,reason='All four GPUs had active compute processes at the initial check. Per user instruction, stop without polling/preemption.',next_gate='Idle GPU plus newly frozen, validated GPU execution entry; no automatic retry scheduled.'))
    report(summary,audit,test.testsRun)
    allfiles={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_index.json'};save('artifact_index.json',dict(excludes=['artifact_index.json'],files=allfiles));print('RESOURCE STOP PACKAGED; CPU tests',test.testsRun,flush=True)

def report(summary,audit,n):
    p=read(OUT/'token_plan.json');t=list(csv.DictReader((OUT/'template_control_audit.csv').open()));counts=collections.Counter(r['decoded'] for r in t)
    assert counts=={'["Answer", "the"]':3000},counts
    lines=['# SPACE / NL视觉写入：GQA3000 CPU冻结与资源停止记录','','**状态：STOP_NO_IDLE_GPU；实验尚未完成。** 初始检查四张GPU都有活动进程，没有真正空闲卡。本轮仅完成CPU输入审计、历史答案审计和设计冻结；没有GPU Smoke、模型加载、新答案生成或后台等待。不能据此判断SPACE/NL是否提高ACC。','',
    '## 已完成的输入核验','',f'全部3000题、六任务各500题、{audit["unique_images"]}张独特图片均复用原冻结Manifest。原Prompt、原生Tokenizer IDs/Offset、S集合及完整词组逐题核对；6000个边界位置全部符合预期。样本不重采、不按输出筛选。','',
    '- SPACE：原生ID **29871**，词片`▁`，单独解码是空串；它是Tokenizer的边界位置，不等于Prompt里有一个独立ASCII空格字符。','- NL：原生ID **13**，词片`<0x0A>`，解码为换行。','- 二者固定为R的前两个位置 **581、582**，Question从583开始；不包含Image或P−1。两者Offset可同为原始换行字符的跨度[4038,4039]，但Tokenizer位置不同。未向Prompt插入SPACE/NL文字。','',
    f'N与D直接复用11fe70e中的完整3000题固定spaCy3.8.7/en_core_web_sm3.8.0解析，按原始Offset闭合到历史完整词组。原120题+12题的S/N/D映射全部严格一致；没有重新训练、替换分析器或按本轮结果修正can等词性。CPU共 **{n}项测试通过**，包含实际Hard Mask原语在真实样本布局上的Prefill和Decode连接范围检查；GPU概率及Logits检查仍未执行。','',
    '模板对照预先固定为Question之后、Instruction中的最早两个完整单Subword词法组，排除Question/B/Image/P−1，并要求均在S及N+D之外。实际全部3000题均选到 **Answer + the**，每题均新增2个Token，与B严格等预算；不加随机种子或临时换词。逐题位置见template_control_audit.csv。','',
    '## 冻结条件与待生成数量','',
    '九个标签为All-Writer、Semantic-k、S+SPACE、S+NL、S+SPACE+NL、Noun+Article、Noun+Article+SPACE+NL、S+Template-Control-2、N+D+Template-Control-2。All-Writer是答案阶段已禁止直接读图的历史基线，不是Native Full。','',
    '已逐份审计原6000个All/S检查点的Git冻结字节、图片/输入/接收集合、生成Token、评分、真实Cache协议及结果哈希。额外严格审计原132题Noun+Article记录并复用。没有复用120题Accuracy替代3000题结果。',f'七个新增条件共21000个标签；同题精确集合去重及历史复用后，仍需 **{p["planned_new_canonical_generations"]}条新的完整Greedy生成**。本轮实际新增为0。token_plan.json保留全部标签、实际接收位置、canonical ID及复用出处。','',
    '## 当前可核验的历史ACC（不是新增条件结果）','','| 子集 | 条件 | 历史可用/目标 | ACC |','|---|---|---:|---:|']
    for r in summary:
        if r['condition'] not in ['All-Writer','Semantic-k','Noun+Article']:continue
        acc=f'{r["accuracy_percent"]:.3f}%' if r['accuracy_percent']!='' else '未完成，不计算总体ACC'
        lines.append(f'| {r["cohort"]} | {r["condition"]} | {r["audited_history_available"]}/{r["cohort_n"]} | {acc} |')
    lines+=['','与Semantic240问题及图片均不重叠的子集重新核验为2640题；不是独立外部验证。原pilot120与historical12单列，所有新条件仍待生成。historical_failure_cases.csv已列出12题×9条件，特别包括001020372、09662582、10763612、0160501；待生成答案明确写NOT_GENERATED，不当作空答案或错误。','',
    '## 预先声明的分析与执行门槛','',
    '主要指标只有原GQA完整答案ACC，不生成大小写敏感Gold NLL。五组比较按用户指定顺序冻结：S+B−S；N+D+B−N+D；S+B−S+模板；N+D+B−N+D+模板；S+SPACE−S+NL。20000次Image SHA整簇配对Bootstrap，seed20261104，保持同图所有题与条件同簇；报告普通探索性95%CI。主3000题五项比较额外报告Bonferroni调整99%区间（每项alpha=.01，family alpha=.05），不将CI跨0当作非劣效。其他子集与六任务另列描述性95%CI。','',
    '一/两Worker的预分片均已冻结；同图所有题和条件在同Worker。当前无设备绑定，也没有GPU执行入口：恢复时须先检查真实资源、冻结执行源码，再通过每卡真实Smoke的HF/No-op严格torch.equal、禁止概率严格0、非目标Mask不变、独立KV、32层MLP、权重/输入Hash及Hook清理，才可运行矩阵。当前不能把历史Smoke当作本轮通过。','',
    '## 资源与停止结论','',
    '初始快照：GPU0/3分别18611/18167MiB已用并有活动进程；GPU1/2虽只使用2335/2337MiB，也分别有活动进程，且利用率70%/53%，不满足真正空闲要求。四卡均未使用，没有杀其他进程、后台监控或自动重试。此快照仅代表本轮初始检查，不声称后续占用不变。','',
    '**SPACE和NL是否提升ACC、是否超过等预算模板、是否修复部分真实失败题：目前均不能回答，新增生成数为0。** 不能用此前较高Attention Mass或已完成120题的N+D结果替代本次因果比较。Accuracy对比图、Budget–Accuracy图、六任务差值热图及新增配对CI均待完整生成后制作；当前未制造空图或补填数值。','',
    '设计、名单、边界/词性/预算审计、历史答案及本次资源停止记录保存在独立分支；main和所有历史结果未修改。本轮按资源不足的停止条件结束，不追加搜索、训练或其他实验。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
