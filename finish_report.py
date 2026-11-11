"""Package measured results after all workers exit; never launches CUDA."""
import csv
import subprocess
from collections import defaultdict
from full_common import *
from statistics_local import paired


def read_csv(name):
    with (OUT/name).open() as stream:
        return list(csv.DictReader(stream))


def markdown(rows, fields):
    lines=['| '+' | '.join(label for key,label in fields)+' |',
           '| '+' | '.join('---' for _ in fields)+' |']
    for row in rows:
        values=[]
        for key,label in fields:
            value=row[key]
            if isinstance(value,float):
                value=f'{value:.3f}'
            values.append(str(value).replace('|','/').replace('\n',' '))
        lines.append('| '+' | '.join(values)+' |')
    return lines


def main():
    p=check()
    done=json.loads((OUT/'p2_completion.json').read_text())
    gqa=json.loads((OUT/'gqa_metadata_binding_audit.json').read_text())
    assert done['status']=='COMPLETE_P2_FULL_QUALITY' and gqa['status']=='PASS_CPU_BINDING_AND_RESCORING'
    release=[]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for shard in [0,1]:
        for stem in ['release','alignment_release']:
            path=OUT/f'{stem}{shard}.json'
            if path.exists():
                r=json.loads(path.read_text())
                assert not Path('/proc/'+str(r['pid'])).exists(),('Own worker still exists',r)
                assert str(r['pid']) not in [line.split(',')[-1].strip() for line in apps.splitlines()]
                assert r['allocated']==0 and r['reserved']==0
                release.append(r)
    write(OUT/'gpu_release_audit.json',dict(status='PASS_OWN_PROCESSES_RELEASED',workers=release,
        final_compute_processes=apps,final_gpu_sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True),
        no_external_processes_terminated=True,authorized_gpu_uuids=UUIDS))
    # Historical P0 is rescored independently for background only, never pooled with P2.
    index={r['key']:r for r in json.loads((OUT/'sample_manifest.json').read_text())}
    pp=defaultdict(dict)
    for state,r in original_records(TASKS):
        assert state=='COMPLETE'
        row=index[r['key']]
        score=scored(row,r['text'],r['truncated'])
        assert score['score']==r['scoring']['score']
        pp[row['key']][r['condition']]=dict(image_sha256=row['image_sha256'],score=score['score'],
            diagnostic=score.get('diagnostic_score',score['score']),circular_group=row.get('circular_group'))
    background=[]
    for task in TASKS:
        for condition in ['Native','SVH']:
            pool=[v[condition] for k,v in pp.items() if index[k]['benchmark']==task]
            if task=='ScienceQA-IMG':
                values=[r['score'] for r in pool]
                diagnostic=[r['diagnostic'] for r in pool]
            else:
                groups=defaultdict(list)
                for r in pool:
                    groups[r['circular_group']].append(r['score'])
                values=[float(all(v)) for v in groups.values()]
                diagnostic=values
            background.append(dict(benchmark=task,condition=condition,protocol='Original P0 only; NOT same-prompt P2 comparison',
                n=len(values),accuracy=100*sum(values)/len(values),diagnostic_accuracy=100*sum(diagnostic)/len(diagnostic)))
    table(OUT/'p0_background_only.csv',background)
    scores=read_csv('p2_paired_comparisons.csv')
    results=[]
    for task in TASKS:
        result=next(r for r in scores if r['benchmark']==task and r['subset']=='All' and r['metric']=='strict' and
                    r['unit']==('question' if task=='ScienceQA-IMG' else 'Circular_group'))
        results.append(dict(task=task,n=int(result['n']),native=float(result['native_acc']),svh=float(result['svh_acc']),
            delta=float(result['delta_pp']),ci=f"[{float(result['ci_low_pp']):+.3f}, {float(result['ci_high_pp']):+.3f}]",
            harms=int(result['harms']),rescues=int(result['rescues'])))
    distribution=read_csv('gqa_type_distribution.csv')
    query=next(r for r in distribution if r['axis']=='structural' and r['type']=='query')
    subtypes=read_csv('gqa_query_subtype_analysis.csv')
    qcolor=next(r for r in subtypes if r['axis']=='query_argument' and r['type']=='color')
    qname=next(r for r in subtypes if r['axis']=='query_argument' and r['type']=='name')
    qposition=next(r for r in subtypes if r['axis']=='detailed' and r['type']=='positionQuery')
    audit=json.loads((OUT/'independent_p2_validation_audit.json').read_text())
    lines=['# Qwen3-VL：P2全量配对评测与GQA官方题型损失归因','',
        '## 首页三项结论','',
        '1. **P2全量结果（不是150题Validation）：** '+ '；'.join(f"{r['task']} Native {r['native']:.3f}%、SVH {r['svh']:.3f}%，差{r['delta']:+.3f}pp" for r in results)+'。下表给出同P2配对区间、损害和纠正。',
        f"2. **GQA总体{gqa['delta_pp']:+.4f}pp主要来自官方structural=query：** Query占{float(query['share_pct']):.2f}%题量，却占{float(query['share_all_harms_pct']):.2f}%原对→错；类内下降{abs(float(query['delta_pp'])):.2f}pp，加权贡献{float(query['contribution_pp']):+.4f}pp，占净损失{float(query['share_net_loss_pct']):.2f}%。非Query净改善5题，所以贡献超过100%并非计数错误。",
        f"3. **不能笼统归因于属性或多跳：** 官方query_argument=name的{qname['n']}题下降{abs(float(qname['delta_pp'])):.2f}pp，颜色查询{qcolor['n']}题下降{abs(float(qcolor['delta_pp'])):.2f}pp；positionQuery的{qposition['n']}题仅变化{float(qposition['delta_pp']):+.3f}pp。基于程序的“定位后问属性”没有比直接属性查询更大损害的证据。关系/类别/名称任务受损明显，但本轮不识别具体写入层的因果贡献。",'',
        '## 一、P2全量主结果','']
    lines+=markdown(results,[('task','任务'),('n','题/完整组'),('native','Native %'),('svh','SVH %'),('delta','SVH−Native pp'),('ci','图像簇95%CI pp'),('harms','原对→错'),('rescues','原错→对')])
    lines+=['','ScienceQA分数是原冻结严格评分；MMBench主分数要求同一Circular组的所有旋转记录都正确。未把旋转记录级分数当Circular分数。',
        f"共{done['terminal_predictions']:,}条规范预测：新生成{done['new_predictions']:,}条、严格复用{done['reused_validation_predictions']:,}条。ScienceQA {done['scienceqa_pairs']:,}题、MMBench {done['mmbench_paired_rows']:,}行/{done['mmbench_complete_groups']:,}完整组。OOM/失败记录{len(done['failures'])}条。",
        '### 样本与独立性','',
        '四个预冻结集合分别报告：All（全队列）、PriorValidation（上一轮Validation题号）、UnselectedImages（排除上一轮全部Discovery图片）、UntouchedImages（同时排除Discovery与Validation图片）。当前全量包含Prompt选择用过的Discovery，不是独立外部确认集。PriorValidation中为维持同图同卡重新生成的4题由运行前结构规则决定，不按分数选择。',
        '','以下为各集合的严格分数：','']
    subsets=[]
    for r in scores:
        if r['metric']=='strict' and r['unit'] in ['question','Circular_group']:
            subsets.append(dict(task=r['benchmark'],subset=r['subset'],n=int(r['n']),native=float(r['native_acc']),svh=float(r['svh_acc']),
                delta=float(r['delta_pp']),ci=f"[{float(r['ci_low_pp']):+.3f}, {float(r['ci_high_pp']):+.3f}]"))
    lines+=markdown(subsets,[('task','任务'),('subset','集合'),('n','题/组'),('native','Native %'),('svh','SVH %'),('delta','Δpp'),('ci','95%CI')])
    lines+=['','### P2高分独立CPU审计','',
        '从ScienceQA官方problems.json和MMBench原始TSV按Question ID重建允许字段；逐项核对原Choices顺序、Hint、旋转Gold。Prompt由Question/Hint/Options和固定指令构成，不读取Gold索引、lecture或solution。使用CPU Tokenizer重新解码每条保存的生成Token IDs，并独立评分；没有以标注答案替代模型调用。历史Native/SVH调用分支与模型生成源码经SHA核验。',
        f"独立重算旧Validation：SQA Native/SVH均148/150；MMBench均128/150完整组。共{audit['validation_rows']}行，{audit['identical_native_svh_token_sequences']}行Token序列相同，{audit['identical_native_svh_scores']}行得分相同；总分相同不等于复制答案。Discovery/Validation图片SHA交集为空。",'',
        'P2逐字节读取旧protocol.json的`prompts.P2_choices`，而非重写近似指令：','```text',p['suffix_exact_from_prior'],'```',
        '协议名：`ATE_HF_QWEN_P2_SHORT_ANSWER`。Qwen原生Chat Template、模型revision、BF16/Eager、原视觉预算、Greedy、EOS及max_new_tokens=128均未改；无约束解码，未人为限制为1 Token。正式预测之间只有输出指令相对旧P0发生变化。',
        '','### ScienceQA增强解析与旧P0背景','']
    diag=[]
    for r in scores:
        if r['benchmark']=='ScienceQA-IMG' and r['subset']=='All':
            diag.append(dict(metric=r['metric'],native=float(r['native_acc']),svh=float(r['svh_acc']),delta=float(r['delta_pp'])))
    lines+=markdown(diag,[('metric','评分'),('native','Native %'),('svh','SVH %'),('delta','Δpp')])
    lines+=['','以下P0只供不同Prompt协议的历史背景；不把P2-SVH对P0-Native当主配对方法差：','']
    lines+=markdown(background,[('benchmark','任务'),('condition','条件'),('n','题/组'),('accuracy','P0严格 %'),('diagnostic_accuracy','P0诊断 %')])
    lines+=['','### 输出长度、异常与时延','',
        '全量每请求的平均/P50/P95生成Token、E2E、TTFT、TPOT、EOS、截断、空答案及解析失败见`p2_output_length_and_latency.csv`；答案字母分布见`p2_letter_distribution.csv`。E2E计时为已有执行器的Vision+语言计算边界，不含图片读取/Processor，不能冒称完整服务端端到端。',
        '固定32 Token性能仅严格复用上一轮完全相同代码、P2输入绑定、模型和视觉预算的独立小型计时探针：每任务2个输入、每输入每方法3次正式重复。原始日志、来源路径与哈希绑定保留，不称为本轮全量固定32测速。它与自然EOS全量每题计时分列；不把回答从几十Token变成一个字母带来的时延下降称为SVH算子加速。',
        '','## 二、GQA12578官方题型归因','',
        f"原始Checkpoint独立重评分：Native {gqa['native']:.5f}%，SVH {gqa['svh']:.5f}%，Δ={gqa['delta_pp']:+.5f}pp；原对→错{gqa['harms']}题，原错→对{gqa['rescues']}题。共有{gqa['image_clusters']}个图片SHA簇（不是12,578张独立图片），所有本地图像文件SHA均核对，问题/Gold/Prompt及两方法输入绑定完全匹配。官方types覆盖100%，UNKNOWN=0。",'']
    group_rows=[]
    for r in distribution:
        if r['axis']=='structural':
            group_rows.append(dict(type=r['type'],n=int(r['n']),share=float(r['share_pct']),native=float(r['native_acc']),svh=float(r['svh_acc']),
                delta=float(r['delta_pp']),harms=int(r['harms']),rescues=int(r['rescues']),lost=int(r['net_lost_questions']),
                contribution=float(r['contribution_pp']),fraction=float(r['share_net_loss_pct'])))
    lines+=markdown(group_rows,[('type','官方类型'),('n','题数'),('share','题量 %'),('native','Native %'),('svh','SVH %'),('delta','Δpp'),
        ('harms','原对→错'),('rescues','原错→对'),('lost','净损失题数'),('contribution','总体贡献 pp'),('fraction','净损失贡献 %')])
    lines+=['','贡献公式为`N_type/N_all * (Acc_SVH_type − Acc_Native_type)`，各互斥完整分组的贡献严格求和回总体差距。Query不只是损害数量最多，也是在五种官方structural类别中类内降幅最大、总体净损失贡献最大。其余类型CI见`gqa_paired_type_bootstrap.csv`，不能将接近0的净变化理解成每道题都不变。',
        '','### Query内部：不是所有Query同样脆弱','']
    selected=[r for r in subtypes if r['axis']=='semantic' or (r['axis']=='query_argument' and r['type'] in ['name','color','material','hposition','size'])]
    lines+=markdown([dict(axis=r['axis'],type=r['type'],n=int(r['n']),native=float(r['native_acc']),svh=float(r['svh_acc']),delta=float(r['delta_pp']),
        ci=f"[{float(r['ci_low_pp']):+.2f}, {float(r['ci_high_pp']):+.2f}]",harms=int(r['harms']),rescues=int(r['rescues'])) for r in selected],
        [('axis','官方字段'),('type','取值'),('n','题数'),('native','Native %'),('svh','SVH %'),('delta','Δpp'),('ci','95%CI pp'),('harms','损害'),('rescues','纠正')])
    lines+=['','`semantic=rel`常见程序是`relate → query:name`：它可能要求说出相关物体名称，不等价于只预测left/right等关系词。官方字段里没有统一名为QueryObject的类别，不凭英文问句制造这种“官方标签”。`query_argument`来自真实官方功能程序，单独列示。',
        f"名称查询净损失{int(qname['harms'])-int(qname['rescues'])}题；颜色查询净损失{int(qcolor['harms'])-int(qcolor['rescues'])}题。位置查询是重要反例：Native {float(qposition['native_acc']):.2f}%、SVH {float(qposition['svh_acc']):.2f}%。activity等低下降子类也保留，但低Native基线可能有地板效应，不能宣称路径无功能。",
        '','### 多步定位假说与反证','',
        '程序中在query前出现relate/filter定义为“定位”，这是明确标为`HEURISTIC_DIAGNOSTIC`的操作性诊断，不是新增官方题型。属性query中：定位后查询925题下降9.84pp，直接select后查询841题下降10.94pp；差异为+1.10pp、图片簇95%CI[−2.57,+4.79]。因此这轮不支持“先定位后问属性更容易受损”。',
        '关系跳数≥2的451题下降18.40pp，0跳下降16.21pp，1跳下降16.78pp；程序深度≥5仅71题下降21.13pp。这些是探索性关联，受任务组成与难度混杂，不能证明多跳本身导致损失。原始视觉Token数、真实问题Token长度、单/多词答案、Yes/No/open以及计数比较程序标签均另表给出，没有据结果改分组。',
        '','### 历史1000/3000题：逐ID绑定后才谈组成','',
        '历史三个分支均从固定Commit读取，样本逐Question ID/Image ID/Question/Gold绑定原GQA val_balanced Parquet，相关本地图片SHA逐文件核对。当前testdev与历史1000/3000队列的Question ID+Image SHA严格重合数均为0，图片SHA重合也为0；因此不把旧答案当当前同题基线，也不能直接把两个总体差值作因果分解。',
        '历史1000题有525题官方query（52.5%）；历史恢复L22–L35写入的+9.2pp，分解后query子集恢复+17.52pp，净改善92题，其余structural类别合计净0。这是该历史队列内部的已测因果比较，不能直接当本轮恢复效果。',
        '历史3000题只有500题query（16.67%），且500题全部都是官方`types.detailed=positionQuery`，不是颜色/名称查询。其余为1500 choose、500 compare、500 logical。其历史QueryAttr的0.4pp写入损失与本轮positionQuery约0.63pp的低降幅在描述上相容；不能据旧“六类”称总体Query已低损验证。组成差异有真实元数据证据，但Prompt、评分和干预背景也不同，不能说组成已经解释全部跨队列差距。',
        '','### 真实损害、纠正与反例','',
        '`gqa_error_cases.csv`按预声明SHA规则，从五个structural类别的损害/纠正单元各取最多4题；另外展示n≥30且净变化最小的3个query细类的哈希案例，并明确这是描述性反例选择。完整12578题答案、官方标签及程序见`gqa_per_question_types.csv`，不只留成功故事。','']
    cases=read_csv('gqa_error_cases.csv')
    chosen=[]
    for cell in ['query|harm','query|rescue','choose|harm','small_net_drop_subtype|positionQuery']:
        found=next((r for r in cases if r['selection_cell']==cell),None)
        if found:
            chosen.append(dict(qid=found['question_id'],cell=cell,question=found['question'],gold=found['gold'],native=found['native_text'],svh=found['svh_text']))
    lines+=markdown(chosen,[('qid','ID'),('cell','选择单元'),('question','问题'),('gold','Gold'),('native','Native'),('svh','SVH')])
    lines+=['','Exact Match也会惩罚词形、修饰词和近义表达：例如固定哈希损害案例201663260中Gold/Native为clean，SVH为very clean。本轮没有引入语义Judge或更改规范化，因此并非每一题评分损失都能直接认定为缺乏视觉证据；但颜色yellow→white等实际内容改变的损害也同时存在。官方得分和原文都保留，不能用少数格式或近义案例抵消全部损失。']
    lines+=['','## 三、有效性、资源和复现','',
        '所有质量差使用题目等权、Image SHA整簇配对Bootstrap 20,000次，固定seed=20261111；MMBench按完整Circular组先求分，再以共享Image SHA整簇重采样。子类、程序诊断及多项比较均为探索性，没有把未校正CI当确认性显著性，也不把CI跨0称无损。',
        '历史严格Dense/Physical数值状态仍为**FAIL_RETAINED**；本轮双卡Smoke仅确认Native/No-op严格torch.equal、SVH重复前向一致、冻结结构/位置/MLP/独立Cache/Hook/输入/权重审计正常。本轮质量成功不能补判严格数值等价。方案B为ATE统一HF协议，不是作者官方协议复现；方案A继续BLOCKED。',
        '全量运行与历史记录全部只读继承；只新增独立工作树文件，未改主实验Prompt/Checkpoint/SVH算法/共享环境，未推送GitHub。为了读取25MB历史Parquet，复用已有video-model环境中的PyArrow子进程，未安装或修改Qwen环境；初次CPU缺少PyArrow的日志原样保留。',
        'GPU1/GPU3各自常驻一份模型，按图片分片，同图规范记录Native/SVH均来自同一物理卡。4道混合复用/新增图簇中的Validation题预先按UUID规则补跑8次，旧记录保留，规范结果不重复计数。GPU0/2外部任务不触碰。任务进程已退出，GPU释放审计见`gpu_release_audit.json`。',
        '所有模型、图片和旧大数组均共享只读，没有复制。本任务压缩检查点和报告使用独立300MiB输出上限、512MiB磁盘最低余量，不改主实验原2GiB保护。',
        '','复现命令（从本工作树执行；正式已有结果应先执行只读审计，不重复生成）：','```bash',
        'CUDA_VISIBLE_DEVICES=\"\" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest -v test_full test_analysis',
        'CUDA_VISIBLE_DEVICES=\"\" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_gqa.py',
        'CUDA_VISIBLE_DEVICES=\"\" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_full.py',
        'CUDA_VISIBLE_DEVICES=\"\" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python finish_report.py','```','',
        '推荐下一步仅作提案：以当前官方query细分为重点，预冻结独立验证集比较保持答案禁图时的写入截止深度；本轮没有执行新写入层、P−1回看或任何算法搜索。P2可作为新的短答协议继续验证，不能据它解决GQA语义交接问题。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    write(OUT/'validation.json',dict(status='COMPLETE_BOTH_TASKS_INTEGRITY',p2=done,gqa=gqa,
        original_algorithm_sources_unchanged=bool(check()),strict_physical_numeric_status='FAIL_RETAINED',
        gpu_workers_released=True,main_not_modified=True,github_not_pushed=True,quality_protocol='ATE_HF_QWEN_P2_SHORT_ANSWER',
        analysis_sources={str(f):sha(f) for f in ROOT.glob('*.py')}))
    write(OUT/'artifact_sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*'))
        if f.is_file() and '.git' not in f.parts and f.name not in ['artifact_sha256.json'] and '__pycache__' not in f.parts})
    print('REPORT_COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':
    main()
