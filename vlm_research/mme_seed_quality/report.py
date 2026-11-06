"""Reproducible figures and Chinese report from complete measured matrices."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .analyze import write_csv

def rows(name):
    p=OUT/name
    return list(csv.DictReader(p.open())) if p.exists() and p.stat().st_size else []

def table(data,columns):
    if not data:return '未完成，未填入推测得分。\n'
    labels=[v for k,v in columns]
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(labels))+' |']
    for r in data:
        values=[]
        for key,label in columns:
            v=r.get(key,'')
            if key not in ('model','task','metric','unit','benchmark','status'):
                try:
                    if key in ('n','images','expected_questions','complete_paired_questions','missing_checkpoints','harm','recovered','min','max','above_2048','questions','eos','truncated','empty','repeated_3gram','unparsed'):
                        v=str(int(float(v)))
                    else:v=f'{float(v):.4f}'
                except (TypeError,ValueError):pass
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)+'\n'

def figures(summary,mme,seed):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,kind in enumerate(('llava','qwen')):
        label='LLaVA-1.5-7B' if kind=='llava' else 'Qwen3-VL-8B (2048 image tokens)'
        a=axes[i,0];vals=[r for r in summary if r['model']==kind and r['benchmark']=='MME']
        categories=['Perception','Cognition','Total'];xx=np.arange(3)
        if vals:
            lookup={r['metric']:r for r in vals}
            a.bar(xx-.18,[float(lookup[c]['native']) for c in categories],.36,label='Native')
            a.bar(xx+.18,[float(lookup[c]['c3']) for c in categories],.36,label='Physical C3-V2')
            a.set_xticks(xx,categories);a.set_ylabel('MME score points');a.legend()
        else:
            a.text(.5,.5,'STOPPED: Native vision encoder OOM\nNo quality score' if kind=='qwen' and (OUT/'qwen_stop_summary.json').exists() else 'NOT RUN / INCOMPLETE',ha='center',va='center',transform=a.transAxes)
            a.set_xticks([]);a.set_yticks([])
        a.set_title(label+'\nMME');a.set_ylim(bottom=0)
        a=axes[i,1];vals=[r for r in summary if r['model']==kind and r['benchmark']=='SEED']
        if vals:
            r=vals[0];bars=a.bar([0,1],[float(r['native']),float(r['c3'])],color=['#1f77b4','#ff7f0e'])
            a.bar_label(bars,fmt='%.2f%%',padding=3);a.set_xticks([0,1],['Native','Physical C3-V2'])
            a.text(.5,.95,f"Δ = {float(r['delta']):+.2f} pp\n95% CI [{float(r['ci_low']):+.2f}, {float(r['ci_high']):+.2f}]",ha='center',va='top',transform=a.transAxes)
        else:
            a.text(.5,.5,'STOPPED: Native vision encoder OOM\nNo quality score' if kind=='qwen' and (OUT/'qwen_stop_summary.json').exists() else 'NOT RUN / INCOMPLETE',ha='center',va='center',transform=a.transAxes)
            a.set_xticks([]);a.set_yticks([])
        a.set_title(label+'\nSEED-Bench-1 Image');a.set_ylabel('Accuracy (%)');a.set_ylim(0,100)
    fig.savefig(folder/'quality_overview.png',dpi=240);fig.savefig(folder/'quality_overview.pdf');plt.close(fig)
    for kind in ('llava','qwen'):
        for bench,data,point,low,high,unit in [('mme',mme,'delta_score_points','score_ci_low','score_ci_high','MME score points'),
            ('seed',seed,'delta_pp','ci_low','ci_high','Accuracy percentage points')]:
            data=[r for r in data if r['model']==kind and r['task']!='Overall']
            if not data:continue
            data.sort(key=lambda r:float(r[point]));fig,ax=plt.subplots(figsize=(9,max(4,len(data)*.38)),layout='constrained')
            d=np.array([float(r[point]) for r in data]);lo=np.array([float(r[low]) for r in data]);hi=np.array([float(r[high]) for r in data])
            y=np.arange(len(data));ax.hlines(y,lo,hi,color='#1f77b4');ax.scatter(d,y,color='#1f77b4')
            ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(y,[r['task'] for r in data]);ax.set_xlabel('Physical C3-V2 − Native ('+unit+')')
            ax.set_title(kind.upper()+' / '+bench.upper()+' — image-cluster paired 95% CI (exploratory)')
            fig.savefig(folder/f'{kind}_{bench}_task_deltas.png',dpi=240);fig.savefig(folder/f'{kind}_{bench}_task_deltas.pdf');plt.close(fig)

def main():
    summary=rows('quality_summary.csv');mme=rows('mme_task_scores.csv');seed=rows('seed_task_accuracy.csv');coverage=rows('coverage.csv')
    figures(summary,mme,seed)
    smoke=[];numeric=[]
    for kind in ('llava','qwen'):
        for rank in (0,1):
            folder=OUT/f'smoke/{kind}_rank{rank}'
            for f in folder.glob('*.json'):
                if f.name=='completion.json' or f.name.endswith('_input.json'):continue
                r=read(f);smoke.append(dict(model=kind,rank=rank,key=r['key'],**r.get('memory_audit',{})))
                pp=r['physical']['token_ids'];dd=r['dense']['token_ids']
                for d in r['dense_physical_observed_diff']:
                    matched=pp[:d['step']]==dd[:d['step']]
                    numeric.append(dict(model=kind,rank=rank,key=r['key'],same_conditioning_prefix=matched,
                        note='numeric comparison' if matched else 'different generated prefix; not a pure numeric error',**d))
    write_csv(OUT/'smoke_memory_audit.csv',smoke);write_csv(OUT/'dense_physical_numerical_audit.csv',numeric)
    refs=[dict(model='LLaVA-1.5-7B',benchmark='MME',published_value=1510.7,
        scope='Published MME Perception reference; not our 2800-point total; different evaluation stack possible',
        source='https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md'),
        dict(model='LLaVA-1.5-7B',benchmark='SEED',published_value=58.6,
        scope='Published table says SEED, not explicitly this image-only mean-NLL protocol; not interchangeable with local Native',
        source='https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md'),
        dict(model='Qwen3-VL-8B-Instruct',benchmark='MME / SEED',published_value='',
        scope='No directly interchangeable 2048-budget score verified; not filled',source='https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct')]
    write_csv(OUT/'published_references.csv',refs)
    status='COMPLETE' if len(coverage)==4 and all(x['status']=='COMPLETE' for x in coverage) else 'PARTIAL'
    text=f'''# LLaVA / Qwen3-VL 原版物理 C3-V2：MME 与 SEED-Bench-1 完整质量对照

状态：**{status}**。只报告已完成同题矩阵；未运行或不完整的条件不填入预测分数。本轮仅测质量，没有运行E2E、Kernel搜索、FLOPs优化、训练或额外层搜索。

## 主要实测结果

差值统一为 Physical C3-V2 − Native。MME单位为官方 **Score Points**；SEED终局指标是 **Accuracy**，差值单位为百分点。

'''+table(summary,[('model','模型'),('benchmark','Benchmark'),('metric','指标'),('native','本地Native'),('c3','本地Physical C3'),('delta','C3−Native'),('ci_low','95%CI下界'),('ci_high','95%CI上界'),('unit','差值单位')])+'''
![Native versus Physical C3-V2 quality](figures/quality_overview.png)

图中缺失条件明确标为未测，不将其画成零分。配对区间为探索性95%CI；MME与SEED的单位分别标注。

## 数据和评分

MME经典版全量2,374问、1,187个图片路径，14子任务。两道同图问题均保留。按官方工具的Yes/No解析，逐任务重算Accuracy与Accuracy+（两问均正确的图片比例），任务分数为二者百分数之和；Perception满分2000、Cognition满分800、合计满分2800。原始下载审计、逐文件图片SHA、原生标注对齐及官方工具输出均保留。

SEED-Bench-1 Image全量14,233条原始图像题、5,624个图片路径。官方Question ID有1个重复，因此保留全部记录，以原始记录索引作为唯一键；没有将其去重为14,232题。每个选项按完整候选文本的平均条件负LogP排序，选最小者，再计算选中选项与Gold一致的Accuracy；没有自由生成字母解析，也没有把NLL冒充ACC。保存每个候选的Token、逐Token LogP和聚合分数。

候选首Token在原始P−1预测，其余候选Token全部经过真实增量Decode，因此C3的答案禁图及Attention物理跳算不会被Teacher Forcing绕过。各候选复制真实独立Prompt Cache，Smoke中与四次独立冷请求的全部有效Logits严格torch.equal。

采用官方mean-NLL答案排序机制，并保留本地冻结模型的HF原生聊天模板和历史Processor。LLaVA候选包含原生助手空格及`</s>`；Qwen遵循官方Qwen接口对候选内容取均值，不额外计EOS。官方参考适配器中的聊天模板、图像预处理与本地HF实现可能不同，因此这是同题、本地Native配对对照，不能把公开表格分数直接填入本地列。

具体而言，官方LLaVA适配器使用`llava_v1`会话模板及其视觉预处理管线，本轮按预冻结协议保持HF Native聊天模板与Processor。因此本地SEED Native分数不能声称复现了官方发布表的58.6，也不能把本地分数与该参考分数相减当作C3损失；C3差值只相对于同题、同输入的本地Native。没有另加模板或预处理条件来解释绝对分数差。

## 两模型的固定物理实现

LLaVA固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/Eager、Transformers4.49.0；R在L0–L19读图，L20前移除Image Hidden，P−1/Decode禁图，L2–L5答案Attention真实跳算。MLP、Norm、Residual、原始RoPE位置和真实压缩KV均保留。没有用Dense输出置零代替主要Physical条件。

Qwen固定revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，BF16/Eager、Transformers4.57.6；R在L0–L21读图，L22前移除Image Hidden，P−1/Decode禁图，L1–L6答案Attention真实跳算，保留mRoPE与DeepStack。复用Physical-Original，不加载cuBLASLt、Zero-Key或后续Repaired路径。**历史Physical-Original尚未通过严格BF16数值等价验收，不能宣称与Dense F严格等价；本轮Qwen因资源门禁停止，没有新的质量分数或数值等价通过结论。** 已完成的LLaVA Smoke数值观察单独保存，前缀已分叉后的Logits差不解释为纯数值误差。

## Qwen统一2,048视觉Token协议

原始最大像素预算可能产生约16,000个视觉Token，超过24GiB RTX3090上Native Eager可行范围。用户明确授权仅将Qwen官方ImageProcessor.size.longest_edge设为2048×32×32，保留其他参数。LLaVA输入协议完全不变。

Qwen主结果若完成，Native与C3均在这一统一2,048视觉Token上限下重新实测，**不是默认分辨率的官方Native分数**，也未用历史无限制预算Native计算差值。逐图真实Processor输出、修改前后Token分布、grid、像素Tensor SHA及完整逐题Prompt Token边界见qwen_*审计文件。没有截断Image Token、删除大图、按结果筛题或OOM后自动下调预算。

## 分任务结果

### MME

'''+table(mme,[('model','模型'),('task','任务'),('native_accuracy','Native ACC%'),('c3_accuracy','C3 ACC%'),('native_accuracy_plus','Native ACC+%'),('c3_accuracy_plus','C3 ACC+%'),('native_score','Native分数'),('c3_score','C3分数'),('delta_score_points','差值Score Points'),('score_ci_low','CI下界'),('score_ci_high','CI上界')])+'''
### SEED-Bench-1 Image

'''+table(seed,[('model','模型'),('task','任务'),('n','题数'),('native_accuracy','Native ACC%'),('c3_accuracy','C3 ACC%'),('delta_pp','差值pp'),('ci_low','CI下界'),('ci_high','CI上界'),('harm','原对变错'),('recovered','原错变对')])+'''
## 统计与局限

每个Benchmark、每个模型分别以Image SHA256整簇进行20,000次配对Bootstrap，seed=20261106。SEED每次重算题目加权Accuracy；MME按图片所在子任务集合分层抽取整个SHA簇，保持固定Benchmark任务构成，同一图片跨任务时仍不拆开，逐次重算14项Accuracy、Accuracy+及Perception/Cognition/总分。区间均为探索性、未经多重校正的95%CI，不作“无损”或非劣效证明。

所有下降、恢复和答案改变均保留在逐题CSV及Checkpoint；异常输出不剔除。官方Published Reference独立列于published_references.csv，不参与本地配对统计。

本轮仅对Native和原版Physical C3进行全量质量评估，Dense F只用于已完成Smoke的数值观察。因此Native−Physical的质量差同时包含算法约束与原版物理实现的影响，不能据此单独量化数值误差造成的质量损失，也不能从任务分数直接识别某个层或组件的独立因果贡献。

## 完成覆盖

'''+table(coverage,[('model','模型'),('benchmark','Benchmark'),('expected_questions','应含题数'),('complete_paired_questions','完整配对题数'),('missing_checkpoints','缺失Checkpoint'),('status','状态')])
    text+='\n## 输出异常\n\n'+table(rows('output_anomalies.csv'),[('model','模型'),('condition','条件'),('questions','题数'),('eos','EOS结束'),('truncated','截断'),('empty','空答案'),('repeated_3gram','重复3gram'),('unparsed','非Yes/No解析')])
    text+='\nSEED使用完整候选评分，不适用自由生成EOS或截断计数；全部候选NLL在评分复核时检查有限，异常记录不按得分筛除。\n'
    text+='\n## 实测结论\n\n'
    if (OUT/'qwen_processor_validation.json').exists():
        qv=read(OUT/'qwen_processor_validation.json');dist=[]
        for bench,versions in qv['statistics'].items():
            for version,stat in versions.items():dist.append(dict(benchmark=bench,task=version,**stat))
        text+='### Qwen逐图实际视觉Token分布\n\n'+table(dist,[('benchmark','Benchmark'),('task','Processor'),('images','图片数'),('min','最小'),('p50','P50'),('p90','P90'),('p95','P95'),('p99','P99'),('max','最大'),('above_2048','超过2048张数')])+'\n'
    for kind in ('llava','qwen'):
        for bench in ('MME','SEED'):
            target=[r for r in summary if r['model']==kind and r['benchmark']==bench and (r['metric']=='Total' if bench=='MME' else True)]
            if not target:text+=f'{kind} / {bench}尚未完成，不报告质量损失或集中退化结论。\n\n';continue
            r=target[0];text+=f"{kind} / {bench}：Native={float(r['native']):.4f}，Physical C3={float(r['c3']):.4f}，C3−Native={float(r['delta']):+.4f} {r['unit']}，95%CI [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]；原对变错{r['harm']}题、原错变对{r['recovered']}题。\n\n"
        mm=[r for r in mme if r['model']==kind];ss=[r for r in seed if r['model']==kind and r['task']!='Overall']
        if mm:
            worst=sorted(mm,key=lambda r:float(r['delta_score_points']))[:4]
            text+='MME任务分差最低的四项：'+ '；'.join(f"{r['task']} {float(r['delta_score_points']):+.2f} Score Points" for r in worst)+'。这描述分布，不等于已经隔离OCR或推理机制的因果来源。\n\n'
            focus=[r for r in mm if r['task'] in ('OCR','text_translation','position','count')]
            text+='预先关心的MME四项：'+'；'.join(f"{r['task']} {float(r['delta_score_points']):+.2f}" for r in focus)
            text+=f" Score Points；四项合计{sum(float(r['delta_score_points']) for r in focus):+.2f}，全14项合计{sum(float(r['delta_score_points']) for r in mm):+.2f}。这里保留正负号，没有把其他任务的恢复删除后再宣称损失集中。\n\n"
        if ss:
            worst=sorted(ss,key=lambda r:float(r['delta_pp']))[:4]
            text+='SEED任务ACC分差最低的四项：'+ '；'.join(f"{r['task']} {float(r['delta_pp']):+.2f} pp" for r in worst)+'。Spatial Relation、Visual Reasoning与Text Understanding的全部结果均保留，未按结论选择类别。\n\n'
            focus=[r for r in ss if r['task'] in ('Text Understanding','Spatial Relation','Visual Reasoning')]
            text+='预先关心的SEED三项：'+'；'.join(f"{r['task']} {float(r['delta_pp']):+.2f} pp" for r in focus)
            groupnet=sum(int(r['recovered'])-int(r['harm']) for r in focus)
            overall=next(r for r in seed if r['model']==kind and r['task']=='Overall')
            text+=f"。三类合计正确题净变化{groupnet:+d}，全体正确题净变化{int(overall['recovered'])-int(overall['harm']):+d}；不能直接相加不同任务的ACC百分点。\n\n"
    if any(r['model']=='llava' and r['benchmark']=='SEED' for r in summary):
        text+='本轮LLaVA没有呈现统一的“OCR、文字理解或空间关系集中退化”模式：相关任务点估计包括持平与上升，实例识别、场景、属性、计数及部分推理任务也出现损害。MME的code_reasoning下降12.50 Score Points，不能将全部推理任务概括为受保护。SEED Text Understanding仅85题，其差值95%CI为[−6.67, +6.58] pp，不能把零点估计解释为无损。这里只描述当前任务划分下的质量分布，不将其归因于单一视觉通路；Qwen没有测得质量分数，无法判断是否存在同样模式。\n\n'
    manifest=read(OUT/'sample_manifest.json')
    counts={bench:len({s['image_sha256'] for s in manifest if s['benchmark']==bench}) for bench in ('MME','SEED')}
    text+=f"\n实际内容SHA簇：MME {counts['MME']}，SEED {counts['SEED']}。图片路径数与内容簇数不同；全部原始记录仍保留，统计按内容SHA整簇配对。\n"
    stops=[]
    for path in OUT.glob('STOP_*.json'):
        e=read(path);stops.append(dict(file=path.name,kind=e.get('kind'),traceback=e.get('traceback'),
            actual_input_shapes=e.get('actual_input_shapes'),last_actual_attention_shape=e.get('last_actual_attention_shape'),
            peak_allocated_bytes=e.get('peak_allocated_bytes'),peak_reserved_bytes=e.get('peak_reserved_bytes')))
    if stops:
        text+='\n## 停止记录\n\n'
        text+='Qwen两卡均未完成Native Smoke，也未执行新的Physical C3、HF/No-op全路径或Dense F对照；这些门禁均没有标记通过。2,048个语言侧视觉Token对应最多8,192个ViT内部Patch。原生视觉Attention在这8,192个位置上构建矩阵，FP32 Softmax申请约4GiB时显存不足。此处没有把多层Attention矩阵保留在GPU，且每卡只加载一份模型。完整两卡现场见qwen_stop_summary.json及environment/qwen_rank*_smoke_failure.json。\n\n'
        for e in stops:
            text+=f"{e['kind']}路径触发停止；完整现场见{e['file']}。未降低2,048预算、未切换Attention Backend，也未补填未完成条件分数。\n\n"
            text+='OOM错误中的GPU0是Worker内的逻辑cuda:0；实际物理设备由环境记录绑定到GPU2/3的UUID。此次OOM发生在Native的视觉编码器，早于语言Decoder与C3干预，因此不能据此评价C3的准确率或新预算下的Dense等价性。\n\n'
            text+='```json\n'+json.dumps({k:v for k,v in e.items() if k!='traceback'},ensure_ascii=False,indent=2)+'\n```\n'
    text+='\n## 复现与文件\n\n源码与输入冻结见protocol.json、source_manifest.json、model_manifests.json；逐题结果见per_question.csv、checkpoints/；官方MME复算见official_mme/；图及其直接输入CSV均在结果目录。SEED完整候选均评分，不以32 Token生成上限截断候选；候选Cache容量取本题最长候选Token数+1，此值及候选全Token保留在执行源码和Checkpoint。GPU释放记录见gpu_release_audit.json。\n\n原始Smoke Logits数组仅保存在服务器资源目录，GitHub保存路径与SHA索引smoke_raw_logits_index.json，不包含数组本身。官方MME工具额外使用隔离CPU目录中的固定版本评分依赖，版本见official_scoring_dependencies.json；没有修改运行中的模型环境。\n'
    (OUT/'report_zh.md').write_text(text,encoding='utf-8')
    print('REPORT AND FIGURES',status,flush=True)

if __name__=='__main__':main()
