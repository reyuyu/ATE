"""Read-only case attribution: distinguish observed omissions from untested causes."""
import collections,time
import numpy as np
from .common import *
from .prepare import tags
from paper_rq1.scorers import gqa_prediction

NOTES={
'001020372':'关系词 left、of、选择连接词 or 和两个颜色选项都已选中；不能把这道错题解释成漏选这些词。遗漏的是 Is、三个 the 和问号，且整个指令/模板均未读图。是否由功能词或边界接收位置造成，尚无补词验证。',
'14236062':'or 与 colored 是词法核心候选，却因整词后缀对照的可行预算缩减而未进入 Semantic-k。cream 已选中但 Gold 是 cream colored；可能同时存在选项连接与答案表面形式问题，尚不能确定哪个遗漏有因果作用。',
'02760157':'penguins、or、panda、bears 全部已选。Semantic 输出 Panda，官方 Gold 为 panda bears；冻结评分器严格判错，但这不等于已经识别成错误动物。此例必须单独标注答案表面形式不匹配，不能全归因于视觉接收失败。',
'09662582':'train 是问题中的视觉类别选项，却因可行预算缩减而被删除；this、truck、or 已选。该输入规则产生了一个很具体的漏选候选，但补回 train 是否能纠正 Truck 尚未测试。',
'08163732':'left、right、of、boy、wearing、coat 等关系及对象锚点均已选。遗漏主要为系动词、冠词与问号；左右判断仍出错说明有关键内容词不等于视觉定位必然正确。',
'15260602':'left/right、两个方向连接及 train car 均已选。不能预设漏了方向词；功能词和模板接收位置是待检验候选，也可能存在多个位置的共同作用。',
'01445490':'What、common、shield、and、bat 已选，输出却从 Gold Color 变为 Material。比较/共性任务的目标属性发生变化，但不能仅凭答案推断视觉表示缺失；is、冠词、问号及模板是否有作用仍未验证。',
'12951806':'What、container、plate、in、common、and 均已选。do、have 被 STOP 规则排除，可能影响提问形式的接收，但 Material 与 Color 的变化并不能单独证明这种解释。',
'0160501':'or、door、window、open 已选，而 Do 与 are 未选。输出 Door 而不是 Yes，体现了是非任务的答案类型/格式偏移；Do 是可检验的接收位置，但这也可能是多位置或指令共同作用。',
'03567478':'状态词 on 被可行预算缩减删除，any、or、TVs、laptops 已选。不能笼统说数量词或逻辑词未选；更具体的假说是电源状态接收位置丢失，尚待单词补回验证。',
'0481307':'On、which、side、small、container 全部已选；遗漏 is、the 和问号。左右答案变化不支持“疑问词一定被漏选”的假设；可优先检验问题边界与模板，但本轮未运行。',
'10763612':'自然 k 被整词后缀对照的可行性约束降到 0，Semantic-k 与 No-Writer 写入集合完全相同。On/which/side 三个核心候选都被预算约束删除；can 同时因固定 STOP 将模态动词与名词罐子混同而未列为核心。这里存在明确的规则设计风险，但不是已验证的补词修复。',
'001025435':'mud、dry、or、wet 已选；冠词、Is、问号及模板未选，Semantic 反而将 Dry 纠正为 Wet。遗漏功能词/模板并不意味着每题都会受损。',
'03678636':'否定词 not 实际已选，连同 white、drawer、chair 和 or；Semantic 将 Chair 纠正为 Drawer。这是防止“否定一定被漏选”这一预设解释的重要反例。',
'03539455':'right、left、of、man、bag 及 or 均已选；冠词与 Is 未选。Semantic 从 Right 纠正到 Left，与两道同类关系损害案例共同表明结果具有题目依赖性。',
'16958949':'On、which、side、of、man 已选，picture 因 STOP 排除。Semantic 从 Right 纠正到 Left；提到图片的词不必在每题都直接读图，不能由单个失败推导全部模板/功能词均必要。'}

def main():
    cases=selected();assert len(cases)==16;protocol=read(OUT/'protocol.json');assert protocol['gpu'] is None,'This entry point is read-only/no-idle-GPU delivery'
    assert not torch.cuda.is_initialized();outputs=[];coverage=[];summaries=[];seen=set();refs={};rawrefs={}
    for c in cases:
        s=c['sample'];qid=c['question_id'];spec=c['spec'];sel=set(spec['writers']['Semantic-k'])
        assert sha256(OUT/c['gallery_image'])==s['image_sha256'];c['interpretation']=NOTES[qid]
        for name,r in c['rows'].items():
            path=HIST/r['source_checkpoint'];row=read(path);g=row['generation'];assert sha256(path)==gitsha(path)
            assert row['result_sha256']==digest(dict(generation=g,teacher_forced=row['teacher_forced']))
            assert g['text']==r['text'] and g['token_ids']==json.loads(r['token_ids']) and g['correct']==int(r['correct'])==int(gqa_prediction(g['text'])==s['answer'])
            assert sha256(row['raw']['path'])==row['raw']['sha256']==r['raw_logits_sha256']
            with np.load(row['raw']['path']) as z:
                assert int(z['first_logits'].argmax())==g['token_ids'][0]
                for label in ['greedy_audit','gold_audit']:
                    assert (z[label][:,0]==0).all() and (z[label][:,4:]==1).all()
            refs[str(path)]=sha256(path);rawrefs[row['raw']['path']]=row['raw']['sha256']
            outputs.append(dict(question_id=qid,kind=c['kind'],task=s['task'],condition=name,answer=g['text'],gold=s['answer'],correct=g['correct'],
                origin='historical_reuse',source_checkpoint=r['source_checkpoint'],source_sha256=sha256(path),token_ids=r['token_ids']))
        for category in RULES['tag_priority']+['question_end','template','comparison_common']:
            gg=[g for g in spec['groups'] if (category in tags(g) and g['question']) or category=='question_end' and spec['question_positions'][-1] in g['positions'] or category=='template' and not g['question'] or category=='comparison_common' and any(w.lower()=='common' for w in g['words'])]
            selected_g=[g for g in gg if set(g['positions'])<=sel];missing=[g for g in gg if not set(g['positions'])&sel]
            coverage.append(dict(question_id=qid,kind=c['kind'],task=s['task'],category=category,present_groups=len(gg),selected_groups=len(selected_g),omitted_groups=len(missing),
                selected_text=' | '.join(g['decoded'] for g in selected_g),omitted_text=' | '.join(g['decoded'] for g in missing),omitted_positions=json.dumps(flatten(missing))))
        summaries.append(dict(question_id=qid,kind=c['kind'],task=s['task'],confirmatory=c['confirmatory'],k=len(sel),R=len(spec['relay_positions']),
            budget_reduction=spec['natural']['budget_reduction_for_whole_word_suffix'],core_omitted_text=' | '.join(g['decoded'] for g in c['semantic_core_dropped_groups']),
            core_omitted_count=len(c['semantic_core_dropped_groups']),k_zero=len(sel)==0,interpretation=NOTES[qid]))
    atomic_json(OUT/'case_annotations.json',dict(cases=cases,posthoc_readonly=True,causal_tests_run=False))
    csvwrite(OUT/'historical_case_outputs.csv',outputs);csvwrite(OUT/'token_feature_coverage.csv',coverage);csvwrite(OUT/'readonly_case_attribution.csv',summaries)
    plan=read(OUT/'candidate_plan.json')['cases'];pending=[]
    for qid,p in plan.items():
        for v in p['singles']:
            options=[(v['id'],v['writers'],v['added'],[]),(v['id']+'_matched_control',v['control_writers'],v['control_added'],[]),(v['id']+'_equal_budget_replace',v['replacement_writers'],v['added'],v['remove'])]
            for name,writers,added,removed in options:
                pending.append(dict(question_id=qid,condition=name,status='NOT_RUN_NO_IDLE_GPU',feasible=writers is not None,conditional='single must succeed first' if name.endswith('replace') else '',
                    added=json.dumps(added),removed=json.dumps(removed),writers=json.dumps(writers),answer='',correct='',no_generated_answer=True))
        for v in p['grouped']:
            pending.append(dict(question_id=qid,condition=v['id'],status='NOT_RUN_NO_IDLE_GPU',feasible=True,conditional='',added=json.dumps(v['added']),removed='[]',writers=json.dumps(v['writers']),answer='',correct='',no_generated_answer=True))
    csvwrite(OUT/'supplementation_results.csv',pending)
    harm=[r for r in summaries if r['kind']=='harm']
    atomic_json(OUT/'readonly_summary.json',dict(selected_harm=12,selected_rescue=4,harm_confirmatory=sum(c['confirmatory'] for c in cases if c['kind']=='harm'),rescue_confirmatory=4,
        harm_core_omission_cases=sum(r['core_omitted_count']>0 for r in harm),harm_core_omitted_groups=sum(r['core_omitted_count'] for r in harm),
        harm_all_core_preserved_cases=sum(r['core_omitted_count']==0 for r in harm),harm_k_zero=sum(r['k_zero'] for r in harm),
        harm_question_end_omitted=sum(r['kind']=='harm' and r['category']=='question_end' and r['omitted_groups']>0 for r in coverage),
        harm_template_omitted_cases=sum(r['kind']=='harm' and r['category']=='template' and r['omitted_groups']>0 for r in coverage),
        clear_answer_surface_mismatch_cases=['02760157'],question_type_output_mismatch_cases=['0160501'],no_population_extrapolation=True))
    atomic_json(OUT/'causal_validation_status.json',dict(status='NOT_RUN_NO_IDLE_GPU',gpu_model_loaded=False,new_generations=0,new_GPU_smoke_executed=False,
        pending_condition_labels=len(pending),candidate_plan_frozen_before_GPU=True,case_selection_not_based_on_supplementation_outcomes=True,
        historical_answers_reused=len(outputs),reason='All four GPUs already had other users compute processes at frozen launch-time check.',gpu_snapshot='gpu_selection.json'))
    atomic_json(OUT/'readonly_validation.json',dict(passed=True,selected_cases=16,historical_logical_outputs=len(outputs),unique_historical_checkpoints=len(refs),
        all_complete_answer_ACC_rescored_exactly=True,all_raw_SHA256_exact=True,all_mask_audits_passed_in_historical_records=True,
        original_images_and_offsets_exact=True,cuda_initialized=torch.cuda.is_initialized(),historical_checkpoint_hashes=refs,historical_raw_hashes=rawrefs,source_sha256=sha256(__file__)))
    print('READONLY AUDIT PASS',len(outputs),'historical outputs;',read(OUT/'readonly_summary.json'),flush=True)

if __name__=='__main__':main()
