"""Append an evidence-based metadata erratum; no input or scoring changes."""
from pathlib import Path
import json
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def main():
 d=json.loads((O/'protocol_description_correction.json').read_text(encoding='utf-8'))
 assert d['status']=='PROTOCOL_DESCRIPTION_CORRECTION_INPUTS_UNCHANGED' and d['identical_prompt_pairs']==1000
 assert not any(d[k] for k in ['model_input_changed','scorer_changed','algorithm_changed','frozen_manifest_changed','predictions_changed','GPU_rerun_required'])
 heading='## OCRBench 协议差异说明更正（实际输入未变）'
 report=(O/'report_zh.md').read_text(encoding='utf-8')
 if heading in report:report=report.split(heading)[0].rstrip()+'\n'
 lines=['',heading,'',
  '`input_manifest.json.protocol_differences` 中“OCRBench 的 ATE 文本含字面反斜杠 n、官方文本使用真实换行”的说明不准确。这是说明字段的误判，冻结 Manifest 原件保留，未悄悄改写。',
  '对可信备份 ZIP 的原始 `inputs.jsonl` 成员与恢复文件逐字节核验后，确认 1,000 道题的 ATE 输入和官方输入本来就都使用真实换行，Prompt 完全相同，且均符合此次核对的作者模板。',
  f"原始成员及恢复输入 SHA256：`{d['source_member_SHA256']}`。审计过程没有重新扫描图片或重新哈希整个数据包。",
  '两种标签保留各自索引及评分记录；同图、同 Prompt、同生成上限的请求可合法共用本轮新生成的预测，因此不能将两个标签视为两次独立推理实验。模型输入、Gold、Scorer、算法及生成参数均未修改，不需要 GPU 重跑。',
  '完整证据见 `protocol_description_correction.json`，可复核脚本为 `analysis/protocol_metadata_audit.py`。']
 (O/'report_zh.md').write_text(report+'\n'.join(lines)+'\n',encoding='utf-8')
 print('OCR_DESCRIPTION_ERRATUM_APPENDED_INPUTS_AND_SCORES_UNCHANGED')
if __name__=='__main__':main()
