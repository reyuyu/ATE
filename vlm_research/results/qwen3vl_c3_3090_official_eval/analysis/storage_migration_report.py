"""Append storage provenance only; never alter inference, scores or statistics."""
from pathlib import Path
import hashlib,json
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def main():
 m=json.loads((O/'local_disk_migration.json').read_text(encoding='utf-8'))
 assert m['status']=='LOCAL_EXT4_ACTIVE_PERSISTENT_CHECKPOINT_PRESERVED'
 r=json.loads((O/'resume_scripts/local_resume_source_manifest.json').read_text(encoding='utf-8'))
 for n,h in r['files'].items():assert hashlib.sha256((O/'resume_scripts'/n).read_bytes()).hexdigest()==h,n
 heading='## 用户授权的本地盘迁移与断点恢复'
 report=(O/'report_zh.md').read_text(encoding='utf-8')
 if heading in report:report=report.split(heading)[0].rstrip()+'\n'
 lines=['',heading,'',
  '用户于本轮运行中明确授权切换为本地盘执行、持续盘保留备份。迁移仅改变存储位置，未改变冻结输入、模型权重、生成参数、算法或评分规则。',
  f"迁移完成时间（UTC）：{m['activated_at_UTC']}。运行副本根目录：`{m['local_root']}`。原绝对路径通过独立 bind mount 指向本地 ext4；因此协议、输入 Manifest 和检查点绑定的 SHA256 保持一致。",'',
  '| 内容 | 本地运行位置 | 持续盘保留副本 |','|---|---|---|']
 for x in m['storage_mappings']:lines.append(f"| {x['label']} | `{x['active_local_path']}` | `{x['persistent_backup_path']}` |")
 lines+=['',
  f"全部模型必需文件重新核验 SHA256；{m['unchanged_input_paths']:,} 个引用图片路径完整存在。数据通过 rsync 传输校验，原有逐张读取时的图片 SHA256 检查继续启用，未重复扫描所有图片哈希。",
  f"本地校验通过：历史预测 {m['historical_predictions_verified']:,} 条、请求预测 {m['request_predictions_verified']:,} 条、逐题索引 {m['benchmark_indices_verified']:,} 条；保留了 {m['unpaired_completed_indices']} 个中断时仅完成单条件的索引，其缺失条件按原规则补齐。",
  '本次迁移新增一次模型加载。隔离的恢复入口只跳过已经完成的历史 Smoke/GQA1000 及完整 Benchmark 视图，并保留已因 OOM 停止的 TextVQA 视图；不重试该 OOM，不重新生成已完成预测。恢复入口与冻结原入口的差异及 SHA256 见 `resume_scripts/local_resume_orchestration.diff` 和 `local_resume_source_manifest.json`。',
  '迁移前已确认旧 GPU 进程退出且显存释放，再启动唯一的新 GPU 进程。迁移前持续盘副本完整保留；新增进度先写本地，完整结果备份状态以最终归档与校验回执为准。',
  '质量生成包含迁移前后的同卡结果，输入与数值协议一致。严格 Dense/Physical 数值门禁的失败记录保留，存储迁移不构成数值等价验收。']
 complete=O/'efficiency_completion.json'
 if complete.exists() and json.loads(complete.read_text())['status']=='COMPLETE':
  lines+=['正式 48 题效率测量已全部在迁移后执行，使用原有统一计时边界，磁盘读取不计入计时。']
 else:lines+=['正式 48 题效率测量尚未完成；后续将按原有统一计时边界在本地盘环境执行，磁盘读取不计入计时。']
 receipt=O/'storage_migration_windows_receipt.json'
 if receipt.exists():
  b=json.loads(receipt.read_text(encoding='utf-8'));lines+=['',f"迁移证据已在 Windows 完成 {b['files']} 个文件的 SHA256 校验（{b['at_UTC']}）；该回执仅覆盖迁移记录和恢复源码，不代表尚未结束的完整实验已完成备份。"]
 (O/'report_zh.md').write_text(report+'\n'.join(lines)+'\n',encoding='utf-8')
 print('STORAGE_MIGRATION_PROVENANCE_APPENDED_NO_SCORE_CHANGES')
if __name__=='__main__':main()
