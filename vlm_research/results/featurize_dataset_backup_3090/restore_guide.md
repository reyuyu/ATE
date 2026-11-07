# 新实例恢复说明

## 当前状态

**尚无成功上传的 Dataset ID。六次实际上传均在平台创建数据集时返回 HTTP 401，退出码 101。当前不能通过 Featurize 平台跨实例恢复。** 不要执行一个虚构的 ID，也不要把本地备份目录误认成平台备份。

## 先解除认证阻塞并完成上传

实例预置 `FEATURIZE_API_TOKEN` 已存在，但当前被平台拒绝。请从 Featurize 控制台进入这台 RTX 3090 的在线终端，确认实例认证有效；若仍报 401，请平台刷新该实例令牌或联系平台客服。不要把 Token 发到 GitHub、聊天或日志。

认证修复后，在现有实例按顺序运行：

```bash
set -e
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-textvqa-ocr
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-mme
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-pope
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-scienceqa-img
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-mmbench
featurize dataset upload /home/featurize/data/ate-recovery/backup_staging_3090/ate3090-gqa-testdev-balanced
```

每次需记录退出码、平台成功提示及真实 ID，填入 `dataset_registry.json` 和 `upload_status.json`；仅退出码为 0 仍应确认平台 Dataset ID。官方说明 CLI 支持断点续传，可对同一个目录重跑命令；当前安装版无单独 `--resume` 参数，401 修复前不要重试。不要重跑数据下载、图片全量 SHA 或 CPU 回归。

## 有真实 ID 后，在另一台 Featurize 实例恢复

实际安装 CLI 的帮助为：`featurize dataset extract <dataset_id> [destination]`。它流式解压，无需额外保存压缩包。下载功能限 Featurize 实例，不是任意外部服务器下载接口。

先取得本证据分支中的登记表与恢复脚本。将下面命令写在它们所在目录运行；当前登记表 ID 为 null，程序会明确退出，不会伪造恢复成功。

```bash
python - <<'PY'
import json, pathlib, subprocess
registry=json.loads(pathlib.Path('dataset_registry.json').read_text())
assert all(d['dataset_id'] and d['upload_status']=='UPLOADED' for d in registry['datasets']), '尚未完成平台上传，缺少真实 Dataset ID'
root=pathlib.Path('/home/featurize/data/ate-restored')
for d in registry['datasets']:
    target=root/d['benchmark']
    target.mkdir(parents=True,exist_ok=True)
    subprocess.run(['featurize','dataset','extract',d['dataset_id'],str(target)],check=True)
PY
```

对每个解压目录找到 `bundle_manifest.json`。CLI 归档是否带顶层目录由实际解压结果决定，不能写死两层路径：

```bash
find /home/featurize/data/ate-restored -name bundle_manifest.json
# BUNDLE 设为实际包含 bundle_manifest.json 的目录，RUNTIME 为一个尚不存在的新目录。
CUDA_VISIBLE_DEVICES='' python restore_bundle.py "$BUNDLE" --output "$RUNTIME"
```

恢复脚本用 CPU 校验小型元数据 SHA、所有 JSON/JSONL、题数、ID/Gold 双向映射、所有图片路径与长度，并在 Pillow 可用时抽读 3 张图。缺少 Pillow 可先安装 `Pillow==11.1.0`；脚本会如实报告是否抽读。此检查复用已有图片 SHA 清单，不宣称进行了新的全图哈希或远端逐文件核验。

新输入在 `$RUNTIME/<view>/inputs.jsonl`，Gold 独立在同目录 `answers.json`。`image_path` 自动指向新解压根目录，不要求旧实例的 `/home/featurize/data/ate-recovery` 存在。`source_manifest.json` 中的旧路径仅为历史证据，不是运行依赖。原始 Prompt、题目顺序和非路径字段完全保留。推理只读取 inputs；评分类逻辑单独读取 Gold，原始 `source_annotations/` 同样可能含 Gold。

## ATE 代码、环境与协议

证据分支只含轻量备份资料，不作为研究代码最新基线。研究源码基线是 `codex/cvpr2026-information-horizon-alignment@e281754ef17b3accc09d6938b5b9f020048b2aec`；GQA 执行源码来自 `codex/c3v2-gqa-testdev-balanced-12578@57e37ce024b1b2adf9fa24fc24aabec32a1fed08`。

当前实例环境与 CPU 验收报告仍在持久化 `/home/featurize/work/ATE/vlm_research/results/featurize_new3090_recovery/`。原 `ate_data.py` 桥接器固定旧 ROOT；新实例应将其路径配置接到本次生成的 runtime 视图，不能直接照抄其旧 SHA 或绝对路径保护。新 runtime manifest 已包含新的 inputs SHA；冻结原实验 manifest 保持不变。也可直接使用 ATE 适配器读取新 inputs 中的原字段；Paper 视图必须使用其已冻结的 `prompt`。

平台数据包不含模型或 Python 环境。此前已验证 Python 3.11.8、torch 2.6.0+cu124、transformers 4.49.0、Pillow 11.1.0。HF 模型仍为 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，其本地文件不属于此次跨实例备份；Author-Native 模型也未被替代。本说明不执行任何模型或 GPU 实验。

MME 单图 SHA 差异、POPE 9000/8910 与 TextVQA 两协议见 report_zh.md。未来成功上传后仍应区分平台确认和下载后逐文件校验，后者本轮状态为 `REMOTE_CONTENT_NOT_FULLY_VERIFIED`。

官方文档：[CLI](https://docs.featurize.cn/docs/manual/cli)、[数据集](https://docs.featurize.cn/docs/manual/dataset)。官方数据集说明还有长期不使用自动清理规则（普通用户 3 个月、付费用户 6 个月），平台备份不代表永久保留。
