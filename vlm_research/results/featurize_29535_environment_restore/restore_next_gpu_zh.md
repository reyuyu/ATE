# ATE 新 GPU 恢复步骤（仅环境恢复）

本次端口：29535。代码和历史结果保留于 `/home/featurize/work/ATE`；模型、数据、虚拟环境均保存至 `/home/featurize` 的本地 ext4 盘，不写入 NFS 持续盘。

## 1. 先查现有资源

```powershell
$SshPort = 29535 # 下次改为新端口
ssh -p $SshPort featurize@workspace.featurize.cn
```

在 Linux 上检查：

```bash
findmnt -T /home/featurize
findmnt -T /home/featurize/work
ls -ld /home/featurize/ate-local-models/llava-1.5-7b-hf /home/featurize/data/ate-recovery
test -x /home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
```

目录存在后还需检查权重索引、所有分片、Processor/Tokenizer 和版本回执；不能假定旧实例本地盘自动迁移。已有可信校验文件直接复用，不重复下载或扫描全盘。

## 2. 从 Windows 主动传输缺失数据

本次只恢复两个原始 ZIP；其 SHA256 固定于 `datasets_restore_plan.json` 和 D 盘 `archive_index.json`。
下列命令可在 Windows PowerShell 执行，密码由 SSH 自己交互读取，不写入脚本：

```powershell
$SshPort = 29535
$Backup = 'D:\ATE-dataset-backups-3090'
$Remote = '/home/featurize/ate-local-recovery-archives'
$Index = Get-Content -LiteralPath (Join-Path $Backup 'archive_index.json') -Raw | ConvertFrom-Json
ssh -p $SshPort featurize@workspace.featurize.cn "mkdir -p '$Remote'"
foreach ($Name in @('ate3090-gqa-testdev-balanced.zip','ate3090-textvqa-ocr.zip')) {
    $Local = Join-Path $Backup $Name
    $Expected = ($Index.datasets | Where-Object filename -eq $Name).sha256
    $Hash = (Get-FileHash -LiteralPath $Local -Algorithm SHA256).Hash.ToLower()
    if ($Hash -ne $Expected) { throw "本机备份校验失败：$Name" }
    $Existing = ssh -p $SshPort featurize@workspace.featurize.cn "if test -f '$Remote/$Name'; then sha256sum '$Remote/$Name'; fi"
    if ($LASTEXITCODE -ne 0) { throw '服务器检查失败' }
    if ($Existing -match "^$Hash\s") { continue }
    if ($Existing) { throw "服务器已有不同内容，保留并检查：$Name" }
    scp -P $SshPort $Local "featurize@workspace.featurize.cn:$Remote/$Name.part"
    if ($LASTEXITCODE -ne 0) { throw "传输失败：$Name" }
    ssh -p $SshPort featurize@workspace.featurize.cn "printf '%s  %s\n' '$Hash' '$Remote/$Name.part' | sha256sum -c - && mv -n '$Remote/$Name.part' '$Remote/$Name'"
    if ($LASTEXITCODE -ne 0) { throw "服务器 SHA256 失败：$Name" }
}
```

本次实际使用 Paramiko SFTP，保留已有顺序上传的前缀，再用 4 个互不重叠的范围续传；每段与完整 ZIP 都做 SHA256 校验。恢复工具与 `transfer_progress.json` 可复用同一实例的断点，不必重传。下一端口要在新工具目录中替换连接端口和恢复目录编号，并重新保存主机密钥，不能覆盖旧记录。

ZIP 内含 `bundle_manifest.json`、`image_sha256.json`、Gold、原题和 `restore_bundle.py`。单独解压、发出新运行视图的通用方式：

```bash
L=/home/featurize/ate-local-data-new
mkdir -p "$L/bundles"
python3 -m zipfile -e /home/featurize/ate-local-recovery-archives/ate3090-gqa-testdev-balanced.zip "$L/bundles"
python3 "$L/bundles/ate3090-gqa-testdev-balanced/restore_bundle.py" "$L/bundles/ate3090-gqa-testdev-balanced" --output "$L/gqa-runtime"
```

TextVQA 同理。Paper-OCR 必须选择 `TextVQA-OCR` 视图，不能换成 Image-only。
若要复用 ATE 原样路径和逐字节冻结 Manifest，应像本次 `restore_datasets.py` 一样，将
`/home/featurize/data/ate-recovery` 链接至本地盘新目录，恢复 `images/` 对应链接，再按源验收记录重新核对原始输入和 Gold SHA256。
不能仅把新绝对路径运行视图当成原历史 Manifest；两者路径表示可能不同。

## 3. 固定模型 Revision 与断点恢复

只有权重缺失且获得下载授权时，使用本次 `restore_model.py`。复用已审计的固定文件清单，独立保存恢复证据：

```bash
R=/home/featurize/work/ATE
S=$R/vlm_research/results/featurize_29535_environment_restore
export ATE_RECOVERY_OUT=$R/vlm_research/results/environment_restore_$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$ATE_RECOVERY_OUT"
cp "$S/hf_revision_manifest.json" "$ATE_RECOVERY_OUT/hf_revision_manifest.json"
export ATE_MODEL_DIR=/home/featurize/ate-local-models/llava-1.5-7b-hf
CUDA_VISIBLE_DEVICES='' python3 "$S/restore_model.py"
```

模型为 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
脚本先尝试 Hugging Face 主站，必要时从同 Revision 的镜像续传；所有权重必须匹配固定 LFS SHA256，其余文件匹配 Git blob 哈希。
`.part` 可以续传；已有不同哈希的最终文件会停止而不覆盖。这里只下载和检查文件，不构造模型。

## 4. 恢复冻结 Python 环境

```bash
E=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449
test -x "$E/bin/python" || python3 -m venv "$E"
S=/home/featurize/work/ATE/vlm_research/results/featurize_29535_environment_restore
CUDA_VISIBLE_DEVICES='' "$E/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r "$S/requirements.lock.txt"
"$E/bin/python" -m pip check
CUDA_VISIBLE_DEVICES='' "$E/bin/python" -c 'import torch,torchvision,transformers; print(torch.__version__,torchvision.__version__,transformers.__version__); assert torch.__version__=="2.6.0+cu124" and torchvision.__version__=="0.21.0+cu124" and transformers.__version__=="4.49.0"; assert not torch.cuda.is_initialized()'
```

实际下载时可用本次 `install_environment.sh` 和本地 wheelhouse 避免重复取已下载的 Torch/Triton。
该入口的证据输出目录含本次端口编号，复制到下一实例时应改成新的独立输出目录；不要写回旧恢复报告。

## 5. CPU 验收后停止

先核对本次 `model_integrity.json`、`dataset_restore.json`、`history_archive_reuse.json`。
本次 `verify_restored_environment.py` 只读取权重头和配置、解码选中图片、重建 200 个历史输入 Token/像素哈希，不反序列化模型，不初始化 CUDA。
下一实例复用时修改基础设施脚本中的证据目录编号、提供本次新回执，再运行 CPU 检查。

验收通过只说明资源与 CPU 输入协议恢复正确；真实 FP16/Eager GPU Prefill/Decode 仍需后续实验任务验收。
保留所有历史逐题预测，不能因恢复环境而重新生成 Benchmark。
