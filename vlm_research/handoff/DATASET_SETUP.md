# 数据与模型恢复：新服务器执行，原服务器不下载

本页目标是恢复可继续统计实验的 **完整 GQA 图片资源与冻结队列**，不是只搬一张图。这里的完整目录是 **MMEB发行中的GQA子集，不等于Stanford GQA全基准全集**；当前冻结统计任务不需要另下全基准。仓库包含题目、ROI、统计、Attention 缓存及逐样本历史，但不包含 GQA 原图或模型权重。旧机原图来自 MMEB-V2 的 `image-tasks/mmeb_v1.tar.gz` 解包版本。优先下载同一来源，避免 JPEG 重编码改变历史 SHA256。

## 1. 数据来源和需要保留的版本

| 资源 | 已验证来源/版本 | 用途 |
| --- | --- | --- |
| GQA ChooseRel CSV | [官方仓库](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)，revision `c9a1111dac4baa175f67716fab473789320187c0` | 964 问题/785 独立图片；已有副本，含 object1/object2 框 |
| GQA 图片 | [TIGER-Lab/MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)，文件 `image-tasks/mmeb_v1.tar.gz` | 优先恢复全部 `MMEB/GQA/`，支持统计性后续实验 |
| 官方解包说明 | [VLM2Vec 固定版本下载脚本](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/f658ebeb49704784330ffe9e68a10796698d75e3/experiments/public/data/download_data.sh) | 脚本确认上述 HF 仓库及 tar 路径；不要整体执行其训练/视频下载 |
| 官方 GQA 下载入口 | [Stanford GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) | 上游备用来源；没有证明其 JPEG 字节等于本实验 MMEB 副本 |
| 模型 | `llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | 必须保持同权重、Tokenizer、Processor、Chat Template |

**重要可复现边界**：旧机原始下载未记录 MMEB HF commit，不能补造一个。交接时已核对官方 GitHub 脚本和本地 6.6GiB tar 的目录结构；当前机器对 HF API 有 DNS/TLS 问题，**本次未重新下载或从 HF 验证 archive 内容**。新服务器先解析并记录 HF revision，再下载固定 revision，最后以冻结 Manifest 的逐图 SHA256 判断是否确实恢复相同实验输入。HF 最新版来源正确也不自动保证历史字节完全相同。

## 2. 下载 MMEB 图片包，固定下载 revision

使用已安装的 `huggingface_hub==0.29.3`。磁盘先执行 `df -h`；本地 tar 约6.6GiB，额外预留解包空间、约14.1GB模型文件和缓存，建议至少40GB可用，实际按 archive 清单调整。不要下载 MMEB 视频、训练或文档任务。

```bash
conda activate vlm_research
export ATE_DATA_ROOT="$HOME/datasets/ATE"
mkdir -p "$ATE_DATA_ROOT"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
root = Path(os.environ['ATE_DATA_ROOT'])
record = root/'mmeb_download_source.json'
repo = 'TIGER-Lab/MMEB-V2'
name = 'image-tasks/mmeb_v1.tar.gz'
if record.exists():
    spec = json.loads(record.read_text())
    assert spec['repo_id']==repo and spec['filename']==name
else:
    info = HfApi().dataset_info(repo, files_metadata=True)
    spec = dict(repo_id=repo, repo_type='dataset', revision=info.sha, filename=name)
    # Save before large transfer so a retry stays on the identical revision.
    record.write_text(json.dumps(spec,indent=2)+'\n')
path = Path(hf_hub_download(repo_id=repo, repo_type='dataset', revision=spec['revision'],
                          filename=name, local_dir=str(root)))
h = hashlib.sha256()
with path.open('rb') as f:
    for part in iter(lambda:f.read(8*1024*1024), b''):h.update(part)
spec.update(downloaded_path=str(path.resolve()), archive_sha256=h.hexdigest(), bytes=path.stat().st_size)
record.write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps(spec,indent=2))
PY
```

网络失败保留下载记录和错误，修复本机网络再重试同一 revision。不要使用 `pip install -U` 修复网络。若原网络代理失效，可针对当前命令取消 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，不修改全局配置；镜像只能作为同一 repo/revision 的传输替代，仍校验逐图哈希。不要在日志中输出 token 或代理凭据。

## 3. 解出完整 GQA 图片目录

```bash
# 先查看 archive 的 GQA 条目；这会顺序读取压缩包。
tar -tzf "$ATE_DATA_ROOT/image-tasks/mmeb_v1.tar.gz" --wildcards 'MMEB/GQA*' \
  > "$ATE_DATA_ROOT/gqa_archive_members.txt"

# 保留 MMEB/GQA 层级，不执行官方脚本的 mv MMEB/* 操作。
tar -xzf "$ATE_DATA_ROOT/image-tasks/mmeb_v1.tar.gz" \
  -C "$ATE_DATA_ROOT/image-tasks" MMEB/GQA
export ATE_GQA_ROOT="$ATE_DATA_ROOT/image-tasks/MMEB/GQA"
```

预期文件 `image_<img_id>.jpg`，例如 `image_2324597.jpg`。如果 archive 的实际 member 不含上述目录，先记录清单调查，不能把命令成功当作已解出图片。下载和解包均在 Git 仓库外，图片不上传 ATE。

已承诺的统计队列：Phase3.2 **240张不同图片，left/right各120**，加 Phase3.1 **12张验证图**，合计 **252张独立图片**。Phase3.4 的48主图和5案例都来自240图，不额外增加图片。CSV的785图是上游资源范围，不是已经完成785图模型实验。

## 4. 校验252张图、文件完整性和历史缓存

在新服务器实际仓库目录执行：

```bash
cd /path/to/ATE/vlm_research
python handoff/verify_recovery.py \
  --image-root "$ATE_GQA_ROOT" \
  --output "$ATE_DATA_ROOT/recovery_audit_001.json"
```

要同时核对官方CSV全部785张图片（为后续统计准备，不代表785图已有模型结果）：

```bash
python handoff/verify_recovery.py --image-root "$ATE_GQA_ROOT" \
  --verify-source-images --output "$ATE_DATA_ROOT/recovery_source785_001.json"
```

其中252图有历史SHA256基准，另外533图只检查存在/可解码/尺寸并保存当前SHA256，不伪称有历史哈希。缺失不能按预测结果替换。

要求252张全部存在、JPEG可解码、尺寸与历史CSV一致、SHA256匹配。脚本还校验交接时冻结的代码/关键CSV/JSON/Attention缓存指纹，且不使用GPU。首次审计输出用新文件名，禁止覆盖。若只想验证 Git 恢复可不传 image-root，但这种 passed **不代表图片已经准备好**。

图片缺失/哈希不一致时：记录img_id、路径、expected/actual哈希，不换图、不重采样、不改冻结Manifest来“通过”。检查是否重编码、文件名映射错误或上游归档变化。若只能取得不同字节的图片，应建立独立数据版本和新Full基线，不能把结果当作原实验无缝续算。对缺失的旧机原始字节可请求原样文件迁移，但不要假称已从公开源恢复成功。

ROI无需另下scene graph：官方CSV的框已在仓库，Phase3.4的实际CLIP裁剪几何、两对象Patch集合、Sink候选与随机控制也已冻结。不要重新调用准备脚本抽样或改ROI规则。

## 5. CSV丢失时的同版本恢复命令

正常git clone已经包含CSV；下列仅用于交叉核对，下载到外部目录，不覆盖已提交文件：

```bash
curl -fL --retry 3 \
  'https://raw.githubusercontent.com/FightingFighting/cross-modal-information-flow-in-MLLM/c9a1111dac4baa175f67716fab473789320187c0/datasets/GQA_val_correct_question_with_relChooser_ChooseRel.csv' \
  -o "$ATE_DATA_ROOT/ChooseRel.upstream.csv"
sha256sum "$ATE_DATA_ROOT/ChooseRel.upstream.csv"
```

预期SHA256：`0bbc45d9c9b609409f30d7ecf9be6f2957347c4cdcc5480b531ae0b8ef5282c0`。原始题号按字符串读取，尤其保留 `0216546` 等前导零。

## 6. 模型恢复与环境

优先复用新服务器已有的同revision权重。没有时只下载同一checkpoint；不要下载量化版/其他LLaVA：

```bash
export ATE_MODEL_ROOT="$HOME/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('llava-hf/llava-1.5-7b-hf',
                  revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
                  local_dir=os.environ['ATE_MODEL_ROOT'])
PY
python handoff/verify_recovery.py --image-root "$ATE_GQA_ROOT" \
  --model-dir "$ATE_MODEL_ROOT" --output "$ATE_DATA_ROOT/recovery_with_model_001.json"
```

校验使用 `configs/model.json` 已记录的文件尺寸及三片权重LFS SHA256，不信任目录名称。读取14GB权重校验会花时间，但不会加载GPU。仓库原`download_model.py`也可下载，但会重写`configs/model.json`本地路径，跨机建议使用上述命令和独立路径适配，保留历史文件不变。

环境优先复用：Python3.11、torch2.6.0+cu124、torchvision0.21.0+cu124、transformers4.49.0；其余见`requirements.txt`和`requirements-lock.txt`。若确实缺环境，按工程README新机器安装步骤创建，不升级旧环境：

```bash
conda create -n vlm_research python=3.11 -y
conda activate vlm_research
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

记录新机器的真实Python patch版本、驱动、GPU、Torch/CUDA/Transformers及安装库源码SHA256。CUDA wheel自带12.4运行库；`nvidia-smi`最高支持版本不是PyTorch实际运行库。新机器的不同GPU架构可能影响FP16逐元素可重现性，不能静默跳过历史断言。

## 7. 原脚本仍有绝对路径：不是改环境变量就能完成迁移

`ATE_GQA_ROOT`和`ATE_MODEL_ROOT`在本页是给下载/适配器用的变量，**旧runner不会自动读取它们**。继续前采用独立远端入口/本机路径适配，不修改冻结数据与核心库：

1. 原`attention_inheritance_pilot.py`的CPU子进程使用`/home/zsy/.conda/envs/vlm_research/bin/python`；远端入口改成当前`sys.executable`。
2. `vlm_common.load_model(model_path=...)`已有模型目录参数。新入口显式传本机模型路径，避免修改已提交`configs/model.json`。
3. 从原240图Manifest读样本后只在**运行时副本**映射`image_path`到`ATE_GQA_ROOT/image_<img_id>.jpg`；原始Manifest、SHA256和题目标签保持原样。新运行配置同时记录原路径、新路径、映射规则与原Manifest SHA。
4. 可将当前单案runner复制为新的`attention_inheritance_remote.py`，只增加`--model-path`、`--image-root`和上述Python路径适配；复用原`VisualAttentionInheritance`、`inputs_and_history`和评分。记录新入口代码SHA256，不能只沿用原runner名称的hash。
5. `roi_sink_probe.PARENT`与Attention缓存路径基于项目ROOT，git clone后可复用；不调用其`config_and_data()`来强行改历史环境绑定。
6. 旧Phase3.2/3.3/3.4 runner默认写历史目录，并把环境、路径、代码hash绑入checkpoint。**不可直接执行来“继续统计实验”**；要新结果目录、新run binding，旧结果只读。不删除binding检查、不复制旧completed标记作为新运行验收。

如果保留完全相同的旧绝对路径，可用同文件符号链接恢复位置；只在自己有权限的目录操作，不覆盖他人环境、不要求管理员权限。一般优先独立适配器。
