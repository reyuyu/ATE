# Decode L0×L1：执行范围审阅

本轮基于 `4408b862e4760b6aadcb5135d2640f422778938b`，只在新分支和新结果目录工作。

## 实际改动

- `engine.py`与历史单层实验的引擎逐字节一致；相对导入的`policy.query_rows`改为本轮固定六条件。
- `query_rows`只在`past>0`且当前层不在开放集合时返回`[0]`。`past==0`的整个Prefill（含全部Image、R和P−1）始终不返回目标位置。
- 原始576个Image Key保持在Cache中。只克隆目标层真实Decode的Attention Mask，并将图像Key切片赋值为`-torch.inf`。非目标Mask逐项保持原值。
- 沿用历史`logits_to_keep=1`前向，每次生成从空Cache和空生成前缀开始。没有物理Token退出、KV删除、层跳算、训练或Prompt修改。
- Smoke临时请求Eager Attention概率，逐层汇总后释放矩阵；正式路径与审计路径全部有效Logits严格逐位一致。Attention行和沿用历史FP16容差，仅用于概率求和；Logits等价检查始终为`torch.equal`。

## 输入与历史复用

旧300题重新在CPU构造原生Prompt、Token IDs和图像预处理张量，全部匹配历史输入哈希。A/B/Native/Decode-None共1200个文件与基础Commit中的Git Blob逐字节核对，官方软评分重新计算。

历史Middle检查点字段名为`execution_source_manifest_sha256`与`execution_protocol_sha256`；其他历史记录使用`source_manifest_sha256`与`protocol_sha256`。适配只读取对应字段并核对实际文件哈希，不改变历史记录或评价含义。

历史文件保存完整Token序列及首Logits向量SHA，并未归档全部逐步Logits数组。因此，Smoke对历史文件核对完整Token和首向量SHA；另使用冻结历史引擎回放A/B，比较全部有效Logits。两类证据分别记录，不声称存在未保存的历史数组。

新800题在预测前冻结。全工作区历史候选计划与实际使用范围分开审计：候选池中未执行的图片不自动算作历史预测；实际TextVQA/TextCaps检查点身份和其他冻结输入清单均纳入排除。选定图片的全部官方问题一起保留，按完整图像SHA分到唯一Worker。

## 执行绑定

`protocol.json`、全部1100份输入Spec、样本Manifest、分片、层集合、模型文件清单及执行Python/HF源码均绑定SHA256。模型权重文件在启动前完整重算SHA。每份新Checkpoint记录源码清单、协议、输入、模型清单、运行Commit、GPU UUID与独立Cache审计。

推理源码冻结后只增加CPU分析、验证、绘图与报告文件；这些文件不参与生成并明确排除在冻结推理清单之外。统计目标、主非劣效界限、Bootstrap次数/种子和辅助检验家族仍由预先冻结的协议决定。

运行上限采用UTC 2026-11-05 10:00开始的保守六小时预算；15:30停止新增GPU任务，16:00为最终交付截止。GPU仅绑定用户指定的物理GPU2/3 UUID。
