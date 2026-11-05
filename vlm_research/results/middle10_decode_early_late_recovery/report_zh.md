# Middle10早层／深层Decode视觉回看恢复

## 当前状态：PARTIAL，等待本轮GPU授权

尚未执行真实GPU Smoke或Early/Late正式生成，不能判断早层还是深层恢复更有效。
没有启动性能测试，也没有加载GPU模型。

## 已完成

- 从指定`0a27cef983b5ae1f20930f2660afd59c11bbac7e`创建独立Worktree与分支`codex/middle10-decode-early-late-recovery`。
- 原500题、309个图片SHA簇不变；图片SHA与可解码性、固定模型文件SHA均通过CPU核验。
- Native与E共1,000条历史答案已严格核对样本、输入绑定、源码、分数及首Logits绑定。
- Native为49.68%，E为45.14%，差4.54pp；二者500题首Logits SHA全部相同。
- 18项CPU回归测试通过，包括完整Prefill不变、真实Decode Image概率屏蔽、合法Self/BOS、MLP、稠密Cache、原位置及异常路径清理。
- 新增代码只复用历史E路径并修改Decode允许读图层，旧源码、历史结果、主分支均未修改。

## 冻结条件

| 条件 | 真实Decode允许读图层 | 状态 | Accuracy |
|---|---|---|---:|
| Native | L0–L31 | 历史500题核验复用 | 49.68% |
| E | L11–L20 | 历史500题核验复用 | 45.14% |
| Early | L0–L20 | 未运行 | 未测量 |
| Late | L11–L31 | 未运行 | 未测量 |

四条件的整个Prefill和P−1完全Native。仅对真实Decode Query→原始Image Key施加Softmax前Hard Mask；不做o_proj置零、物理Token退出或KV优化。

## 待执行

1. 收到本轮物理GPU0/1授权后，重新检查UUID、显存和进程，冻结代码与协议。
2. 双卡各3题真实Smoke：Native/No-op严格一致、历史E复现、独立原生Mask参考、禁止概率为零、RoPE、Cache与Hook。
3. 按原309图片簇分片，Early/Late各500条；同图所有条件同卡，原子检查点支持校验续跑。
4. Image SHA整簇20,000次配对Bootstrap；输出四条件汇总、逐题答案及全部损害／恢复案例。
5. 无论是否达到47.68%，本轮均不测速、不实现物理版本。达到门槛时仅建议后续物理方案，随后停止。

## 复现

环境：`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`。

```bash
python -m vlm_research.middle10_decode_regions.prepare
# 实际GPU授权确认、源码提交后，才可执行：
python -m vlm_research.middle10_decode_regions.run --freeze --devices 0,1 --authorization-text "实际用户授权原文"
python -m vlm_research.middle10_decode_regions.run --driver
python -m vlm_research.middle10_decode_regions.analyze
```

本报告明确为准备阶段记录，不代表1,000条新增生成已经完成。
