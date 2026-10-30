# Late-Layer Attention Source Decomposition

**状态：partial，尚无正式实验结果。**

冻结200张新图片；采集CPU测试20项通过，独立统计测试见statistics_tests.txt。冻结采集Commit：`9352a999c0cb4b5c0ffbdc7fd2941165874630ab`。
已完成真实GPU Smoke的物理卡：[1, 2, 3]；这是一张独立Smoke图在各卡的重复验证，不是正式200图结果。
Native Full完成图片数：0；正式消融完成图片数：0；目标词位置尚未选择/冻结。

## 阻塞原因

GPU0被其他用户的计算任务占用，初次预检拒绝加载。没有终止、修改或抢占该任务。GPU1–3的Smoke进程已退出。
保留四卡要求和全部Smoke通过后才开始Full的门槛；没有改成三卡、混用旧结果或把Smoke冒充机制结论。

## 已验证和未执行

本轮已完成Smoke的来源分解检查60项；具体输出、No-op/Cache/Hook审计和原始Logits哈希均保留。
正式来源损失、Visual−Function对比、L25–L31连续性、Recent−Older效应均尚未测得；不填零、不预测方向、不绘制正式论文图。
五张主图、CSV汇总和中文研究报告的生成代码已实现，但受全部200图完成校验约束。

## 恢复

GPU0及其余三卡空闲后，按[README.md](README.md)原命令恢复；有效Smoke记录会复用，冻结数据/采集源码不变。
本次快照：`partial/1793348048945420922/status.json`；实际GPU状态记录于该快照，不能把以后资源变化当作已有实验结果。
