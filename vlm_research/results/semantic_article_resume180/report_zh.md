# S＋ASSISTANT＋冠词：180题续跑准备

本次资源核验：四张GPU均有其他人的活动计算进程。尽管各卡仍有约21GB空闲显存，也不视为真正空闲；没有加载模型、共用GPU、抢占或终止进程，没有后台等待或自动重试。

本次仅完成CPU准备，不产生新的ACC。冻结了原180题、8个S保留策略，共1440份逐题接收集合；精确位置、输入SHA、源码快照及可复用检查点见frozen_requests及coverage_and_budget。缺失生成数按同题同集合去重。已有检查点按提交字节、输入及NPZ SHA核验；只匹配历史条件但未逐条检查历史记录的条目标明需运行前审计。

主候选为S-assistant-article及S-assistant-noun-article，对照为S、S-assistant、S-article、S-noun-article、S-instruction4及S-question-extra4-role。使用原冻结选择规则，不重新选词，不增加数据，不改变原Prompt。所有集合均包含原S、保持整词闭包，且均处于原R，不包含P−1或Image Query。

原60题初筛仍为：S55/60，S+ASSISTANT57/60，S+ASSISTANT+冠词58/60，All及Question-Only57/60。180题、360题和完整3000题的新候选ACC均未取得；不把60题高分解释成超过完整3000题最佳基线。旧轮次超时记录没有修改。

恢复时须重新检查真正空闲GPU，重新通过严格7B Smoke，再使用本次冻结位置完成缺失生成；不得在已有他人进程的卡上运行。当前状态为资源阻塞、CPU准备完成，研究目标尚未完成。
