# interVL-SVH-R-new：OCRBench v1全集

用户命名的配置：InternVL3-8B，Late14=L14–L27（0-based），其余冻结ATE协议不变。配置见../../configs/interVL-SVH-R-new.json。

预注册状态：PARTIAL，100/1000题继承上一轮完整预测；仅授权生成缺失900题。未填写或推测全量分数。先做4道未完成题的Physical结构Smoke，失败即停止。

该命名与完整覆盖不代表已通过历史全局数值等价门禁，EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER继续保留。沿用单448图块/256视觉Token的ATE协议，不混入官方动态24图块协议。
