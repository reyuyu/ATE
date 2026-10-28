# Phase 4.0：可审计的跨数据集评测套件

基线main：e89c8acce6e674ecfbd2c8acad57faf71bf4d547。

## 范围

新增 `vlm_research/benchmarks/` 与独立结果目录。
不修改历史Phase3.2～3.8结果、基础模型/共享源码、main或其他Worktree。
只运行18例Native Full Smoke，无全量Benchmark、B4/C2/Physical Exit、训练或Router。

## 冻结数据

- ChooseAttr240/240图；QueryAttr240/240图；LogicalObj240/240图。
- POPE三个策略各300题，共同50张COCO val2014原图，完整6题图片组。
- ScienceQA-IMG官方Test500题，500个原始图片文件/475个不同图片SHA。
- TextVQA v0.5.1 Val500题/309图，保留每题十个答案。
- 总计2620条评测记录、1528个不同图片SHA；官方GQA Balanced另行登记为blocked，不冒充已完成。

## 验证

GPU前136项CPU通过；最终141项通过，无跳过。
18个预定样本的实际Processor输入、生成和完整候选评分通过；原始全词表Logits可独立重算。
模型参数/Buffer前后字节哈希相同，无NaN/Inf/OOM，GPU0已释放。
权威重叠审计按数据集命名空间与图片SHA，避免跨数据集数字ID碰撞。
制品清单与最终源码哈希封存，新增图片/权重/环境不入Git。

## 评审重点

诊断候选Margin与生成官方评分隔离；QueryAttr不造二选一；TextVQA十答案软评分；POPE五项指标。
ScienceQA发布者顺序与全部4241条官方Test字段匹配后恢复PID。
所有样本在模型输出前冻结，失败恢复不换图。
完整Benchmark原图未全量下载；image-only子集结果不能直接冒充论文榜单。
B4/C2及Physical Exit仍需新任务/多Token/生成语义正确性验收，本PR不证明任何干预收益。

完整报告：`vlm_research/results/phase4_0_dataset_suite/report.md`。
操作顺序：`operations.md`；用法：`vlm_research/benchmarks/README.md`。
请勿自动合并main。
