# 复现与审计说明

本分支继承2d671/9602及3297559的冻结依赖；source_manifest.json记录本轮实际运行源码SHA。核心SVH-R V2未修改，不使用C1/C2。

当前服务器SSD运行目录：/home/featurize/ate-runs/internvl3-svh-r-fast-query-search-3090
Python：/home/featurize/.venvs/ate-frozen/bin/python
模型：/home/featurize/ate-local-models/InternVL3-8B

CPU准备入口：vlm_research.internvl3_svh_r_fast_query_search.prepare
一次加载的生成入口：vlm_research.internvl3_svh_r_fast_query_search.run
独立评分入口：vlm_research.internvl3_svh_r_fast_query_search.analyze

未来重新复现应使用新的空运行目录并重新冻结协议与来源；设置PYTHONPATH指向完整本分支源码、ATE_FAST_RUN指向新目录、CUDA_VISIBLE_DEVICES为协议GPU UUID；为历史helper设置ATE_SEVEN_ROOT和ATE_SEVEN_RUN。恢复父实验480题Manifest、GQA图片和匹配的历史Native检查点，先执行CPU prepare，再执行run_safe，最后analyze。不能通过改本轮协议、覆盖已有检查点来重跑。

run_safe.py是仅修正最后SQLite关闭/状态日志顺序的后续副本，未参与本轮测量；原run.py和SHA原样保存。数据全部完成后的原始错误、CPU核验和GPU释放证据见post_run_bookkeeping_incident.json、failure.json和validation.json。

GitHub包含per_question.csv的全部1432条质量记录（1152新增、280历史Native参考）及Token IDs/Gold/评分、逐题测速、完整配置和压缩的逐层KV CSV。kv_layout_quality.csv.gz解压后SHA见artifact_manifest.json。未上传模型、图片、CPU像素缓存和SQLite；服务器server_only保留完整SQLite、历史Native绑定与日志。SHA256SUMS路径相对于Git仓库根目录，artifact_manifest.json列出Git与服务器文件范围。

本轮所有条件均为探索性数值豁免，不能声称数值等价或完整GQA非劣效。未来新机器时间不得与本轮拼接。质量生成正常EOS、上限16；固定32仅用于测速。
