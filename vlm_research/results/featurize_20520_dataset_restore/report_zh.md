# 新实例20520：Windows数据恢复

**状态：PASS**

来源：`D:\ATE-dataset-backups-3090`；服务器：`workspace.featurize.cn:20520`。
实际数据位于本地ext4盘：`/home/featurize/ate-local-data-20520/ate-recovery`。兼容入口：`/home/featurize/data/ate-recovery`（软链接）。

| 数据集 | 基础题数 | 图片数 |
|---|---:|---:|
| MMBench | 4377 | 1132 |
| POPE | 9000 | 500 |
| GQA-testdev-balanced | 12578 | 398 |
| ScienceQA-IMG | 2017 | 2017 |
| TextVQA-OCR | 5000 | 3166 |
| MME | 2374 | 1187 |
| OCRBench-v1 | 1000 | 930 |

8个传输包（七类数据及补充图片/来源元数据），共2,226,646,474字节，服务器端SHA256全部通过。六个原始包与D盘固定archive_index一致，OCRBench直接从D盘目录打包。ZIP解压执行CRC检查，未重复全量图片SHA扫描。
六类数据的14个输入视图及Gold文件与原验收SHA256逐字节一致；全部Question ID/Gold键集合一致，图片路径全部可达。OCRBench保留1000题官方行序与原始ID，930张图全部存在。
POPE完整9000题及论文8910题视图同时保留，论文三策略为2910/3000/3000。TextVQA保留Image-only与Paper-OCR独立视图。MME已知单图SHA差异原样保留。补充GQA验证图2346600单独存放，不混入testdev冻结题集。
七类各取一张图片解码检查：`{"MMBench": {"status": "PASS", "size": [302, 232], "mode": "RGB"}, "POPE": {"status": "PASS", "size": [640, 427], "mode": "RGB"}, "GQA-testdev-balanced": {"status": "PASS", "size": [640, 427], "mode": "RGB"}, "ScienceQA-IMG": {"status": "PASS", "size": [302, 232], "mode": "RGB"}, "TextVQA-OCR": {"status": "PASS", "size": [1024, 664], "mode": "RGB"}, "MME": {"status": "PASS", "size": [416, 114], "mode": "RGB"}, "OCRBench-v1": {"status": "PASS", "size": [104, 27], "mode": "RGB"}}`。
本地盘剩余389.02 GiB。本轮未下载模型、未运行GPU实验；结束时GPU进程：`无`。

传输与恢复证据：datasets_restore_plan.json、transfer_progress.json、dataset_restore.json、restore_datasets.py。完整数据留服务器本地盘，Windows原备份保留。
