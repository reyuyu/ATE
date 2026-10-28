# Phase 4.0：冻结跨数据集资源

入口说明见 `../../benchmarks/README.md`。

本目录包括八个独立 Manifest（六个评测单元，其中 POPE 分为三个策略）、全集清单、
冻结抽样、全部图片哈希、Prompt/候选 Token 映射、协议与审计。
Native Full Smoke 的真实状态仅以 `smoke/validation.json` 为准；没有该文件不能宣称 GPU 已验证。
本轮不运行任何干预或全量 Benchmark，后续正确性前提见最终 `report.md`。
