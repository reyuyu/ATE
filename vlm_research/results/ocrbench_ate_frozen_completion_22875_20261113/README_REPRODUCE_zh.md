# CPU复核与恢复说明

本目录仅补原冻结ATE协议OCRBench，不是InternVL动态24图块新协议。

1. 使用标准Python 3.11+运行 `python verify_cpu.py`，无需GPU、模型或原图。
2. 冻结核心分别从 `protocols/{llava,qwen,internvl3}/protocol.json` 的 source_commit 及 source_SHA256 精确取回；不可用当前 main 中相似文件替代。
3. execution/ 下入口的 run_common.py 记录本次服务器资源根目录；恢复到其他实例时仅调整隔离入口资源路径，先CPU比对 inputs/ 处理后输入哈希，再允许GPU。不要改旧Manifest/核心算法。
4. InternVL已完整1000题，不再生成；LLaVA已完成条件同理。Qwen若BLOCKED，只能在原协议允许的足够显存资源上恢复缺失请求；更换分辨率或Attention Backend属于新协议，须另行授权与标记。
5. 每题独立KV，原始输出逐题原子持久化。只允许恢复缺失ID，完整记录必须验证原protocol哈希后复用；不得把Smoke成功样本再跑一遍。

源分支：
- LLaVA: codex/featurize-new3090-c3v2-reaccess-sevenbench @0414e553ba6fd1e21626ecf741489d225b44279c
- Qwen: codex/qwen3vl-c3-3090-official-eval @7e18895a6ccd57c4cb3aaedd5d512789e7467b59
- InternVL: codex/internvl3-svhr-late9-sevenbench-2x3090 @c60a9919112a3cf5bec8c494a533cba56a97f29b
- InternVL旧预测审计归档: 93ff60f839644fabf81f202d699bbc0638484f42
- LLaVA经输入验证的10+10条补充归档: 5ee2a0dbe78df336229b55fed14a4a6d0d880bed

新结果全部位于独立目录，历史输出保持只读。
