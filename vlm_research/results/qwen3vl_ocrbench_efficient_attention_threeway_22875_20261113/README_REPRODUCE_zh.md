# 冻结 Qwen SVH-R OCRBench1000 复核说明

本目录独立于已完成的Native/Original实验；不改历史目录。原始模型/图片不在GitHub或结果包内，需按既有模型Revision与930原图SHA恢复到本地盘。

正式协议见protocol.json。read_layers=19..35；主视觉Hidden在L22前退出；L1..6答案Attention跳算；辅助图像前缀构造L22..34完整块与L35仅QKV。不得替换为InternVL/LLaVA层集合。

执行器svhr_adapter.py只继承冻结Physical并桥接其KV捕获钩子。sources/中36项基础依赖复用已完成Base实验的7e18895a6ccd57c4cb3aaedd5d512789e7467b59源码快照，新增SVH-R physical.py/engine.py来自GitHub f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760并逐字节核验。两处Original核心SHA相同：a719b590975e6282ad983fcd27b80f2e031c302caf47eaf54d50e1f3e36cca77。L19..35配置来自此前overhead audit已冻结cross_runtime.py，SHA 14a19c24b33f2693d7cd1e69d4d09562b69fda5b6ba8343109fc81ebcb67460c，与旧source_manifest.json一致；不声称这个Controller路径已在f212提交。

完整归档在本地/持续盘与Windows备份，含predictions/1000个原始JSON、物理KV/Mask执行审计、Gate Logits、源文件、冻结输入布局和原注释。GitHub仅放轻量逐题预测、输入/Gold/哈希、CSV、报告与必要源码，勿上传模型或图片。

便携CPU复核：在此目录执行 `python -B verify_light_cpu.py .`，只需Python标准库，无需Torch、模型、图片或网络。它对三份轻量预测共3000条逐题复算OCRBench v1得分，核对相同Gold和全部输入哈希；结果写入light_verification_result.json。Gold保存在预测文件中，inputs_light.jsonl不含Gold。

完整CPU独立复核：按本说明中的服务器布局恢复后，在既有qwen3vl-torch260-tf4576环境执行 `python -B final_audit.py`；它只读取预测/Tokenizer/原注释，不加载GPU模型，另外核对Token解码、真实KV与物理执行审计。脚本包含首次终态保护；再次审计应在独立副本上调整状态/路径，禁止改写原证据。SOURCE_SHA256.json对应完整服务器归档；GitHub只保留可核验的轻量子集。

GPU脚本 `run.py` 使用逐题原子保存、fresh KV及断点读取；已完成1000题，禁止直接重跑。重新复现需新的独立目录，保持protocol/所有输入Tensor哈希不变。依赖版本Torch2.6.0+cu124、Transformers4.57.6、BF16，单线程TF32关闭；只启用SDPA_EFFICIENT_ATTENTION，禁用MATH/FLASH/CUDNN。

严格数值门禁FAIL和对应警告完整保留。Score不是Eager完整对照、不是数值等价证明，三组同一输入/新后端才可配对比较。既有base预测和archive SHA均未改变。

归档与Windows真实SHA核验分别见archive_receipt.json、windows_backup_receipt.json；GitHub真实Commit以发布后独立回执为准。GPU释放见gpu_release.json，未关闭服务器。
