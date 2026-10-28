# Phase 3.1 数据与方法来源

- Cross-modal Information Flow in Multimodal Large Language Models，CVPR 2025：[论文](https://arxiv.org/abs/2411.18620)，[官方仓库](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)。本次读取 revision `c9a1111dac4baa175f67716fab473789320187c0` 的 README、ChooseRel CSV；没有运行其完整路径扫描或安装其 LLaVA-NeXT 环境。
- Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings（DyVTE），NeurIPS 2025：[官方仓库](https://github.com/DoubtedSteam/DyVTE)，读取 revision `b4a33966da27c913681754b99b2998a7cfaa4001` 的 README 和 `llava/model/language_model/modeling_llama.py`。官方实现包含 Router，并在退出时切片 hidden_states、position_ids 和 Attention Mask；本次不移植 Router、不使用其固定视觉起始位置或训练环境。
- 本项目沿用官方转换的 LLaVA-1.5-7B 与 Transformers 4.49.0，在本地独立 Decoder 循环中建立固定 K 的物理退出诊断。这里不主张首次提出 Visual Exit，也不把本试验当成 DyVTE 的完整复现。

`GQA_val_correct_question_with_relChooser_ChooseRel.csv` 是上述固定 revision 的原始 CSV 副本。`manifest.json` 在模型运行之前固定 12 条样本：固定种子 20261028，left/right 各 6 条，12 张不同图片，未使用模型结果或本地可用性筛选。

全部图片已存在 `/data/wdy/datasets/MMEB-V2/image-tasks/MMEB/GQA/image_<img_id>.jpg`，没有下载完整 GQA 图片包，也未改动该目录。Manifest 保存每张图片的 SHA256、尺寸与原始行哈希；本地文件尺寸逐一与 CSV 对齐。它们是实际使用的本地 GQA 副本，不声称已与官方 ZIP 内 JPEG 字节做独立比对。图片本身未复制进 Git，复现须提供相同哈希图片，缺图时停止而非换样本。

上游 CSV 名称含 `correct_question`，本身是论文作者预选子集。因此同时报告本次 Full 的全部样本和 Full 候选评分正确的固定子集，不把这 12 条结果推广为完整 GQA 成绩。
