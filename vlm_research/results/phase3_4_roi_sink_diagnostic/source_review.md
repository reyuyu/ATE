# Phase 3.4 文献与官方映射核对

四篇原文已读取，URL和PDF SHA256见references.json；SAGE更准确为Findings of EMNLP2026（arXiv作者备注），而非本次独立核实了EMNLP主会正式卷。

## Cross-modal Information Flow

官方revision c9a1111：InformationFlow.py第85–87行将object1/2的xywh转xyxy，第94行用create_mask_with_bbox构造掩码并经相同process_images；utils.py第167行用RGB绿底红框填充，PIL rectangle端点为包含式。InformationFlow.py第184–186行只要Patch任一像素匹配纯红就选中。blockdesc2range_patches在第314行起将对象Patch索引加实际图像Token起点，“Image Without Central Object”排除对象和Pad。utils.py prepare_image_patch_bbx按unfold及flatten保留行优先顺序，并处理其Pad/AnyRes/HD布局。

本项目为HF固定LLaVA1.5单图中心裁剪，不能照搬其Pad/HD路径。连续半开几何框在真实整数Resize和floor Center Crop后取正面积交集，记录每Patch覆盖比例及center-inside的敏感性视图。与官方RGB插值/纯红阈值/PIL端点的定义存在细小边界差异，明确记录而不冒称逐Patch严格复现。240图重建pixel_values均与真实Processor逐元素相同；真实CLIPVisionEmbeddings CPU测试验证卷积flatten的r*24+c顺序。

## See What You Are Told

ICLR2025，Seil Kang等。第4.1节按raw pre-layer hidden特定维度相对于RMS的激活phi定义Sink；文中LLaVA1.5-7B/LLaMA2的维度为1415/2533，tau=20。文中发现和VAR收益属于其评价协议，不能预先转为本阶段结论。这里额外记录相同signed phi、K/V投影范数和归一化后最大绝对激活，用于检查持续Attention热点是否伴随异常激活；phi不参与改写本轮冻结的候选定义。

## SAGE

Jeonghyo Song、YoungJoon Yoo，arXiv2610.11469v1。第3节/图3以top5%视觉Attention集合的跨Prompt IoU观察prompt-invariant sinks，报告中层条件性、早晚层持续热点及谱集中；其ROI可由CLIP/ViT/DINOv3产生，并用logit bias引导。这里使用真实对象框；中性Prompt的持续/显著阈值是独立预注册诊断规则（top58且5倍均匀基线、8/32层、3/4视图），不是SAGE算法或新方法，也不执行结果驱动的层扫描。

## When Sinks Help or Hurt

Choi等，ECCV2026。原文区分ViT-emerged与LLM-emerged sinks，描述全局先验与局部证据的任务依赖权衡，并研究Key缩放及训练LSG。本轮没有足够观测将热点可靠归类为这两个来源，不训练/使用Gate，不把所有背景或高Attention默认当作有害噪声。只对冻结候选的定向Mask效应作有限陈述。

## 本轮判定边界

Attention统计只建立相关性。Mask Knockout改变softmax归一化和后续计算；其效应属于指定Query/Keys/窗口这一复合干预，不是单Token固有贡献。Physical Exit同时移除视觉状态和后续视觉计算，不能由部分Mask效应直接等价解释。主48图与五个历史结果驱动案例分开；区域面积不同的ROI与Sink不能按原始损失直接排名，各自对照同数量、尽量匹配位置分布的三随机种子背景。空间匹配不足和无候选必须报告，不能后调阈值。
