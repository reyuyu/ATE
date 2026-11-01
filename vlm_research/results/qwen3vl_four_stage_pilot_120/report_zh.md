# Qwen3-VL 四阶段Attention快速探索（120题）

## 首页结论

1. **没有观察到完整的LLaVA式四阶段功能结构，仅有部分Mass形态相似。** 最初两层Anchor很低，L7–L10出现高Anchor平台，L25–L35再次升高；但“中层为主要敏感峰”的关键部分未复现。预声明描述性检查为H1/H2/H4满足、H3不满足；这些不是统计检验或拟合的阶段边界。
2. **全局LogP敏感峰在末层L35，不在中层。** Mean Signed Loss为0.061279 nat，图像簇Bootstrap普通95%CI为[0.045644, 0.079734]；Mean Absolute Change为0.061717 nat。中层有L17、L24等局部变化；Top-1翻转率在L17和L24并列最高（各3/120，2.5%）。任务全局峰为ChooseCat L30、LogicalObj L24，另外四任务L35，每任务仅20题，不能确认任务特异性。
3. **早层存在高Anchor与低敏感的描述性重合，但不是普遍规律。** L7–L10的平均绝对变化0.006897 nat，低于预声明中层探查范围L13–L24的0.016035 nat。低Anchor的L1–L6同样低敏感；高Anchor的深层还包含全局敏感峰L35，因此不能把高Anchor当作Attention不必要的充分或必要条件。
4. **历史窗口仅提供初步跨模型低敏感线索。** Qwen L1–L4、L2–L5、L3–L6、L4–L7的240题Accuracy翻转均为0，包括与LLaVA候选相同的L2–L5；但L0–L3损害9题、纠正4题，净降2.083pp。既不是独立确认，也未证明总体非劣效、跨模型普遍安全或真实加速。

## 设计与有效性

从已冻结Qwen 240题中按seed与输入哈希选择六任务各20题，共120个不同Image SHA256簇。全部输入与历史Prompt、Processor、Token布局精确一致。BF16/Eager、Transformers4.57.6、固定模型Revision，只使用物理GPU1。
正式运行120次Native Prefill及4320次单层P−1 o_proj OFF Prefill；没有答案生成、Decode、来源Mask、新窗口或确认集。Native Full只运行一次并复用于36个单层比较。
Start Anchor是第一个真实<|im_start|>用户角色起始符，不是BOS。五来源互斥完备；Prompt=Question+Template，排除Anchor和Self。所有原生概率来自未经改变的官方Eager Softmax，逐层只复制P−1行到CPU。
CPU测试与三题GPU Smoke通过；有效首答案完整词表Logits的No-op/Mass Observer差值均为0。单层只置零目标P−1，其他Prefill Query精确不变，MLP调用、BF16残差加法、DeepStack三次注入及36层KV长度正确。原生Full首Token与历史结果逐题一致。
全部120题Full固定首Token的LogP也与历史值精确一致，最大差值为0。正式执行源码Commit为`56aecd386968c9f431282b7197eedfcfdb81dbd8`，协议SHA256为`5c5f85353061384071866eb864dfd92da7e6a7d6039a44fa1dbae0033b07312b`。240个逐题检查点及全部原始张量SHA256、4,320行干预Logits摘要和预测ID在运行后再次校验，120题无缺失、无重复、无失败。

## 实测层级形态

|实测位置或描述范围|Anchor Mass|关键观测|
|---|---:|---|
|L0|0.001901|Image 0.093713，Prompt 0.750056，Self 0.154300；不是视觉Mass占优|
|L1|0.002584|Image 0.028683，Prompt 0.863924；仍以文本Template/Question为主|
|L7–L10|逐层0.581–0.641|连续高Anchor，但单层绝对LogP变化仍为0.004933–0.008701 nat|
|L13–L24探查范围|均值0.392154|L17/L24各有3次Top-1翻转；不是全局LogP损失峰|
|L25–L35探查范围|均值0.711174|高Anchor再现，但L35也是最大LogP损失层|

这些是对未平滑曲线的核读，不声称真实阶段恰好在上述端点切换。Qwen最初主要读取文本，而历史LLaVA L0的Image Mass约0.655，二者并不具有完全相同的早期视觉分配形态。
36层Anchor Mass与Mean Absolute LogP Change的跨层Pearson为+0.209、Spearman为+0.258，并非整体负相关；层号、Query、模板和任务都可能构成混杂，不能由这些描述性相关性推断机制。
L35的P95绝对变化为0.221732 nat，而Top-1仅翻转2/120；反过来L17的LogP均值较低却有3次翻转。概率幅度、Top-1变化与完整答案Accuracy不能混同。

## 任务敏感峰

|任务|Signed峰层|Mean Signed Loss|Absolute峰层|Mean Absolute Change|
|---|---:|---:|---:|---:|
|ChooseAttr|L35|0.047922|L35|0.047922|
|ChooseCat|L30|0.096950|L30|0.096956|
|ChooseRel|L35|0.046841|L35|0.046841|
|CompareAttr|L35|0.143687|L35|0.145355|
|LogicalObj|L24|0.045738|L24|0.046770|
|QueryAttr|L35|0.046063|L35|0.046063|

## 统计与解释边界

区间为Image SHA256整簇配对Bootstrap，10000次，普通描述性95%区间；题目加权。Median/P90/P95、负Loss及极端样本保留在CSV。不因均值接近0声称每个样本都低敏感，不用CI跨0作安全证明。
未拟合变点、未平滑、未强制四段。预声明的早/中/深探查范围仅用于描述比较，不是真实阶段边界；高Anchor连续段以Mass≥0.5、至少3层记录，完整曲线优先。
LLaVA对照来自已有3000题统计，Qwen是120题子集；模板、精度、架构和样本规模不同。LLaVA历史delta_logp=OFF−Full，此处取负统一为Full−OFF。LLaVA BOS与Qwen Start Anchor并非同一种Token，图不能证明跨架构普遍规律。
**跨模型来源口径对齐：** 历史LLaVA的`prompt_mass`包含P−1 Self，无法从已保存的聚合表中单独扣除。因此Figure3两侧统一画“Prompt + Self”：LLaVA沿用其历史总和，Qwen从本轮逐题数据把Prompt和Self相加后重新整簇Bootstrap，保留协方差；Figure1及Qwen主CSV仍分别展示Prompt与Self。具体源数据为`cross_model_harmonized_mass.csv`，处理记录为`comparison_partition_compatibility.json`。仅增加CPU绘图，不修改任何冻结推理代码或增加前向。
Mass与敏感性的跨层和层内相关均仅描述；层内相关另存CSV，不能以跨层相关推断局部因果。更新范数未在本轮测量，因此不能把高Mass解释为大Attention更新。

## 历史窗口背景（不重新生成）

|窗口|历史ΔAccuracy|损害/纠正|
|---|---:|---:|
|W0 [0, 1, 2, 3]|-2.083pp|9/4|
|W1 [1, 2, 3, 4]|+0.000pp|0/0|
|W2 [2, 3, 4, 5]|+0.000pp|0/0|
|W3 [3, 4, 5, 6]|+0.000pp|0/0|
|W4 [4, 5, 6, 7]|+0.000pp|0/0|
|W5 [16, 17, 18, 19]|+0.000pp|2/2|

历史W5虽然净Accuracy不变，仍有2题损害与2题纠正，不能描述为逐题无损。历史W0还存在2次截断。上述仅为既有240题背景，不是本轮新增生成。本轮没有完整答案Accuracy测量，单层首Token结果也不能替代窗口联合生成结果。

## 产物与资源

Figure1：36层来源Mass（含Self），附六任务来源图。Figure2：Signed/Absolute/Flip三面板及六任务热图。Figure3：LLaVA与Qwen并排对照；全部提供PDF/PNG和源CSV。
正式GPU耗时10.11分钟；CUDA allocated峰值17940.9MiB，reserved 18860.0MiB，nvidia-smi采样峰值19176MiB。
原始Native/单层完整词表Logits及每Head Mass存放于服务器 `/data/lk/ATE-resources/experiments/qwen3vl_four_stage_pilot_120`，路径和SHA256绑定在逐题检查点及CSV，不上传Git。资源释放见gpu1_release.json。
GPU1已释放；结束检查只有27MiB桌面占用，无本轮或其他计算进程。本轮未使用GPU0/2/3、未终止外部进程。CPU回归16项、三题真实GPU Smoke、运行后张量和哈希验证全部通过。
到此停止。当前结果值得保留为“局部低敏感与来源Mass结构存在部分跨模型相似”的探索证据，但不支持完整四阶段的普遍机制解释；本轮不启动扫描扩展、独立确认或其他新实验。
