# OCR依赖性分类与统计冻结协议

分类对象为全部原300题，不重抽样。分类界面只含Question ID、原问题、按冻结SHA核验的真实图片，以及后续能准确对齐的公开OCR文字标注；不含Gold、模型输出、分数、Attention Mass。历史总体结果及先前报告中的少数案例已知，因此这是本Agent的结果屏蔽复核，不声称独立人类双盲或人工真值。标签在加载逐题比较前冻结，所有不确定项保留。

OCR-Direct：图像明确给出须准确转录的具体字词、数字、姓名或名称；通过空间位置找到一处文字后直接抄录仍为Direct。OCR-Reasoning：必须读字后进行语义推断、数值比较、计算、时间/用途/类型判断或跨对象选择，答案不只是被问文字的转录。仅从商店名称得店名为Direct，从名称推断店铺类型为Reasoning。Non-OCR Visual：颜色、形状、对象、数量、位置等可由非文字外观回答，文字内容不必要。Ambiguous：品牌/车型/商品既可凭logo外观又可凭文字、实际文字不可辨或问题指代/意图不明；不把普通品牌题一律当OCR。明确要求“written/read/printed/spelled/what does ... say”的题，在图片有对应文本区域时可判Direct，即使局部难读；图像根本无可核验目标或意图不明则Ambiguous。是否可读、直接/推理操作和替代视觉线索分别写入逐题证据。所有300幅原图均检查；缩略图不足时回看原尺寸。公开OCR不等于问题依赖标签。

标签与证据由Codex多模态视觉检查并记录，未使用另一个分类器、OCR模型或新LLaVA推理，不声称人类专家共识。允许Ambiguous和低置信度。标签冻结后Gold可描述文字匹配、答案长度，不能再改主要标签。多个可能类别只在evidence中保留；主统计不强行归并。若Non-OCR样本很少，必须突出区间宽与不支持总体机制主因归因。

主组Direct、Reasoning、Combined（前两者）、Non-OCR、Ambiguous。12个全300条件均按组报告，120子集条件另表并明确分母，不与300混算。组内主要效应固定A−B、F−B、G−B、Both16−B、F−G。按ImageSHA整簇配对Bootstrap 20000次，seed=20261105；CI为普通双侧95%点位区间。组内5组×5效应共25项用整簇配对sign-flip随机化p（20000次，加一修正）再Holm校正。主交互Combined/Direct/Reasoning相对Non-OCR在上述5效应差共15项，在全样本图像簇共同重采样保持同图依赖，明确是在观察标签上的效应异质性。组间p使用中心化整簇bootstrap零假设，15项单独Holm；不是标签随机试验，不作OCR因果识别。空组不检验、输出NA并记录比较族实际大小。单图组CI不具有总体推断意义。

长度/数字/首词等均为描述性混杂检查，不训练或拟合分类器。短/长问题按blind question whitespace word count预先阈值≤10/>10；答案用冻结后10个Gold标注的modal原文（频率并列按首次出现）做单词/多词、ASCII数字有/无，绝不按生成长度重新挑样。what开头与其他按原问题第一词。Gold tokenizer长度仅用本地原Tokenizer CPU，不加载任何权重。对每一形式分层都完整报告含空组的类别/样本数，未测图像复杂度不假称控制。

Decode分叉位置1-based，包含EOS；相同序列标unchanged。一方为另一方严格前缀时分叉为shorter_length+1，记录prefix_extension。首Token为原Token ID，第一个内容子词为累计Tokenizer decode第一次含非空白/非special内容的位置（不等于整词），首完整英文词用保存文本的首个ASCII字母连续词作为可审计文本代理，单独记录无英文词。OCR相关片段只能在分类冻结后用官方OCR标注与实际分叉词匹配作为词面代理，不能把不匹配断言Non-OCR；局部人工查看仅描述、不改标签。所有分叉和异常均保留。

案例预先规则：按blind manifest数值QuestionID升序，对要求的七类情形分别取第一例；无可行例明确写无。完整negative_cases保存所有组别所有A/B/F/G/预算主要配对的损害，不按成功挑样。图片仅保留少量案例缩略图和SHA/原路径，300题的blind contact sheets仅私有资源目录。图中所有点真实观测，CI不插值。

本轮CPU-only，CUDA_VISIBLE_DEVICES为空，不请求GPU、不加载完整模型、不生成答案。只在/data写入新结果，历史文件及main不改。全部统计必须在classification_freeze.json出现且验证标签/证据SHA之后运行。
