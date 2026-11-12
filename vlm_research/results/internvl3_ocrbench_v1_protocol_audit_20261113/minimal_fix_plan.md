最小修复方案（本轮仅提出，未启动生成）

1. 保留751/1000原始结果并标记ATE_SVH_1TILE_SCORER_ALIGNED；不改分、不改图、不覆盖协议与Checkpoint。它适用于原单图块同条件内部比较，不是作者880的同协议复现。
2. 若仅补齐旧Late9：使用cpu/image_integrity.csv重绑定本地SSD路径，保持旧single-tile、原Prompt、BF16/Eager、128上限和冻结算法。997题所需原图已恢复；先在新卡对0053/0963/0988及少量缺失题进行绑定、原生No-op、物理Cache/位置Smoke，再严格恢复997题；本轮未执行该步骤，不保证跨GPU逐Token一致。
3. 若要对照作者公开配置：另建Native-only版本协议，固定官方VLMEvalKit Commit及模型Revision，使用OCRBench配置max_num=24、448动态块和thumbnail、官方问题后缀、原生chat及Greedy/正常EOS/公开默认max_new_tokens=4096。不是根据成绩选参数。先测1块/多块/最大块数资源Smoke；3090显存不足就记录BLOCKED，不偷偷缩图、降块数或换backend。未获后续正式评测指令前不跑1000题。
4. 固定256视觉Token的现有SVH/Late9不能直接套入多图块；任何将来动态多块联合对照都需独立输入/角色/KV正确性门槛，不能混入本轮或复用旧三题冒充同协议。
5. 评分器原样保留。可直接复用本轮官方原始循环与VLMEvalKit实际class evaluator双重CPU核验，不替换成exact match、ANLS、TextVQA软分或追分规范化。
6. 要解释129分各项贡献，还需要预先冻结、单独授权的分辨率/Prompt控制实验。当前证据只能确定协议不一致，不能承诺修复后880或把全部129分归因于切块。
