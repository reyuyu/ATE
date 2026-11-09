# GQA 作者源码补充核对

GQA[作者评测页](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)定义每题完整答案精确匹配，按全部问题平均；推荐使用 balanced Test-Dev。页面链接的完整评测包为784MB，包含本轮不需要的辅助数据，因此未下载。

直接读取页面所链接的作者基线仓库 `stanfordnlp/mac-network` 的 `gqa` 分支，固定 Commit `4f3a052ac9e22f88216ef594f9bad024ba857a3f`。

- [main.py](https://github.com/stanfordnlp/mac-network/blob/4f3a052ac9e22f88216ef594f9bad024ba857a3f/main.py) 的 `avg` 函数逐条比较 `prediction == answer`，再除以完整问题数。文件SHA256：`1105c983d61dc1b728ea16b4ac516befcf1904d61c6699ae5ef40453a5609455`。
- [model.py](https://github.com/stanfordnlp/mac-network/blob/4f3a052ac9e22f88216ef594f9bad024ba857a3f/model.py) 的GQA路径同样执行精确相等、求和及均值；VQA软评分分支不会用于GQA。文件SHA256：`9591bf7c31b632876be27e1edeca6bfd4e721e992123fffd68f2746745882a3b`。

`analysis/gqa_author_crosscheck.py` 仅提取并执行无依赖的作者 `avg` 函数，核对保存的完整预测与独立复算分数，不运行作者模型、不修改历史预测或Gold、不重新生成。ATE冻结的提交转换 `text.rstrip('.').lower()` 保持原样；转换规则与作者的最终exact-match聚合分开记录。

这些补充文件不会更改已冻结的推理协议、初始Scorer或源码绑定。
