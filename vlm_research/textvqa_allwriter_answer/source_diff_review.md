# 源码与作用域审计

本模块直接加载固定revision的原生LLaVA模型，不调用历史C3/Middle的
`generate`、FastHandoff、ResearchWriter或AnswerWriter执行路径。历史目录仅用于
CPU读取冻结输入、模型权重摘要和归档质量记录；旧计时字段不进入新生成或新CSV。

- N：原生Dense模型逐步Greedy，H1 LM Head，匹配历史Native；六题同时与HF
  `generate`全部有效Logits严格比较，并完整复现原答案及首Logits哈希。
- A：self_attn前置Hook仅克隆原Mask，将原始P−1/真实Decode这一行的576个
  Image Key列置为`-inf`。所有其他Mask元素用`torch.equal`验证保持原值。
  每层实际注意力权重验证禁用概率严格0，R→Image实际概率保持开放。
- B：同一Mask；仅在L2–L5的o_proj后置Hook将原始P−1或单个真实Decode行置零。
  Prefill其余所有行与原输出逐位相等。没有跳过QKV、Softmax、AV、o_proj或KV计算。
- P−1定义始终是冻结Prompt最后位置；真实Decode由独立Cache长度和原始绝对位置
  验证，包含实际EOS预测。原生R包含图像结束之后、P−1之前的问题和模板。
- 32层原生Decoder、Norm、MLP、Residual和RoPE实现未替换。Hook逐层核对完整
  Query数量、32层Block/MLP/两次Norm调用数、Cache全长和原始位置；Smoke验证
  N/A/B每层P−1之前全部Hidden严格一致、视觉特征一致和RoPE一致。
- 每请求重新进行原生Vision并从空Native DynamicCache构建，不共享生成前缀或
  Cache。Smoke持有两份Cache并逐层比较底层Key/Value存储地址确实不同。

源码冻结记录含本模块推理文件、公共原子JSON/摘要函数、固定官方评分器及已安装
Transformers LLaVA/Llama/Processor源文件。原样本Manifest、每题输入Tensor哈希、
全模型文件哈希和协议另行冻结，逐请求Checkpoint绑定其摘要。

独立加载器不包含计时。首次启动遗漏历史`vlm_common.py`模块级设置的
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，在CLIP首个cuBLAS操作前被确定性检查拒绝。
失败文件保留；补齐相同历史环境后通过全部Smoke，不关闭确定性算法或放宽门槛。

旧All-Writer的300题与本队列规模/输入不同，不当作500题A基线。
原物理C3-V2只在L0–L19允许R读图且L20退出Image Hidden，与Dense B不等价。
本轮不运行物理路径、测速、FLOPs或其他层组合。
