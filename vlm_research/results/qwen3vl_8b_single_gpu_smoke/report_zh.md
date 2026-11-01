# Qwen3-VL-8B-Instruct 单卡可行性验证

**PASS — Full BF16 + Eager**

本轮仅做原生推理和静态结构检查，不运行 Attention 消融、240题扫描或准确率评测。

## 模型与环境
- 官方模型：`Qwen/Qwen3-VL-8B-Instruct`。
- HF revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- 本地绝对路径：`/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- HF直连不可达；固定HF元数据及非权重文件经hf-mirror.com传输，Git blob/LFS身份逐项核验。四个权重分片来自官方ModelScope固定提交，SHA256与固定HF清单逐项相同；实际下载文件再次校验。不是第三方量化或转换模型。
- 完整性：True；权重字节：17534339512；文件：16；Tensor：750；dtype：{'BF16': 750}。
- 新环境独立于LLaVA；实际依赖见environment_sdpa.json/environment_eager.json及requirements-actual.txt。
- 独立环境：`/data/lk/深度学习/.envs/ate-qwen3vl`；Python3.11、PyTorch2.6.0+cu124、Transformers4.57.6、Accelerate1.4.0、HuggingFace Hub0.36.0。依赖下载因PyPI直连缓慢切换清华PyPI镜像，固定版本不变。
- 旧LLaVA环境前后全部依赖版本一致，Transformers仍为4.49.0，见llava_environment_unchanged.json。
- 物理GPU：0；UUID：`GPU-a63c6362-dcfd-4a26-c44b-014b40127c89`；进程内仅cuda:0可见；GPU1未使用。

## 实测结果
| 模式 | 完成案例 | 加载 | 首Token | 生成 | CUDA allocated峰值 MiB | reserved峰值 MiB | nvidia-smi采样进程峰值 MiB |
|---|---:|---|---|---|---:|---:|---:|
| 原生SDPA/BF16 | 13 | True | True | True | 16877.4 | 17022.0 | 17338 |
| 显式Eager/BF16 | 13 | True | True | True | 17244.6 | 17680.0 | 17996 |

每模式使用一个本地真实图片问答及六类GQA各2题（12道真实问题），样本按固定输入哈希选择，不按模型结果筛选。
全程官方Chat Template、batch1、BF16、Greedy、max_new_tokens32、正常KV Cache、无Offload/量化；未请求或保存全层Attention矩阵。
时间是同步后的端到端墙钟时间：第一Token回调为Prefill，后续回调为Decode，包含生成与测量开销。显存采样可能漏掉短暂峰值，CUDA allocator峰值另列。具体文本、Token、输入长度和逐题时间见native_smoke.json/eager_smoke.json。

12道真实问题的输入长度为120–455个Token，实际视觉Token为88–425，均使用原始Processor默认设置。两模式均13/13正常EOS，空答案、截断、NaN/Inf和OOM为0；两模式生成Token序列在这些输入上相同，不声称Logits精确相同。

排除第一个预热演示后，12道GQA的平均Prefill墙钟时间为SDPA0.110秒、Eager0.126秒；整段Decode墙钟时间分别为0.041秒、0.049秒。本轮答案很短，以上不是正式吞吐或加速评测。

## 架构与输入映射
- 实际语言Decoder：36层；Q heads=32，KV heads=8，GQA=True。
- Attention路径见architecture_audit.json；Q/K带RMSNorm。视觉编码器与语言Decoder分别配置并验证SDPA/Eager。
- DeepStack在视觉编码器[8, 16, 24]层提取，经独立Merger后，在语言Decoder[0, 1, 2]层输出之后的视觉位置附加注入；两种层编号不可混淆。
- 视觉Token数由实际image_grid_thw和空间merge决定，不是固定576。Question通过实际展开文本的offset映射；P−1为最后有效输入Token，不等于最后一个Question Token。
- 保留实际特殊Token、Question、模板/指令、自身位置及M-RoPE坐标。配置BOS不保证实际出现在输入，禁止把位置0直接当BOS。

## 结论与边界
具备下一阶段适配P−1来源干预的基础技术条件，但来源Mask本身尚未实现或验证，长上下文和完整Attention矩阵保留也未测试。
即使Eager通过，也只证明本轮短上下文、小图片下的基础技术可行性。后续来源Hard Mask必须适配动态视觉Token、GQA、原生因果Mask、三轴M-RoPE、KV Cache和DeepStack；本轮未实施，不能声称已验证干预算法。
不能从这些技术Smoke推断正式GQA Accuracy，也不能保证长上下文或全层Attention矩阵提取能在24GB运行。

## 已知异常与修复
初次原生演示已成功，但后续保存Token字符串时因进程默认编码出现UnicodeEncodeError。仅修改独立模块，显式使用UTF-8读写，并新增C locale下的Unicode序列化CPU回归测试；随后固定新源码并完整重跑13个原生案例和13个Eager案例。初次异常JSON、日志及进程退出记录保留，未更改模型、精度、输入分辨率或数值容差。

GPU释放确认：True。未终止其他用户进程；不修改main和LLaVA环境。本轮到此停止。
