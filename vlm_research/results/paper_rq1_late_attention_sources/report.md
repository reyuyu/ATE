# Late-Layer Attention Source Decomposition

状态：三卡调度已获授权，正式结果尚未完成。

冻结200图及四个50图逻辑分片不变，GPU1负责分片0/3，GPU2负责分片1，GPU3负责分片2。Full与干预固定同图同卡；GPU0完全不参与。旧四卡尝试及Smoke已归档，不混入新绑定。重新完成CPU验证、冻结代码和三卡Smoke后才运行正式采集。详见three_gpu_amendment.json。
