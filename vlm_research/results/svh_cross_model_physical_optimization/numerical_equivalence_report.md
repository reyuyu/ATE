# 数值与输出回归

门槛在正式GPU前冻结于protocol.json。B/C要求有效Logits逐位一致，D采用固定BF16容差；没有放宽失败门槛。

|Backbone|Method|阶段|样本数|PASS数|精确Logits数|精确Token数|最大Logits差|
|---|---|---|---|---|---|---|---|
|InternVL / refine_screen|SVH|E|30|30|30|30|0.000|
|InternVL / screen|SVH|B|30|30|30|30|0.000|
|InternVL / screen|SVH-R|B|30|30|30|30|0.000|
|InternVL / screen|SVH-R|C|30|30|30|30|0.000|
|InternVL / screen|SVH-R|D|1|0|0|1|3.500|
|LLaVA / screen|ApET+SVH|B|30|30|30|30|0.000|
|LLaVA / screen|FastV+SVH|B|30|30|30|30|0.000|
|LLaVA / screen|SparseVLM+SVH|B|30|30|30|30|0.000|
|LLaVA / screen|V2Drop+SVH|B|30|30|30|30|0.000|
|Qwen / refine_screen|SVH|E|30|30|30|30|0.000|
|Qwen / screen|SVH|B|30|30|30|30|0.000|
|Qwen / screen|SVH|C|30|30|30|30|0.000|
|Qwen / screen|SVH|D|1|0|0|0|1.625|
|Qwen / screen|SVH-R|B|30|30|30|30|0.000|
|Qwen / screen|SVH-R|C|30|30|30|30|0.000|

Audit Qwen: {"status": "PASS", "checks": [{"method": "SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SVH", "stage": "C", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SVH-R", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SVH-R", "stage": "C", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}]}

首个记录分歧 Qwen: {"step": 0, "layer": 1, "shape": [1, 226, 4096], "torch_equal": false, "max_abs": 0.015625, "mean_abs": 1.495275455454248e-06, "relative_L2": 7.98498367657885e-05}

Audit InternVL: {"status": "PASS", "checks": [{"method": "SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SVH-R", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SVH-R", "stage": "C", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}]}

Audit LLaVA: {"status": "PASS", "checks": [{"method": "FastV+SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "SparseVLM+SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "V2Drop+SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}, {"method": "ApET+SVH", "stage": "B", "full_tokens_equal": true, "all32_logits_equal": true, "audit_enabled": true}]}

完整32步Token和早期/中间/末步Hidden比较，及真实缓存指针/容量，均为独立请求审计，不混入正式耗时。B/C匹配30题输出回归不等于重跑全量质量Benchmark。D若失败，正式效率与质量绑定均NOT_RUN。
