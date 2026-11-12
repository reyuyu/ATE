# 18条件预检与来源映射

GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6，RTX3090。原30题GQA15+TextVQA Paper-OCR15，未重抽样；LLaVA全部输入Tensor哈希与原冻结记录一致。仅计时，不运行质量Benchmark。

|ID|Backbone|方法|实现|源码Commit|配置|当前状态|
|---|---|---|---|---|---|---|
|L01|LLaVA-1.5-7B|Native Full|e2e32_runtime.generate Native|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|576 visual; unmodified HF 4.49|SMOKE_PASS_FORMAL_RUNNING|
|L02|LLaVA-1.5-7B|SVH|MiddleWriter cutoff20|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|writer0..19; exit20; zero visual KV; skip2..5|SMOKE_PASS_FORMAL_RUNNING|
|L03|LLaVA-1.5-7B|SVH-R|MiddleWriter cutoff25 keep11..24|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|writer0..19; exit25; keep11..24; skip2..5|SMOKE_PASS_FORMAL_RUNNING|
|L04|LLaVA-1.5-7B|FastV|FastV K3 R2/3 staticKV|d1659729b5bf1be225e99ee15783deeea80f63b1|576->192 enteringL3|SMOKE_PASS_FORMAL_RUNNING|
|L05|LLaVA-1.5-7B|SparseVLM|OfficialCompat SparseVLM|734d9003879689713a132b83297dfad060a5c9af|576/309/204/113; mean206.5|SMOKE_PASS_FORMAL_RUNNING|
|L06|LLaVA-1.5-7B|V²Drop|V2Drop frozen hf_adapter|0c94eaa451a6000e2a7e0bec5570278a1de12292|576->288->144->1 before3/17/22; not paper192|SMOKE_PASS_FORMAL_RUNNING|
|L07|LLaVA-1.5-7B|ApET|OfficialCompat ApET|70cc01b215f4d74c5661c699919fe3aeda48bed0|288->96 entering16; mean192|SMOKE_PASS_FORMAL_RUNNING|
|L08|LLaVA-1.5-7B|DUET-VLM (C+all)|OfficialCompat DUET|cd69e489f39a2a51e3b278a470dc8dc32c00e0fb|C+all 307->153->0 before16/24; mean191.75|SMOKE_PASS_FORMAL_RUNNING|
|L09|LLaVA-1.5-7B|FastV+SVH|fastv_svh.engine.Fusion|7570ded8b431ea53accbb349ef51c04b94a4faf7|576->192 before3; exit20; zero visualKV; skip2..5; selector extra QK charged|SMOKE_PASS_FORMAL_RUNNING|
|L10|LLaVA-1.5-7B|SparseVLM+SVH|sparsevlm_svh.engine.Fusion|95a37ae3fbdbb1aae08cfbe2427c04b4619d094a|576/309/204/113 prefix; exit20; zero visualKV; skip2..5; original/hybrid RoPE|SMOKE_PASS_FORMAL_RUNNING|
|L11|LLaVA-1.5-7B|V²Drop+SVH|fusion_engine.PrefixSVH|453f96fc5bf061ea51e6e1f5eab705b7144fcc6f|L3/L17 prefix576->288->144, exit20; noL22 stage; zero visualKV; skip2..5|SMOKE_PASS_FORMAL_RUNNING|
|L12|LLaVA-1.5-7B|ApET+SVH|code/apet_fusion.ApETFusion|NEW_ADAPTER_FROM_PINNED_OFFICIAL_AND_SVH|288->96 at16; exit20; zero visualKV; skip2..5; official compact RoPE retained upon SVH exit|SMOKE_PASS_FORMAL_RUNNING|
|Q01|Qwen3-VL-8B-Instruct|Native Full|unmodified HF Qwen3VL|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; writer0..21; exit22; skip1..6; R keep19..35 incl charged causal-prefix visual KV construction|CPU_SOURCE_AUDITED_GPU_PENDING|
|Q02|Qwen3-VL-8B-Instruct|SVH|EarlyAlignedExecution, frozen cuBLASLt descriptor alignment plus transient zero-key/query alignment|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; writer0..21; exit22; skip1..6; R keep19..35 incl charged causal-prefix visual KV construction|CPU_SOURCE_AUDITED_GPU_PENDING|
|Q03|Qwen3-VL-8B-Instruct|SVH-R|qwen3vl_svhr_search.physical.Physical, late causal prefix KV builder|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; writer0..21; exit22; skip1..6; R keep19..35 incl charged causal-prefix visual KV construction|CPU_SOURCE_AUDITED_GPU_PENDING|
|I01|InternVL3-8B|Native Full|official remote-code InternVL with base embeddings|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; single448 block/256 visual; writer0..19; skip2..5; SVH exit20; SVHR keep19..27 and terminalexit28; frozen numerical waiver retained|CPU_SOURCE_AUDITED_GPU_PENDING|
|I02|InternVL3-8B|SVH|internvl3_svh_physical_d_pilot.engine.Control Physical-D|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; single448 block/256 visual; writer0..19; skip2..5; SVH exit20; SVHR keep19..27 and terminalexit28; frozen numerical waiver retained|CPU_SOURCE_AUDITED_GPU_PENDING|
|I03|InternVL3-8B|SVH-R|internvl3_svh_r_terminal_windows_textvqa.engine.Control late9|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; single448 block/256 visual; writer0..19; skip2..5; SVH exit20; SVHR keep19..27 and terminalexit28; frozen numerical waiver retained|CPU_SOURCE_AUDITED_GPU_PENDING|

模型固定Revision与本地存储见protocol.json及resources/*_model_integrity.json。旧GPU延迟只作协议参考，绝不填入本轮。SparseVLM首轮Smoke只因路径缺失失败，原诊断保留；路径修复后PASS。ApET+SVH新适配器通过物理/Dense前8步数值门槛；不将其冒充尚未定位的历史质量Runner。Qwen/InternVL尚未做本次GPU Smoke，只有通过后才能正式计时。
