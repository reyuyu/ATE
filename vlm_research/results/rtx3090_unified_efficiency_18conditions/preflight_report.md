# 18条件最终来源与完成映射

所有18条件通过本次GPU Smoke，并完成30输入×2 Session。旧预检保留在source_snapshots/preflight_initial.md。

|ID|Backbone|Method|Runner / 实现|源码Commit|配置|正式题数S1/S2|状态|
|---|---|---|---|---|---|---|---|
|L01|LLaVA-1.5-7B|Native Full|code/run_llava_v2.py / e2e32_runtime.generate Native|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|576 visual; unmodified HF 4.49|30/30|COMPLETE|
|L02|LLaVA-1.5-7B|SVH|code/run_llava_v2.py / MiddleWriter cutoff20|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|writer0..19; exit20; zero visual KV; skip2..5|30/30|COMPLETE|
|L03|LLaVA-1.5-7B|SVH-R|code/run_llava_v2.py / MiddleWriter cutoff25 keep11..24|5ee2a0dbe78df336229b55fed14a4a6d0d880bed|writer0..19; exit25; keep11..24; skip2..5|30/30|COMPLETE|
|L04|LLaVA-1.5-7B|FastV|code/run_llava_v2.py / FastV K3 R2/3 staticKV|d1659729b5bf1be225e99ee15783deeea80f63b1|576->192 enteringL3|30/30|COMPLETE|
|L05|LLaVA-1.5-7B|SparseVLM|code/run_llava_v2.py / OfficialCompat SparseVLM|734d9003879689713a132b83297dfad060a5c9af|576/309/204/113; mean206.5|30/30|COMPLETE|
|L06|LLaVA-1.5-7B|V²Drop|code/run_llava_v2.py / V2Drop frozen hf_adapter|0c94eaa451a6000e2a7e0bec5570278a1de12292|576->288->144->1 before3/17/22; not paper192|30/30|COMPLETE|
|L07|LLaVA-1.5-7B|ApET|code/run_llava_v2.py / OfficialCompat ApET|70cc01b215f4d74c5661c699919fe3aeda48bed0|288->96 entering16; mean192|30/30|COMPLETE|
|L08|LLaVA-1.5-7B|DUET-VLM (C+all)|code/run_llava_v2.py / OfficialCompat DUET|cd69e489f39a2a51e3b278a470dc8dc32c00e0fb|C+all 307->153->0 before16/24; mean191.75|30/30|COMPLETE|
|L09|LLaVA-1.5-7B|FastV+SVH|code/run_llava_v2.py / fastv_svh.engine.Fusion|7570ded8b431ea53accbb349ef51c04b94a4faf7|576->192 before3; exit20; zero visualKV; skip2..5; selector extra QK charged|30/30|COMPLETE|
|L10|LLaVA-1.5-7B|SparseVLM+SVH|code/run_llava_v2.py / sparsevlm_svh.engine.Fusion|95a37ae3fbdbb1aae08cfbe2427c04b4619d094a|576/309/204/113 prefix; exit20; zero visualKV; skip2..5; original/hybrid RoPE|30/30|COMPLETE|
|L11|LLaVA-1.5-7B|V²Drop+SVH|code/run_llava_v2.py / fusion_engine.PrefixSVH|453f96fc5bf061ea51e6e1f5eab705b7144fcc6f|L3/L17 prefix576->288->144, exit20; noL22 stage; zero visualKV; skip2..5|30/30|COMPLETE|
|L12|LLaVA-1.5-7B|ApET+SVH|code/run_llava_v2.py / code/apet_fusion.ApETFusion|NEW_ADAPTER_FROM_PINNED_OFFICIAL_AND_SVH|288->96 at16; exit20; zero visualKV; skip2..5; official compact RoPE retained upon SVH exit|30/30|COMPLETE|
|Q01|Qwen3-VL-8B-Instruct|Native Full|code/run_cross.py --kind Qwen / unmodified HF Qwen3VL|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; all36 native layers; native dynamic visual tokens; full visual KV; no exit/bypass|30/30|COMPLETE|
|Q02|Qwen3-VL-8B-Instruct|SVH|code/run_cross.py --kind Qwen / EarlyAlignedExecution, frozen cuBLASLt descriptor alignment plus transient zero-key/query alignment|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; writer0..21; exit22; skip1..6; zero persistent image KV; frozen temporary numeric alignment cost charged|30/30|COMPLETE|
|Q03|Qwen3-VL-8B-Instruct|SVH-R|code/run_cross.py --kind Qwen / qwen3vl_svhr_search.physical.Physical, late causal prefix KV builder|f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760|BF16/Eager TF4.57.6; writer0..21; main hidden exit22; skip1..6; ImageKV19..35; causal-prefix visual KV construction fully charged|30/30|COMPLETE|
|I01|InternVL3-8B|Native Full|code/run_cross.py --kind InternVL / official remote-code InternVL with base embeddings|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; single448 block/256 visual; all28 native layers; full visual KV; no exit/bypass|30/30|COMPLETE|
|I02|InternVL3-8B|SVH|code/run_cross.py --kind InternVL / internvl3_svh_physical_d_pilot.engine.Control Physical-D|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; writer0..19; physical exit20; skip2..5; zero persistent image KV; frozen numerical waiver|30/30|COMPLETE|
|I03|InternVL3-8B|SVH-R|code/run_cross.py --kind InternVL / internvl3_svh_r_terminal_windows_textvqa.engine.Control late9|c60a9919112a3cf5bec8c494a533cba56a97f29b|BF16/Eager TF4.49; writer0..19; keepImageKV19..27; terminal hidden exit28; skip2..5; temporary reduction padding fully charged|30/30|COMPLETE|

ApET+SVH独立适配器SHA256：e944b84b906c764881ebef1149f59d7e42ddf94d6b4c2dd89b14ff1cf9bd35be。不绑定未定位的历史质量成绩。
性能告警和限制详见final_report.md；完整代码SHA清单见delivery_source_manifest.json。
