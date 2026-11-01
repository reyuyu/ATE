"""Summarize only the observed feasibility evidence and resource release."""
import argparse
import json
from pathlib import Path
import subprocess

from common import sha256, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    def read(name):
        path = out / name
        return json.loads(path.read_text()) if path.exists() else {}
    download = read("download_validation.json")
    manifest = read("model_manifest.json")
    native = read("native_smoke.json")
    eager = read("eager_smoke.json")
    arch = read("architecture_audit.json")
    cpu = read("cpu_tests.json")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    tested_pids = {read("environment.json").get("pid")}
    tested_pids.update(read(f"environment_{mode}.json").get("pid") for mode in ("sdpa", "eager"))
    tested_pids.discard(None)
    current_pids = {int(row.split(",")[0]) for row in active.splitlines() if row.strip()}
    released = not tested_pids.intersection(current_pids)
    release = {"tested_pids": sorted(tested_pids), "tested_processes_absent": released,
               "current_compute_apps": active,
               "gpus": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
               "other_processes_terminated": False}
    write(out / "gpu_release.json", release)
    if native.get("passed") and eager.get("passed"):
        status = "PASS — Full BF16 + Eager" if eager.get("image_budget") == "official defaults" else "LIMITED — Reduced visual resolution"
    elif native.get("passed"):
        status = "PARTIAL — Full BF16 only"
    else:
        status = "FAIL — Single GPU infeasible" if native.get("exception", {}).get("type") == "OutOfMemoryError" else "INCOMPLETE — See exception log; hardware infeasibility not established"
    validation = {"status": status, "download_passed": download.get("passed", False),
        "cpu_passed": cpu.get("passed", False), "native_passed": native.get("passed", False),
        "eager_passed": eager.get("passed", False), "static_audit_complete": bool(arch), "gpu_released": released,
        "no_ablation_no_training_no_quantization_no_offload": True,
        "next_stage": "Basic source-intervention adaptation is feasible, not yet implemented/validated; long contexts and full attention-matrix retention remain untested." if eager.get("passed") else "Do not start source ablations; resolve feasibility limits first."}
    write(out / "validation.json", validation)
    report = ["# Qwen3-VL-8B-Instruct 单卡可行性验证", "", f"**{status}**", "",
        "本轮仅做原生推理和静态结构检查，不运行 Attention 消融、240题扫描或准确率评测。", "",
        "## 模型与环境", f"- 官方模型：`{manifest.get('repo_id')}`。", f"- HF revision：`{manifest.get('revision')}`。",
        f"- 本地绝对路径：`{manifest.get('local_path')}`。",
        "- HF直连不可达；固定HF元数据及非权重文件经hf-mirror.com传输，Git blob/LFS身份逐项核验。四个权重分片来自官方ModelScope固定提交，SHA256与固定HF清单逐项相同；实际下载文件再次校验。不是第三方量化或转换模型。",
        f"- 完整性：{download.get('passed', False)}；权重字节：{manifest.get('weight_bytes')}；文件：{download.get('files')}；Tensor：{download.get('tensors')}；dtype：{download.get('dtype_counts')}。",
        "- 新环境独立于LLaVA；实际依赖见environment_sdpa.json/environment_eager.json及requirements-actual.txt。",
        f"- 物理GPU：{native.get('physical_gpu')}；UUID：`{native.get('gpu_uuid')}`；进程内仅cuda:0可见；GPU1未使用。", "",
        "## 实测结果", "| 模式 | 完成案例 | 加载 | 首Token | 生成 | CUDA allocated峰值 MiB | reserved峰值 MiB | nvidia-smi采样进程峰值 MiB |",
        "|---|---:|---|---|---|---:|---:|---:|"]
    for label, result in (("原生SDPA/BF16", native), ("显式Eager/BF16", eager)):
        report.append(f"| {label} | {len(result.get('cases', []))} | {result.get('load_passed', False)} | {result.get('first_token_forward_passed', False)} | {result.get('generate_passed', False)} | {result.get('peak_allocated_mib', 0):.1f} | {result.get('peak_reserved_mib', 0):.1f} | {result.get('nvidia_smi_sampled_process_peak_mib', 0)} |")
        if result.get("exception"):
            report.append(f"\n异常：`{result['exception']['type']}: {result['exception']['message']}`。完整堆栈见对应JSON。")
    report += ["", "每模式使用一个本地真实图片问答及六类GQA各2题（12道真实问题），样本按固定输入哈希选择，不按模型结果筛选。",
        "全程官方Chat Template、batch1、BF16、Greedy、max_new_tokens32、正常KV Cache、无Offload/量化；未请求或保存全层Attention矩阵。",
        "时间是同步后的端到端墙钟时间：第一Token回调为Prefill，后续回调为Decode，包含生成与测量开销。显存采样可能漏掉短暂峰值，CUDA allocator峰值另列。具体文本、Token、输入长度和逐题时间见native_smoke.json/eager_smoke.json。", "",
        "## 架构与输入映射",
        f"- 实际语言Decoder：{arch.get('language_decoder_layers')}层；Q heads={arch.get('query_heads')}，KV heads={arch.get('kv_heads')}，GQA={arch.get('grouped_query_attention')}。",
        "- Attention路径见architecture_audit.json；Q/K带RMSNorm。视觉编码器与语言Decoder分别配置并验证SDPA/Eager。",
        f"- DeepStack在视觉编码器{arch.get('deepstack_vision_extraction_layers')}层提取，经独立Merger后，在语言Decoder{arch.get('deepstack_language_injection_after_layers')}层输出之后的视觉位置附加注入；两种层编号不可混淆。",
        "- 视觉Token数由实际image_grid_thw和空间merge决定，不是固定576。Question通过实际展开文本的offset映射；P−1为最后有效输入Token，不等于最后一个Question Token。",
        "- 保留实际特殊Token、Question、模板/指令、自身位置及M-RoPE坐标。配置BOS不保证实际出现在输入，禁止把位置0直接当BOS。", "",
        "## 结论与边界", validation["next_stage"],
        "即使Eager通过，也只证明本轮短上下文、小图片下的基础技术可行性。后续来源Hard Mask必须适配动态视觉Token、GQA、原生因果Mask、三轴M-RoPE、KV Cache和DeepStack；本轮未实施，不能声称已验证干预算法。",
        "不能从这些技术Smoke推断正式GQA Accuracy，也不能保证长上下文或全层Attention矩阵提取能在24GB运行。", "",
        f"GPU释放确认：{released}。未终止其他用户进程；不修改main和LLaVA环境。"]
    (out / "report_zh.md").write_text("\n".join(report) + "\n")
    hashes = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_sha256.json"}
    write(out / "artifact_sha256.json", hashes)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
