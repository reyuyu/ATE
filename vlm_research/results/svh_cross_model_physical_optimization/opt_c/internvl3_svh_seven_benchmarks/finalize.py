"""Final CPU-only delivery checks and output-pause diagnosis; never resumes generation."""
from .common import *
import gzip
def main():
    independent=readj(OUT/"independent_artifact_validation.json")
    assert independent["status"]=="PASS" and len(independent["benchmarks"])==7
    assert len(readj(RUN/"completion.json")["publication"]["commit"])==40
    environment=readj(OUT/"environment.json")
    for bench,e in environment["benchmarks"].items():
        assert e["GPU_UUID"]==GPU and e["model_revision"]==REV
        assert e["torch"]=="2.6.0+cu124" and e["transformers"]=="4.49.0"
        assert e["weights_all_gpu"] and e["all_float_weights_bf16"]
        assert e["language_layers"]==28 and e["language_attention"]=="eager"
    for bench in BENCHES:
        s=readj(OUT/bench/"smoke_validation.json")
        assert s["status"]=="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER"
        assert all(x["native_noop_official_logits_strict_equal"] for x in s["checks"])
    # Diagnose the already saved anomalous OCR pair without any new inference.
    bench="OCRBench";qid="0480"
    with gzip.open(OUT/bench/"per_question.csv.gz","rt") as f:
        pair={x["method"]:x for x in csv.DictReader(f) if x["question_id"]==qid}
    assert set(pair)==set(METHODS)
    a,b=pair["native"],pair["physical_d"]
    assert a["input_binding"]==b["input_binding"] and b["eos"]=="True" and b["truncated"]=="False"
    with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:
        sample=next(x for x in (json.loads(line) for line in f) if x["question_id"]==qid)
    compact=lambda x:{k:x[k] for k in ["text","token_ids","score","generated_tokens","max_new_tokens","eos","truncated","input_binding","prompt_sha256","sample_sha256"]}
    diagnosis={"status":"PAUSED_OUTPUT_FORMAL","question_id":qid,"question":sample["question"],"gold":sample["gold"],"native":compact(a),"physical_d":compact(b),"image_sha256":sample["image_sha256"],"actual_prompt_LF_count":sample["prompt"].count("\n"),"actual_prompt_literal_backslash_n_count":sample["prompt"].count("\\n"),"cause_observed":"Physical-D repeated calendar-title phrases (59 tokens) before the correct EOS. Native answered the book title correctly. This is not a cap hit or missing-EOS failure.","structural_and_native_noop_smoke":"PASS with existing numerical waiver","all_final_KV_and_input_bindings":"PASS (independent_artifact_validation.json)","finite_output_guard":"No NaN/Inf exception; worker guard active on every LM forward.","cause_not_established":"The exact internal cause of the wrong/repetitive text is not proven. Do not infer that a protocol bug or a single physical component caused it.","action":"Preserve predictions and stop this benchmark under the frozen repetition rule; no prompt/cap/engine modification and no resumed generation.","full_1000_score":"NOT_REPORTED"}
    writej(OUT/bench/"output_diagnosis.json",diagnosis)
    q=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.used,temperature.gpu,clocks.current.sm,power.draw","--format=csv,noheader"],capture_output=True,text=True,timeout=30,check=True)
    assert GPU in q.stdout and "3090" in q.stdout
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],capture_output=True,text=True,timeout=30,check=True)
    assert not apps.stdout.strip(),("GPU_NOT_IDLE",apps.stdout)
    own=[]
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:cmd=(p/"cmdline").read_bytes()
            except (OSError,ProcessLookupError):continue
            if b"internvl3_svh_seven_benchmarks.worker" in cmd or b"internvl3_svh_seven_benchmarks.controller" in cmd:own.append(int(p.name))
    assert not own,("OWN_GPU_OR_CONTROLLER_PROCESS_REMAINS",own)
    release={"status":"PASS_GPU_RELEASED","checked_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"nvidia_smi":q.stdout.strip(),"compute_processes":apps.stdout.strip(),"own_worker_or_controller_processes":own,"no_model_load_or_CUDA_context_created_by_final_audit":True,"server_not_shutdown":True}
    writej(OUT/"gpu_release.json",release)
    v=readj(OUT/"validation.json")
    assert v["status"]=="PARTIAL" and len(v["completed_benchmarks"])==4
    v.update(independent_artifact_validation="PASS",frozen_runtime_source_recheck="PASS",all_seven_native_noop_official_smokes="PASS",gpu_released=True,final_audit_utc=release["checked_utc"])
    writej(OUT/"validation.json",v)
    report=OUT/"report_zh.md"
    text=report.read_text()
    marker="\n## 最终独立审计与暂停解释\n"
    text=text.split(marker)[0]
    text+=marker+"""
全量完成 4/7 项：ScienceQA-IMG、POPE、MME、MMBench。GQA 保存 772 对，TextVQA 保存 358 对，OCRBench 保存 490 对；这三个子集不报告全量分数、差值或置信区间。

- GQA Q201185759：Physical-D 输出无关元解释，13 Token 后正常 EOS；按冻结短答案规则暂停。
- TextVQA Q34935：Native 转录较长意大利文书名，32 Token 达上限且无 EOS。可能是合理的真实长书名，但仍触发本轮预先冻结的保守规则；这不是 Physical-D 长输出或退化的证据，未事后提高上限。
- OCRBench Q0480：Native 正确输出 The Botanical Garden 2012；Physical-D 输出反复的 2016 Calendar 词组，59 Token 后正常 EOS，未达128上限。按重复规则暂停，完整原文与Token保留。输入/Prompt相同、EOS正确、Native/No-op和结构Smoke通过、最终逐层KV审计通过；重复行为的内部原因尚未确定，未据此放宽门槛或修改机制。

预检样本为冻结全集成员。输入、协议、源码绑定一致的预检预测在全集中复用一次，不重复计分；Smoke生成不计入Benchmark分数。各暂停项的保存数也包含已完成预检成员。

MME Cognition下降10原始分：代码推理−7.5、数值计算−2.5；计数、位置、MME OCR的官方分数保持不变，颜色+5。其余子任务和损害案例均保留，不以总体平均掩盖。未完成TextVQA/OCRBench，故不能给出这两项总体OCR能力损失结论。POPE Random Accuracy的普通95%CI低于0，属于多重描述性比较，未作校正。

最终CPU独立审计覆盖全部保存预测：冻结Question/Gold/Prompt/Image SHA/输入ID Tensor字节哈希、逐方法输入绑定、真实EOS/上限、完整题数及28层实际KV布局/Live字节均通过；九个冻结运行文件SHA256未变。官方数据/Scorer对齐、单图块口径与原严格数值Waiver仍须共同引用，不宣称作者排行榜或NUMERICAL_PASS。

GPU最终释放检查见 gpu_release.json：指定3090无计算进程、无本轮Worker/Controller；不关闭服务器。全部原始检查点与日志留在服务器持久化工作盘，轻量源码/预测/报告通过独立分支同步；实际Commit以最终publication.json和GitHub分支头为准。
"""
    report.write_text(text)
    print("FINAL_CPU_AUDIT_PASS_GPU_RELEASED",flush=True)
if __name__=="__main__":main()
