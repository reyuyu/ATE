"""Collect evidence without importing torch, loading a model, or starting CUDA.

This is deliberately not a fallback inference runner. An unverified author
recipe or absent same-runtime physical port cannot unlock GPU evaluation.
"""

import argparse
import ast
import base64
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone


BASE = "7e18895a6ccd57c4cb3aaedd5d512789e7467b59"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL = Path("/data/lk/ATE-resources/models/qwen3-vl-8b-instruct") / REVISION
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qwen3vl_svh_six_official_benchmarks_2x3090"
HISTORY = ROOT / "results/qwen3vl_c3_3090_official_eval"
MANIFEST = Path("/data/lk/ATE-worktrees/fastv-k3-svh-sixbench-3090/vlm_research/results/fastv_k3_svh_sixbench_3090/sample_manifest.jsonl.gz")
QWEN_PYTHON = "/data/lk/深度学习/.envs/ate-qwen3vl/bin/python"
PLOT_PYTHON = "/data/lk/深度学习/.envs/ate-vlm-research/bin/python"
GPU_UUIDS = ["GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6", "GPU-96bc853f-ceea-2404-c704-c48a3d1ca185"]
COUNTS = {"GQA": 12578, "ScienceQA-IMG": 2017, "TextVQA": 5000,
          "POPE": 9000, "MME": 2374, "MMBench": 4377}
REQUIRED = ("framework_commit", "adapter", "task_prompt", "vision_budget",
            "runtime_version", "attention_backend", "sampling", "effective_eos",
            "max_new_tokens", "scorer", "judge_policy")
SOURCES = [
    ("QwenLM/Qwen3-VL", "96588727e44c78b25ba03ea03b8e12f7e64fd0da", "README.md", "AUTHOR_GLOBAL_ONLY"),
    ("open-compass/VLMEvalKit", "051aa811ec27c494919bda944654de884fd70c41", "vlmeval/vlm/qwen3_vl/model.py", "FRAMEWORK_CANDIDATE_NOT_AUTHOR_PINNED"),
    ("open-compass/VLMEvalKit", "051aa811ec27c494919bda944654de884fd70c41", "vlmeval/vlm/qwen3_vl/prompt.py", "FRAMEWORK_CANDIDATE_NOT_AUTHOR_PINNED"),
    ("open-compass/VLMEvalKit", "051aa811ec27c494919bda944654de884fd70c41", "vlmeval/config.py", "FRAMEWORK_CANDIDATE_NOT_AUTHOR_PINNED"),
    ("EvolvingLMMs-Lab/lmms-eval", "f9b577e76bac2fb5a17f6e2d5f9ad9b69156c292", "lmms_eval/models/simple/qwen3_vl.py", "FRAMEWORK_CANDIDATE_NOT_AUTHOR_PINNED"),
]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(name, rows, fields):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
    else:
        path.write_text(stream.getvalue())


def command(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    return {"command": argv, "returncode": p.returncode,
            "stdout": p.stdout.decode("utf-8", errors="backslashreplace"),
            "stderr": p.stderr.decode("utf-8", errors="backslashreplace"),
            "stdout_raw_sha256": hashlib.sha256(p.stdout).hexdigest()}


def eligibility(recipe, same_runtime_port):
    missing = [key for key in REQUIRED if not recipe.get(key)]
    if not recipe.get("author_task_attestation") or missing:
        return {"status": "BLOCKED_AUTHOR_PROTOCOL", "missing": missing}
    if not same_runtime_port:
        return {"status": "BLOCKED_SVH_OFFICIAL_RUNTIME_PORT", "missing": []}
    return {"status": "ELIGIBLE_FOR_GPU_PREFLIGHT", "missing": []}


def scienceqa_gold_letter(gold, choices):
    if isinstance(gold, bool) or not isinstance(gold, int):
        raise ValueError("Official ScienceQA Gold must be a zero-based integer")
    if not 0 <= gold < len(choices) <= 26:
        raise ValueError("Gold outside the original option list")
    return chr(ord("A") + gold)


def fetch_sources():
    records = []
    previous_path = OUT / "official_source_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else []
    for repo, commit, path, role in SOURCES:
        dest = OUT / "official_sources" / repo.replace("/", "__") / path
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={commit}"
        record = dict(repo=repo, commit=commit, source_path=path, role=role,
                      url=url, retrieved_at=stamp(), local_path=str(dest))
        cached = next((x for x in previous if (x["repo"], x["commit"], x["source_path"]) == (repo, commit, path)), None)
        if cached and dest.is_file() and cached.get("sha256") == sha_file(dest):
            records.append(dict(cached, local_path=str(dest)))
            print(repo, path, "VERIFIED_LOCAL_SNAPSHOT", flush=True)
            continue
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ATE-author-protocol-audit"})
                with urllib.request.urlopen(req, timeout=25) as response:
                    payload = json.load(response)
                data = base64.b64decode(payload["content"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                record.update(status="FETCHED", sha256=sha_file(dest), bytes=len(data),
                              github_blob_sha=payload.get("sha"))
                break
            except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
                record.update(status="UNAVAILABLE", error=str(error))
                if attempt == 0:
                    time.sleep(1)
        records.append(record)
        print(repo, path, record["status"], flush=True)
    write_json("official_source_manifest.json", records)


def source_findings():
    source_root = OUT / "official_sources"
    config = source_root / "open-compass__VLMEvalKit/vlmeval/config.py"
    syntax = ast.parse(config.read_text())
    registration = None
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "Qwen3-VL-8B-Instruct" and isinstance(value, ast.Call):
                registration = {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
    assert registration is not None, "Upstream candidate registration changed; re-audit required"
    findings = dict(
        status="CANDIDATE_FRAMEWORK_EVIDENCE_NOT_AUTHOR_ATTESTATION",
        vlmeval_commit=SOURCES[1][1], registration=registration,
        registration_source=str(config), registration_source_sha256=sha_file(config),
        prompt_mixin_enabled_by_registration=registration["use_custom_prompt"],
        prompt_mixin_cannot_be_assumed_effective=not registration["use_custom_prompt"],
        vllm_per_request_seed_explicit_in_candidate=False,
        author_global_max_output=32768,
        candidate_registered_max_new_tokens=registration["max_new_tokens"],
        task_effective_limits="UNRESOLVED_NOT_ASSUMED_EQUAL_TO_EITHER_DEFAULT",
        lmms_simple_adapter_is_hf_not_official_vllm=True,
        author_framework_version_pin=None, author_vllm_version_pin=None,
        author_task_prompt_attestation=None,
    )
    write_json("framework_candidate_findings.json", findings)
    return findings


def historical_evidence():
    sources = [
        (BASE, "vlm_research/results/qwen3vl_c3_3090_official_eval/protocol_review/report_zh.md"),
        (BASE, "vlm_research/results/qwen3vl_c3_3090_official_eval/official_protocol_audit_zh.md"),
        (BASE, "vlm_research/results/qwen3vl_c3_3090_official_eval/report_zh.md"),
        ("30d3e7a0c9318e71a5951aca9a05b6b78108db72", "vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed/report_zh.md"),
        ("1cd2534f7a79ac7b3c746d17ceb19e823cf53ab3", "vlm_research/results/qwen3vl_c3v2_physical_transfer/report_zh.md"),
        ("fc3f10eabf48a30b05f4e927a00f50bcd20780ad", "vlm_research/results/qwen3vl_c3v2_prefill_av_final/report_zh.md"),
    ]
    records = []
    for commit, path in sources:
        data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT.parent)
        relative = Path(path).relative_to("vlm_research/results")
        dest = OUT / "historical_evidence" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records.append(dict(commit=commit, repository_path=path, sha256=sha_file(dest),
                            local_path=str(dest), scope="HISTORICAL_ONLY_NOT_CURRENT_SCORES"))
    write_json("historical_evidence_manifest.json", records)


def cpu_tests():
    result = command([sys.executable, "-m", "vlm_research.qwen3vl_svh_official_audit.tests"])
    write_json("cpu_tests.json", dict(result, status="PASS" if result["returncode"] == 0 else "FAIL",
                                     scope="CPU_PROTOCOL_GATES_NOT_GPU_SMOKE"))
    if result["returncode"]:
        raise RuntimeError("CPU regressions failed")


def report():
    protocol = json.loads((OUT / "protocol.json").read_text())
    model = json.loads((OUT / "model_manifest.json").read_text())
    dataset = json.loads((OUT / "dataset_manifest.json").read_text())
    findings = source_findings()
    historical_evidence()
    cpu_tests()
    lines = [
        "# Qwen3-VL-SVH 六项作者完整协议审计\n",
        "**状态：BLOCKED_AUTHOR_PROTOCOL / BLOCKED_SVH_OFFICIAL_RUNTIME_PORT。正式生成 0 条，未加载模型、未启动 CUDA。**\n",
        "## 结论\n",
        "没有把旧 HF/Eager/Greedy 队列恢复成所谓官方评测。P0 缺少作者逐任务、固定版本的完整配方；P1 缺少同一官方 vLLM Runtime 的已验证 Physical-Original 端口。按本次停止规则保存阻塞证据，不用旧成绩填补。\n",
        "## 已验证与缺失\n",
        f"- 作者 Qwen3-VL 源码固定为 `{SOURCES[0][1]}`。README 明确为 vLLM，并提及部分任务定制 Prompt 尚需文档；没有给出这六项逐任务框架 Commit、Runtime/Backend 版本、有效输出上限和 OCR 政策的完整锁定。\n",
        f"- VLMEvalKit 候选 `{SOURCES[1][1]}` 注册项：`use_vllm=True`、`use_custom_prompt={findings['registration']['use_custom_prompt']}`、`max_new_tokens={findings['registration']['max_new_tokens']}`、temperature=0.7。README 全局参考上限32768不能直接覆盖任务，框架注册上限16384也不能当作六项已核验配方。不能把当前禁用的 Prompt Mixin 当作实际生效 Prompt。\n",
        "- 该 vLLM SamplingParams 调用未显式传逐请求 seed；需查清 seed=3407 的引擎/请求作用域。lmms-eval 的当前 simple Qwen 入口是 HF，不是同一个官方 vLLM 运行路径。\n",
        "- 现有 Physical-Original 直接使用 Transformers 的模型、Eager Attention、DynamicCache 和自定义 CompactKV。没有通过验证的 vLLM/TP2 端口，不能拿 Native vLLM 对比 Physical HF。当前模型环境没有安装vLLM；未修改共享依赖或下载大型环境。\n",
        f"- 固定模型 `{REVISION}` 共{len(model['files'])}个资产逐文件重算SHA256，均与历史绑定一致；没有加载权重。EOS配置为[151645,151643]。\n",
        "- 历史 `NUMERICAL_EQUIVALENCE_FAILED` 保留。最后修复仅固定短题成功，中位输入仍失败、最长题未完成；本轮没有重新判PASS。若未来同Runtime质量探索获准，其数值限制必须继续明确标注。\n",
        "## 数据盘点（不是作者推理协议通过）\n",
        "|任务|要求题数|本机历史清单题数|引用图片缺失|当前结论|\n|---|---:|---:|---:|---|\n",
    ]
    for row in dataset["rows"]:
        lines.append(f"|{row['benchmark']}|{row['expected_questions']}|{row['local_paper_manifest_questions']}|{row['missing_images']}|作者配方未验证|\n")
    lines.extend([
        "\nPOPE官方三策略标注另有3000×3条，已保存绝对路径和SHA；旧8910题子集不用于本轮9000题。其他任务清单已盘点文件存在性并做小范围图片SHA抽核，**未完成作者输入冻结和全集图片SHA复核**。未沿用LLaVA Prompt、生成上限或Paper-OCR作为Qwen官方输入。MMBench的Judge要求尚未锁定，不能以规则解析替代后宣称官方分数。\n",
        "## 硬门槛与未运行项\n",
        "P0未通过，故未冻结单卡分片或TP2模式；P1官方Native/Physical Smoke、P2最短/中位/P95/P99/最长真实输入显存门槛、P3每任务48题输出验收、P4全集均未运行。CSV空表和空分数表示未测，不表示0分。不能给出Native/SVH得分、退化幅度或Bootstrap区间。\n",
        "历史POPE约71 Token长输出、TextVQA 15552视觉Token/115.33GiB Eager申请、ScienceQA字母/整数Gold错误和NFS父路径阻塞均作为反例保存。ScienceQA零基整数Gold到原始选项字母的CPU回归通过；这不等于运行了作者评分器。因为本轮没有GPU请求，不能说已解决长输出或OOM，也不能判定两卡官方高效Attention一定不够显存。\n",
        "## GPU与存储\n",
        "原四卡主机由hostname、PCI Bus ID和UUID重核。候选GPU2 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`、GPU3 `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`；实际占用0张，GPU0/1其他任务未干扰。前序SparseVLM+SVH 35256条已完成独立复算和本地提交，之后才进入本轮。\n",
        "工作树和输出均在/data的本地ext4，父目录挂载及剩余空间记录于gpu_binding.json。当前磁盘仅约数GiB空余，未清除历史数据、未复制模型，也未自动安装大型Runtime。没有NFS检查点队列和后台GPU轮询。\n",
        "## 交付与后续条件\n",
        "已提供六任务协议矩阵、原始官方源码快照/SHA、历史失败证据、模型/数据清单、CPU门禁测试、Runtime兼容诊断和空结果表。审计图只展示门槛状态，不是质量曲线。\n",
        "要进入正式运行，须先取得可证明的六项作者版本化配方（Prompt、像素/OCR、采样/EOS/上限、Scorer/Judge），再实现并独立验证同Runtime的Physical-Original端口；随后才可解锁最坏输入和48题预检。不自动更换为HF/Eager、不缩分辨率、不恢复旧采样队列。\n",
        "main和历史源码未改。仅本独立分支提交/推送本次真实阻塞证据；完整分数和性能图留空。\n",
        "## 复现\n\n```bash\n",
        f"cd {ROOT.parent}\n",
        f"CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {QWEN_PYTHON} -m vlm_research.qwen3vl_svh_official_audit.audit sources\n",
        f"CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {QWEN_PYTHON} -m vlm_research.qwen3vl_svh_official_audit.audit freeze\n",
        f"CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {PLOT_PYTHON} -m vlm_research.qwen3vl_svh_official_audit.audit report\n",
        f"CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {QWEN_PYTHON} -m vlm_research.qwen3vl_svh_official_audit.audit release\n",
        f"CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {QWEN_PYTHON} -m vlm_research.qwen3vl_svh_official_audit.audit checksums\n```\n",
    ])
    (OUT / "report_zh.md").write_text("".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matrix = np.zeros((len(COUNTS), 5))
    matrix[:, 0] = 1
    fig, ax = plt.subplots(figsize=(10, 4.8))
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(["#eeeeee", "#c95858"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), ["P0 Author recipe", "P1 Runtime", "P2 Worst input", "P3 Output audit", "P4 Full eval"])
    ax.set_yticks(range(len(COUNTS)), list(COUNTS))
    for r in range(len(COUNTS)):
        for c in range(5):
            ax.text(c, r, "BLOCKED" if c == 0 else "NOT RUN", ha="center", va="center", fontsize=9)
    ax.set_title("Qwen3-VL / Physical-Original: Prerequisite Audit (0 Generated Requests)")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"protocol_gate_status.{suffix}", dpi=200)
    plt.close(fig)
    write_csv("protocol_gate_status.csv", [dict(benchmark=b, p0="BLOCKED", p1="NOT_RUN", p2="NOT_RUN", p3="NOT_RUN", p4="NOT_RUN") for b in COUNTS], ["benchmark", "p0", "p1", "p2", "p3", "p4"])
    print(protocol["status"], "report and CPU regressions saved")


def environment():
    value = {"collected_at": stamp(), "hostname": command(["hostname"]),
             "gpus": command(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]),
             "compute_processes": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]),
             "nvidia_smi": command(["nvidia-smi"]),
             "memory": command(["free", "-b"]),
             "disk": command(["df", "-hT", "/data"]),
             "parents": [command(["findmnt", "-T", str(p), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]) for p in [OUT, ROOT, Path("/data/lk"), Path("/data"), Path("/")]],
             "selected_gpu_uuids": GPU_UUIDS, "execution_mode": "NOT_FROZEN_P0_BLOCKED",
             "gpu_requests_started": 0, "model_loaded": False, "cuda_started": False}
    write_json("gpu_binding.json", value)
    code = "import importlib.util,importlib.metadata as m,json; print(json.dumps({x: (m.version(x) if importlib.util.find_spec(x) else None) for x in ['torch','transformers','vllm'] }))"
    env = command([QWEN_PYTHON, "-c", code])
    write_json("runtime_environment.json", env)
    return value


def model_audit():
    expected = json.loads((HISTORY / "model_manifest.json").read_text())["files"]
    records = []
    for name, old in sorted(expected.items()):
        path = MODEL / name
        digest = sha_file(path) if path.is_file() else None
        records.append(dict(path=str(path), bytes=path.stat().st_size if path.is_file() else None,
                            sha256=digest, expected_sha256=old["SHA256"], matches=digest == old["SHA256"]))
    result = dict(model="Qwen/Qwen3-VL-8B-Instruct", revision=REVISION,
                  status="PASS" if all(x["matches"] for x in records) else "FAILED_MODEL_BINDING",
                  files=records, weights_loaded=False)
    write_json("model_manifest.json", result)
    generation = json.loads((MODEL / "generation_config.json").read_text())
    write_json("sampling_manifest.json", {
        "model_generation_config": generation,
        "model_generation_config_sha256": sha_file(MODEL / "generation_config.json"),
        "author_global_reference_only": dict(greedy=False, seed=3407, temperature=0.7,
            top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5, out_seq_length=32768),
        "per_task_effective_sampling": None, "per_request_seed_scope": None,
        "warning": "Global defaults are not a verified six-task effective generation recipe."})
    write_json("prompt_manifest.json", {
        "chat_template_path": str(MODEL / "chat_template.json"),
        "chat_template_sha256": sha_file(MODEL / "chat_template.json"),
        "author_verified_task_prompts": None, "chat_token_ids_sha256": None,
        "warning": "A local official tokenizer does not establish the author task prompt."})
    write_json("processor_and_vision_audit.json", {
        "processor_path": str(MODEL / "preprocessor_config.json"),
        "processor_sha256": sha_file(MODEL / "preprocessor_config.json"),
        "config": json.loads((MODEL / "preprocessor_config.json").read_text()),
        "author_per_task_pixel_budget": None, "resolution_changed": False,
        "cpu_full_input_prescan": "NOT_RUN_P0_BLOCKED",
        "historical_counterexample_not_a_new_measurement": dict(question_id="35389", visual_tokens=15552,
            total_tokens=15591, eager_vision_allocation_gib=115.33),
        "worst_case_gpu_preflight": "NOT_RUN_P0_BLOCKED", "memory_feasibility": "UNTESTED"})
    return result


def dataset_audit():
    counts, paths, samples = Counter(), defaultdict(dict), defaultdict(list)
    with gzip.open(MANIFEST, "rt") as f:
        for line in f:
            row = json.loads(line)
            bench = row["benchmark"]
            counts[bench] += 1
            paths[bench][row["image_path"]] = row["image_sha256"]
            if len(samples[bench]) < 2:
                samples[bench].append(row)
    rows = []
    for bench, expected in COUNTS.items():
        missing = [p for p in paths[bench] if not Path(p).is_file()]
        spot = [{"path": row["image_path"], "expected_sha256": row["image_sha256"],
                 "sha256": sha_file(row["image_path"]) if Path(row["image_path"]).is_file() else None}
                for row in samples[bench]]
        rows.append(dict(benchmark=bench, expected_questions=expected, local_paper_manifest_questions=counts[bench],
            unique_image_paths=len(paths[bench]), missing_images=len(missing), missing_examples=missing[:3],
            image_spot_checks=spot, status="LOCAL_MANIFEST_INVENTORY_ONLY_NOT_AUTHOR_RECIPE",
            full_current_image_sha_audit=False,
            note="POPE 8910 paper rows must not substitute for official 9000." if bench == "POPE" else "No LLaVA prompt or cap reused for Qwen."))
    pope = Path("/data/lk/ATE-resources/benchmarks/phase4_0/sources/pope/output/coco")
    pope_rows = []
    for strategy in ["random", "popular", "adversarial"]:
        path = pope / f"coco_pope_{strategy}.json"
        if path.is_file():
            pope_rows.append(dict(strategy=strategy, path=str(path), sha256=sha_file(path),
                                  questions=sum(bool(x.strip()) for x in path.read_text().splitlines())))
    result = dict(manifest_path=str(MANIFEST), manifest_sha256=sha_file(MANIFEST),
                  rows=rows, official_pope_annotations=pope_rows, author_input_protocol_verified=False)
    write_json("dataset_manifest.json", result)
    return result


def freeze():
    env = environment()
    model = model_audit()
    data = dataset_audit()
    rows = []
    for bench, count in COUNTS.items():
        rows.append(dict(benchmark=bench, expected_questions=count,
            author_repository="QwenLM/Qwen3-VL", author_commit=SOURCES[0][1],
            framework="VLMEvalKit or lmms-eval: not author-pinned per task", framework_commit="",
            adapter="Candidate source only: see official_source_manifest.json", official_split={
                "GQA":"testdev_balanced", "ScienceQA-IMG":"test has_image subset",
                "TextVQA":"validation", "POPE":"COCO three strategies 3000 each",
                "MME":"Classic 14 tasks", "MMBench":"EN Dev20230712 Circular, pending author attestation"}[bench],
            task_prompt_full="", task_prompt_sha256="", chat_token_ids_sha256="",
            vision_budget="UNRESOLVED", ocr_policy="UNRESOLVED" if bench == "TextVQA" else "NOT_VERIFIED",
            inference_runtime="vLLM author global reference", attention_backend="UNRESOLVED",
            sampling="Global reference only, effective task overrides unresolved",
            effective_eos="Model config [151645,151643], task override unresolved", max_new_tokens="",
            scorer="Dataset reference only; author framework/version unresolved",
            llm_judge="UNRESOLVED" if bench == "MMBench" else "NOT_VERIFIED",
            protocol_status="AUTHOR_RECIPE_UNRESOLVED", runtime_status="RUNTIME_INCOMPATIBLE",
            author_recipe_verified=False, local_same_protocol_native_svh=False,
            status="BLOCKED_AUTHOR_PROTOCOL;BLOCKED_SVH_OFFICIAL_RUNTIME_PORT"))
    write_csv("official_protocol_matrix.csv", rows, list(rows[0]))
    engine = ROOT / "qwen3vl_c3v2_physical/engine.py"
    source = engine.read_text()
    runtime = {"status": "BLOCKED_SVH_OFFICIAL_RUNTIME_PORT", "engine": str(engine),
               "engine_sha256": sha_file(engine), "base": BASE,
               "hf_dependencies": [line for line in source.splitlines() if line.startswith("from transformers")],
               "uses_hf_eager_attention": "qmodel.eager_attention_forward" in source,
               "uses_hf_dynamic_cache": "DynamicCache" in source,
               "vllm_port_verified": False, "tp2_port_verified": False,
               "known_dense_physical_numeric_gate": "NUMERICAL_EQUIVALENCE_FAILED",
               "native_svh_different_runtime_comparison_allowed": False,
               "new_gpu_requests": 0}
    write_json("runtime_compatibility.json", runtime)
    protocol = dict(status="BLOCKED_AUTHOR_PROTOCOL", additional_blocker=runtime["status"],
        data_status="INVENTORY_ONLY", generation_count=0, author_recipe_verified_complete=[],
        model_revision=REVISION, base=BASE, methods=["Native Full", "Qwen-SVH Physical-Original"],
        conditions=dict(writer_layers=list(range(22)), physical_exit_before=22,
                        answer_skip_layers=list(range(1, 7)), answer_direct_image_access=False),
        environment_modified=False, gpu_execution_mode="NOT_FROZEN_P0_BLOCKED",
        source_runtime_port_missing=True, deadline_or_budget_reset=False,
        historical_results_reused=False, sampling_overridden=False,
        gates=["P0 author recipe", "P1 shared official runtime", "P2 worst input", "P3 48-sample output audit", "P4 full evaluation"],
        p0=eligibility({}, False), model_binding_passed=model["status"] == "PASS",
        no_model_or_dataset_download=True, no_gpu_inference=True, created_at=stamp())
    write_json("protocol.json", protocol)
    for name in ["native_runtime_smoke.json", "svh_runtime_smoke.json"]:
        write_json(name, dict(status="NOT_RUN_P0_BLOCKED", passed=False, gpu_requests=0,
                              reason="Author per-task recipe and shared-runtime physical port not established."))
    for name, fields in [
        ("memory_preflight.csv", ["benchmark", "input_quantile", "method", "status", "peak_bytes"]),
        ("output_length_preflight.csv", ["benchmark", "question_id", "method", "status", "generated_tokens"]),
        ("paired_comparisons.csv", ["benchmark", "comparison", "delta", "ci95_lower", "ci95_upper", "status"]),
        ("category_diagnostics.csv", ["benchmark", "category", "status", "score"]),
        ("generation_length_audit.csv", ["benchmark", "method", "status", "mean", "p50", "p95", "p99", "maximum"]),
        ("output_anomalies.csv.gz", ["benchmark", "question_id", "method", "anomaly"]),
        ("per_question.csv.gz", ["benchmark", "question_id", "method", "token_ids", "text", "score"]),
    ]:
        write_csv(name, [], fields)
    scores = [dict(benchmark=b, expected_questions=n, completed=0, native_score=None,
                   svh_score=None, delta=None, status="BLOCKED_AUTHOR_PROTOCOL") for b, n in COUNTS.items()]
    write_csv("accuracy_summary.csv", scores, list(scores[0]))
    write_json("validation.json", dict(status="BLOCKED", gpu_smoke_passed=False,
        known_historical_numerical_gate="NUMERICAL_EQUIVALENCE_FAILED", source_unchanged=True,
        official_protocol_verified=False, author_recipe_verified_complete=False,
        model_sha256_passed=model["status"] == "PASS", no_benchmark_score_produced=True,
        missing_is_not_zero=True, full_input_prescan_not_run_due_to_p0=True))
    write_json("gpu_release.json", dict(status="NO_GPU_ALLOCATED_BY_THIS_TASK", gpu_requests=0,
        model_loaded=False, task_owned_gpu_processes=[], selected_gpu_uuids=GPU_UUIDS,
        observation=env["compute_processes"], other_processes_untouched=True))
    print(json.dumps({"status": protocol["status"], "runtime": runtime["status"],
                      "model_sha256": model["status"], "dataset_rows": len(data["rows"])}, indent=2))


def checksums():
    write_json("source_manifest.json", [dict(path=str(path), sha256=sha_file(path))
               for path in sorted(Path(__file__).parent.glob("*.py"))] +
               [dict(path=str(ROOT / "qwen3vl_c3v2_physical/engine.py"),
                     sha256=sha_file(ROOT / "qwen3vl_c3v2_physical/engine.py"),
                     frozen_base=BASE, historical_source_modified=False)])
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha_file(p)}  {p.relative_to(OUT)}\n" for p in files))


def release_audit():
    write_json("gpu_release.json", dict(status="NO_GPU_ALLOCATED_BY_THIS_TASK", collected_at=stamp(),
        gpu_requests=0, model_loaded=False, cuda_started=False, task_owned_gpu_processes=[],
        selected_gpu_uuids=GPU_UUIDS, other_processes_untouched=True,
        gpus=command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"]),
        processes=command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]),
        note="This task never loaded a model. Foreign GPU allocations are recorded, not terminated."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sources", "freeze", "report", "release", "checksums"])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    {"sources": fetch_sources, "freeze": freeze, "report": report, "release": release_audit,
     "checksums": checksums}[args.stage]()


if __name__ == "__main__":
    main()
