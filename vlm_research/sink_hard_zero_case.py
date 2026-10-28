"""Case15133706 only: frozen layer scan of exact Sink-key zeroing.

prepare -> CPU checks / frozen protocol -> gpu (resource gated) -> summarize.
No automatic GPU polling. Attempts and completed condition checkpoints are kept.
"""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch
import transformers

from attention_inheritance import TOLERANCES
from attention_inheritance_pilot import native_signature
from attention_sink_hard_zero import VisualSinkHardZero
from attention_sink_steering import conditional
from attention_sink_steering_case import regional_rows
from knockout_probe import model_layers, resource_preflight
from roi_sink_common import forward
from roi_sink_probe import inputs_and_history, score
from vlm_common import ROOT, load_model

QID = "15133706"
SOURCE = ROOT / "results/attention_sink_steering_case_15133706"
P34 = ROOT / "results/phase3_4_roi_sink_diagnostic"
P32 = ROOT / "results/phase3_2_fixed_k_exit"
OUTPUT = ROOT / "results/sink_hard_zero_layer_scan_15133706"
TESTS = ["test_attention_sink_hard_zero", "test_attention_inheritance", "test_attention_sink_steering"]
WINDOWS = [("early",2,5),("middle_old",8,16),("middle",14,24),("deep",24,32)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def make_conditions(entry):
    targets=[("sink",entry["sink_nonroi_indices"],None)]
    for seed in [3401,3402,3403]:
        c=entry["random_controls"][f"random_sink_s{seed}"]
        if not c["available"] or not c["exact_count"]:
            raise ValueError("Frozen count-matched control unavailable")
        targets.append((f"random_s{seed}",c["indices"],seed))
    scopes=[(f"L{l:02}","single",[l]) for l in range(32)]
    scopes += [(name,"window",list(range(start,stop))) for name,start,stop in WINDOWS]
    rows=[]
    for scope,kind,layers in scopes:
        for target,indices,seed in targets:
            if len(indices)!=13 or len(set(indices))!=13:
                raise ValueError("Expected exactly 13 unique blocked patches")
            if target!="sink" and set(indices)&(set(entry["roi_indices"])|set(entry["sink_indices"])):
                raise ValueError("Control overlaps ROI or frozen Sink")
            rows.append(dict(id=f"{scope}_{target}",scope=scope,kind=kind,target=target,
                             control_seed=seed,active_layers=layers,patch_indices=indices))
    return rows


def cpu_checks(path):
    result=subprocess.run([sys.executable,"-m","unittest",*TESTS,"-v"],cwd=ROOT,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
        capture_output=True,text=True)
    Path(path).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f"CPU tests failed: {path}")


def validate(output):
    p=read(output/"protocol.json")
    if p["conditions"]!=make_conditions(p["sink_entry"]) or p["tolerances"]!=TOLERANCES:
        raise ValueError("Frozen conditions/tolerances changed")
    for name,expected in {**p["source_sha256"],**p["code_sha256"]}.items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f"Frozen source/code changed: {name}")
    if native_signature()!=p["library"]:
        raise ValueError("Installed Transformers source changed")
    if sha(p["sample"]["image_path"])!=p["sample"]["image_sha256"]:
        raise ValueError("Image changed")
    return p


def committed_code(protocol):
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for name,expected in protocol["code_sha256"].items():
        content=subprocess.check_output(["git","show",f"{commit}:vlm_research/{name}"],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError(f"GPU execution requires frozen code committed first: {name}")
    return commit


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"protocol.json").exists():
        raise ValueError("Already prepared; do not overwrite frozen protocol")
    old=read(SOURCE/"protocol.json")
    frozen=read(SOURCE/"artifact_sha256.json")["files_sha256"]
    if not all(sha(SOURCE/k)==v for k,v in frozen.items()):
        raise ValueError("Historical result changed")
    cache=P34/"attention_rows"/f"{QID}_task.npz"
    inventory=read(P34/"artifact_sha256.json")["files"]
    if sha(cache)!=inventory[str(cache.relative_to(P34))]["sha256"]:
        raise ValueError("Full attention cache changed")
    sample,entry=old["sample"],old["sink_entry"]
    assert sample["question_id"]==QID and sample["answer"]=="right"
    assert sha(sample["image_path"])==sample["image_sha256"]
    assert len(entry["sink_nonroi_indices"])==13 and entry["roi_sink_overlap_indices"]==[364]
    assert not set(entry["sink_nonroi_indices"])&set(entry["roi_indices"])
    if transformers.__version__!="4.49.0":
        raise ValueError("Keep pinned Transformers 4.49.0")
    conditions=make_conditions(entry)
    source_files=[SOURCE/"protocol.json",SOURCE/"artifact_sha256.json",cache,P34/"manifest.json",
                  P34/"sink_manifest.json",P32/"manifest.json",P32/"per_sample_results.csv",
                  P32/"checkpoints/run"/f"{QID}.json",ROOT/"configs/model.json"]
    protocol=dict(question_id=QID,created=now(),sample=sample,sink_entry=entry,conditions=conditions,
        condition_count=len(conditions),query="Only dynamic P-1; all 32 heads",
        formula="r=a_visual; r[blocked]=0; new_visual=sum(a_visual)*r/sum(r)",
        placement="Native FP32 softmax cast to FP16, then zero/visual-only renormalize before P@V",
        zero_support="Fail if original visual mass is positive but remaining support is zero; zero mass stays zero",
        controls="13 frozen non-ROI Sink keys and three frozen spatial/count-matched backgrounds; same keys across all layers/windows",
        overlap="ROI/Sink overlap patch364 is spared; no claim of removing all Sink candidates",
        planned_scopes="All 32 individual layers plus L2-4/L8-15/L14-23/L24-31; fixed before any new prediction",
        tolerances=TOLERANCES,minimum_free_mib=16000,library=native_signature(),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_files},
        code_sha256={f.name:sha(f) for f in sorted(ROOT.glob("*.py"))},
        historical_full_margin=-.03125,historical_physical_k24_margin=.0390625,
        no_op="Empty blocked set, override all 32 layers; P-1 full vocabulary logits and Question hidden must be exact Full",
        gpu_smoke="No-op plus single-layer L14 Sink/random_s3401; cleanup Full before formal conditions",
        interpretation="One selected FP16 weak-margin case; diagnostic full teacher forcing, no training/decode/quantization/speed claim")
    write(output/"protocol.json",protocol)
    write(output/"status.json",dict(status="prepared_cpu_pending",gpu_completed=False,
          completed_conditions=0,planned_conditions=len(conditions),updated=now()))
    cpu_checks(output/"cpu_tests.txt")
    write(output/"status.json",dict(status="cpu_passed_gpu_pending",gpu_completed=False,
          completed_conditions=0,planned_conditions=len(conditions),updated=now()))
    summarize(output)
    print("PREPARED",output,flush=True)


def checked_forward(model, inputs, qpos, apos, modules, visual, patches, active, full):
    with VisualSinkHardZero(modules,visual.cpu(),apos,patches,active) as edit:
        out=forward(model,inputs,qpos,apos);edit.assert_complete()
    if any("forward" in m.__dict__ for m in modules):
        raise AssertionError("Instance wrapper leaked")
    if not torch.equal(out["question_hidden"],full["question_hidden"]):
        raise AssertionError("Causally earlier Question hidden states changed")
    if not patches and not torch.equal(out["prediction_logits"],full["prediction_logits"]):
        raise AssertionError("No-op differs from Full")
    return out,edit


def evaluate_condition(model,candidates,qpos,apos,modules,visual,condition,full,baseline,entry):
    outputs=[];audits=[];pairs=None
    for i,(inputs,records) in enumerate(candidates):
        out,edit=checked_forward(model,inputs,qpos,apos,modules,visual,condition["patch_indices"],
                                 condition["active_layers"],full[i])
        this=np.stack(edit.rows)
        if pairs is None:pairs=this
        elif not np.array_equal(pairs,this):raise AssertionError("Future candidate affected edited row")
        outputs.append(out);audits.append(dict(candidate=i,layers=edit.records,
            question_hidden_bitwise_full=True,measurement=out["measurement"]))
    if not torch.equal(outputs[0]["prediction_logits"],outputs[1]["prediction_logits"]):
        raise AssertionError("Candidate leaked to P-1")
    logps=[score(out,records) for out,(_,records) in zip(outputs,candidates)]
    margin=logps[1]-logps[0]
    regional=regional_rows(pairs,condition["active_layers"],entry,condition["id"])
    row=dict(question_id=QID,condition=condition["id"],scope=condition["scope"],kind=condition["kind"],
        target=condition["target"],control_seed=condition["control_seed"],first_layer=condition["active_layers"][0],
        last_layer=condition["active_layers"][-1],blocked_patches=len(condition["patch_indices"]),
        left_logp=logps[0],right_logp=logps[1],right_probability=float(np.exp(logps[1])),
        margin=margin,full_margin=baseline["margin"],gain_vs_full=margin-baseline["margin"],
        predicted_candidate="right" if margin>0 else "left" if margin<0 else "tie",
        candidate_correct=margin>0,wrong_to_correct=baseline["margin"]<=0 and margin>0,
        historical_physical_k24_margin=.0390625,
        diagnostic_forward_seconds=sum(o["measurement"]["forward_seconds"] for o in outputs),
        peak_allocated_mib=max(o["measurement"]["gpu_memory"]["peak_allocated_mib"] for o in outputs),
        peak_reserved_mib=max(o["measurement"]["gpu_memory"]["peak_reserved_mib"] for o in outputs),
        all_checks_passed=True)
    for region in ["roi","sink_nonroi","overlap"]:
        for stage in ["before","after"]:
            row[f"{region}_{stage}_share"]=float(np.mean([r[f"{region}_share"] for r in regional if r["stage"]==stage]))
    return row,audits,pairs,regional,outputs


def checkpoint(folder,condition,row,audits,pairs,regional,outputs,binding):
    folder.mkdir(exist_ok=False)
    write(folder/"audits.json",audits)
    write_csv(folder/"regional_attention_per_head.csv",regional)
    np.savez_compressed(folder/"rows.npz",attention_before_after=pairs,layers=condition["active_layers"])
    np.savez_compressed(folder/"predictions.npz",
        logits=np.stack([o["prediction_logits"].numpy() for o in outputs]),
        question_hidden=np.stack([o["question_hidden"].numpy() for o in outputs]))
    files={f.name:sha(f) for f in folder.iterdir() if f.is_file()}
    body=dict(binding=binding,condition=condition,row=row,files_sha256=files)
    # Complete marker written last; incomplete folders are never silently reused.
    write(folder/"complete.json",dict(body=body,sha256=digest(body)))


def read_checkpoints(run,protocol,binding):
    folder=run/"conditions";folder.mkdir(exist_ok=True)
    allowed={c["id"]:c for c in protocol["conditions"]};found={}
    for directory in folder.iterdir():
        if directory.name not in allowed or not directory.is_dir():raise ValueError("Unexpected condition checkpoint")
        if not (directory/"complete.json").exists():
            raise ValueError(f"Partial checkpoint retained at {directory}; use a new attempt, do not overwrite")
        envelope=read(directory/"complete.json");body=envelope["body"]
        if envelope["sha256"]!=digest(body) or body["binding"]!=binding or body["condition"]!=allowed[directory.name]:
            raise ValueError("Checkpoint binding mismatch")
        if not all(sha(directory/name)==value for name,value in body["files_sha256"].items()):
            raise ValueError("Checkpoint artifact changed")
        found[directory.name]=body["row"]
    return found


def gpu(output,resume=None):
    protocol=validate(output)
    commit=committed_code(protocol)
    if read(output/"status.json")["gpu_completed"]:raise ValueError("Completed experiment is immutable")
    if resume:
        run=(output/resume).resolve()
        if not run.is_relative_to((output/"gpu_attempts").resolve()) or not run.is_dir():
            raise ValueError("Resume requires an existing attempt within this output")
    else:
        run=output/"gpu_attempts"/datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True,exist_ok=False)
    lock=(output/"gpu.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed={}
    try:
        cpu_checks(run/f"cpu_tests_{datetime.now().strftime('%H%M%S_%f')}.txt")
        resource=resource_preflight(protocol["minimum_free_mib"])
        write(run/f"resource_{datetime.now().strftime('%H%M%S_%f')}.json",resource)
        if not resource["ready"]:
            state=dict(status="gpu_resource_blocked",gpu_completed=False,completed_conditions=0,
                       planned_conditions=len(protocol["conditions"]),latest_attempt=str(run.relative_to(output)),
                       updated=now(),reason=resource.get("reason"))
            # On resumed runs preserve prior completed count even if GPU is unavailable.
            state["completed_conditions"]=len(list((run/"conditions").glob("*/complete.json")))
            write(output/"status.json",state);summarize(output)
            print("GPU_BLOCKED",resource.get("reason"),flush=True);return
        model,processor,env=load_model()
        binding=digest(dict(protocol_sha256=sha(output/"protocol.json"),execution_commit=commit,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            environment={k:env[k] for k in ["python","torch","transformers","cuda_runtime","gpu","model_revision"]}))
        existing=run/"binding.json"
        if existing.exists() and read(existing)["binding"]!=binding:raise ValueError("Resume environment/commit changed")
        if not existing.exists():write(existing,dict(binding=binding,execution_commit=commit,environment=env))
        prompt,candidates,qpos,apos,history,qrecords,counts=inputs_and_history(model,processor,protocol["sample"])
        visual=torch.where(prompt["input_ids"][0]==model.config.image_token_index)[0]
        modules=[layer.self_attn for layer in model_layers(model)]
        assert len(visual)==576 and len(modules)==32 and apos not in qpos and max(qpos)<apos
        write(run/"input_mapping.json",dict(question_id=QID,answer_query=apos,visual_indices=visual.cpu().tolist(),
              question_mapping=qrecords,counts=counts,candidate_mapping=history["candidate_mapping"]))
        full=[forward(model,inputs,qpos,apos,observe=i==0) for i,(inputs,_) in enumerate(candidates)]
        logps=[score(out,rec) for out,(_,rec) in zip(full,candidates)]
        old_logps=[r["logp"] for r in history["full_candidates"]]
        cached=np.load(P34/"attention_rows"/f"{QID}_task.npz")["attention"]
        agreement=dict(logp_differences=[a-b for a,b in zip(logps,old_logps)],
                       fresh_attention_bitwise_cache=bool(np.array_equal(full[0]["rows"],cached)))
        write(run/"historical_full_comparison.json",agreement)
        if logps!=old_logps or not agreement["fresh_attention_bitwise_cache"]:
            raise AssertionError("Same-machine Full differs from frozen history; investigate without relaxing tolerance")
        if not torch.equal(full[0]["prediction_logits"],full[1]["prediction_logits"]):raise AssertionError("Candidate leakage")
        baseline=dict(left_logp=logps[0],right_logp=logps[1],margin=logps[1]-logps[0],
                      historical_physical_k24_margin=.0390625,measurements=[o["measurement"] for o in full])
        write(run/"baseline.json",baseline)
        np.savez_compressed(run/"baseline.npz",logits=np.stack([o["prediction_logits"].numpy() for o in full]),
                            question_hidden=np.stack([o["question_hidden"].numpy() for o in full]))
        smoke=[]
        for i,(inputs,_) in enumerate(candidates):
            out,edit=checked_forward(model,inputs,qpos,apos,modules,visual,[],list(range(32)),full[i])
            smoke.append(dict(condition="all_layer_noop",candidate=i,logits_bitwise_full=True,audits=edit.records,
                              measurement=out["measurement"]))
        for target in ["sink","random_s3401"]:
            condition=next(c for c in protocol["conditions"] if c["scope"]=="L14" and c["target"]==target)
            out,edit=checked_forward(model,candidates[0][0],qpos,apos,modules,visual,
                                     condition["patch_indices"],[14],full[0])
            smoke.append(dict(condition=condition["id"],audits=edit.records,measurement=out["measurement"]))
        clean=forward(model,candidates[0][0],qpos,apos)
        if not torch.equal(clean["prediction_logits"],full[0]["prediction_logits"]):raise AssertionError("Smoke cleanup failed")
        write(run/"gpu_smoke.json",dict(passed=True,checks=smoke,post_cleanup_full_exact=True))
        completed=read_checkpoints(run,protocol,binding)
        for condition in protocol["conditions"]:
            if condition["id"] in completed:continue
            row,audits,pairs,regional,outputs=evaluate_condition(model,candidates,qpos,apos,modules,visual,
                                                               condition,full,baseline,protocol["sink_entry"])
            checkpoint(run/"conditions"/condition["id"],condition,row,audits,pairs,regional,outputs,binding)
            completed[condition["id"]]=row
            ordered=[completed[c["id"]] for c in protocol["conditions"] if c["id"] in completed]
            write(run/"per_condition_results.json",ordered);write_csv(run/"per_condition_results.csv",ordered)
            write(output/"status.json",dict(status="gpu_partial",gpu_completed=False,completed_conditions=len(completed),
                planned_conditions=len(protocol["conditions"]),latest_attempt=str(run.relative_to(output)),updated=now()))
            print(json.dumps(row),flush=True)
        after=forward(model,candidates[0][0],qpos,apos)
        if not torch.equal(after["prediction_logits"],full[0]["prediction_logits"]) or not torch.equal(after["question_hidden"],full[0]["question_hidden"]):
            raise AssertionError("Final cleanup Full changed")
        validate(output)
        assert len(read_checkpoints(run,protocol,binding))==144
        write(run/"validation.json",dict(passed=True,completed_conditions=144,all_local_fp16_audits_passed=True,
            question_hidden_bitwise_full=True,post_cleanup_native_exact=True,library_unchanged=True,binding=binding))
        write(output/"status.json",dict(status="completed",gpu_completed=True,completed_conditions=144,
              planned_conditions=144,latest_attempt=str(run.relative_to(output)),updated=now()))
        summarize(output)
    except BaseException as exc:
        write(run/f"failure_{datetime.now().strftime('%H%M%S_%f')}.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        write(output/"status.json",dict(status="gpu_incomplete",gpu_completed=False,completed_conditions=len(completed),
              planned_conditions=len(protocol["conditions"]),latest_attempt=str(run.relative_to(output)),updated=now(),reason=repr(exc)))
        summarize(output);raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def summarize(output):
    p=read(output/"protocol.json");state=read(output/"status.json")
    run=output/state["latest_attempt"] if state.get("latest_attempt") else None
    rows=read(run/"per_condition_results.json") if run and (run/"per_condition_results.json").exists() else []
    if rows:
        write(output/"per_condition_results.json",rows);write_csv(output/"per_condition_results.csv",rows)
    text=f"""# 15133706：不同层硬置零Sink实验

## 实际状态

**{state['status']}：真实GPU完成 {state['completed_conditions']}/{p['condition_count']} 条件。**
{state.get('reason','')}

历史Full Margin=−0.03125、历史Physical K24=+0.0390625，仅作旧对照。未完成时不得将它们写成新基线或推断任何层的预测收益。

## 冻结设计

仅Girl/Boy案例15133706，原问题/候选left/right，gold right；FP16/eager/Transformers4.49.0、use_cache=False、batch1。动态定位真实P−1和576视觉位置，只改该Query、全部32Head，Question隐藏状态应保持因果不变。

固定32个单层L0～31及四窗口L2～4、L8～15、L14～23、L24～31。每个scope含冻结13非ROI Sink置零，以及三组Phase3.4等数量/空间bin匹配背景置零，共144条件。不扫描强度、不按结果选层/Head/窗口；单层和多层累计窗口分开比较。ROI/Sink交集#364不被抑制，不能称为所有Sink均已清除。

在Softmax转原生FP16后、乘当前V之前：`r=a_visual; r[blocked]=0; new_visual=sum(a_visual)*r/sum(r)`。保留当前Head视觉总质量、非视觉权重、其他Query与因果Mask；零质量保持零，正质量却无剩余支持则停止，无任意fallback。不同于旧Phase3.4整行Mask后Softmax重分配，也不同于×0.1软抑制或L22−L2离线作图。被选Patch必须严格为零。

复用未修改的实例局部eager包装、FP16公式逐元素舍入及dtype推导界。行和atol0.002；视觉Mass atol2e−5+rtol0.001；原conditional L1阈值0.002保留告警，不据结果放宽。先CPU测试，再7B无干预Full/全层no-op、L14硬置零Sink/随机对照Smoke，最后正式评分。每条件两完整候选全词表LogP、Margin、原始logits、Question hidden、区域份额、局部Before/After行、审计和原子完成指纹保存；绝不保存完整Attention矩阵。

## 结果和限制

"""
    if not rows:
        text+="**尚无本轮真实Full或干预分数，不能回答不同层扣除是否改善预测。没有绘制虚构Margin曲线。** GPU不足时不加载7B，不终止其他任务；没有自动轮询。\n"
    else:
        targets=[r for r in rows if r["target"]=="sink"]
        text+="| scope | Margin | Gain vs同机Full | right正确 |\n| --- | ---: | ---: | --- |\n"
        text+="\n".join(f"| {r['scope']} | {r['margin']:+.8f} | {r['gain_vs_full']:+.8f} | {r['candidate_correct']} |" for r in targets)+"\n"
        if state["gpu_completed"]:plot_results(output,rows)
        else:text+="\n以上为partial，不作全层总结。\n"
    text+="""
单个事后选出的弱Margin FP16案例；高Attention不等于有害噪声。转正只是二候选排序变化，须同时报告正确词全词表LogP和三背景对照；无泛化、因果机制已证明或推理加速声明。Hook正常/异常清理，历史代码/权重/结果保持原样。

## 执行/恢复

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
# 已冻结，勿再次prepare覆盖：
# CUDA_VISIBLE_DEVICES='' python sink_hard_zero_case.py --stage prepare
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  python sink_hard_zero_case.py --stage gpu
# 只有该attempt存在有效完成checkpoint且代码Commit/环境相同时：
# python sink_hard_zero_case.py --stage gpu --resume-attempt gpu_attempts/<原目录>
```

固定代码Commit必须先于真实7B运行；运行中的源码不修改。资源门槛16,000MiB空闲。每次gpu默认新建尝试目录，保留失败/资源记录；`--resume-attempt`严格核验冻结协议、代码Commit、环境及逐条件文件SHA，拒绝不完整checkpoint覆盖或重复计数。若原尝试没有分数，直接新attempt即可。结果汇总仅在全部144条件及清理验证通过后画图；partial保留真实分数但不伪装完整。

协议见[protocol.json](protocol.json)，状态见[status.json](status.json)，CPU记录见[cpu_tests.txt](cpu_tests.txt)。只推送独立实验分支，不合并main。
"""
    (output/"report.md").write_text(text)
    (output/"README.md").write_text("# 单案例分层硬置零实验\n\n[真实状态/报告](report.md) · [冻结协议](protocol.json) · [CPU测试](cpu_tests.txt)\n")


def plot_results(output,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets=[r for r in rows if r["target"]=="sink" and r["kind"]=="single"]
    assert len(targets)==32
    x=np.array([r["first_layer"] for r in targets])
    controls=np.array([[next(r["margin"] for r in rows if r["scope"]==t["scope"] and r["target"]==f"random_s{s}")
                        for s in [3401,3402,3403]] for t in targets])
    fig,ax=plt.subplots(figsize=(12,5.5))
    ax.plot(x,[r["margin"] for r in targets],"-o",label="13 non-ROI Sink keys set to zero")
    ax.plot(x,controls.mean(1),color="gray",label="Three spatial/count-matched backgrounds: mean")
    ax.fill_between(x,controls.min(1),controls.max(1),color="gray",alpha=.2,label="Background min/max (not CI)")
    ax.axhline(targets[0]["full_margin"],color="black",ls="--",label="Same-machine native Full")
    ax.axhline(0,color="black",lw=.8);ax.axhline(.0390625,color="green",ls=":",label="Historical Physical K24 (different intervention)")
    ax.set(xlabel="Only this layer is intervened (L0-L31)",ylabel="logP(right) - logP(left)",
           title="15133706: hard-zero Sink attention at individual layers; visual mass retained")
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/"single_layer_margin.png",dpi=180);plt.close(fig)
    windows=[r for r in rows if r["target"]=="sink" and r["kind"]=="window"]
    fig,ax=plt.subplots(figsize=(9,5))
    for j,r in enumerate(windows):
        ctrl=[t["gain_vs_full"] for t in rows if t["scope"]==r["scope"] and t["target"]!="sink"]
        ax.bar(j-.16,r["gain_vs_full"],.32,color="#2277aa",label="Sink" if j==0 else None)
        ax.bar(j+.16,np.mean(ctrl),.32,color="gray",label="Matched background mean" if j==0 else None)
        ax.scatter(np.full(3,j+.16),ctrl,color="black",s=12)
    ax.axhline(0,color="black",lw=.8);ax.set(xticks=range(len(windows)),
        xticklabels=[f"L{r['first_layer']}-L{r['last_layer']}" for r in windows],ylabel="Gain vs native Full (nats)",
        title="15133706: cumulative windows; three random controls are not independent images")
    ax.legend();fig.tight_layout();fig.savefig(output/"window_gain_comparison.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["prepare","gpu","summarize"],required=True)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--resume-attempt")
    args=parser.parse_args();output=args.output_dir.resolve()
    if args.stage=="prepare":prepare(output)
    elif args.stage=="gpu":gpu(output,args.resume_attempt)
    else:validate(output);summarize(output)


if __name__=="__main__":main()
