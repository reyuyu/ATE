"""CPU-only frozen cohort/prompt/mask verification and protocol creation."""
import importlib.util,json,hashlib,sys,platform,subprocess
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import torch,transformers
from transformers import AutoTokenizer
from .engine import *

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"vlm_research/results/internvl3_visual_writing_figure3_3000"
MODEL=Path("/home/featurize/ate-local-models/InternVL3-8B")
REV="853e3a797a661694b1b8ece0cb72dc2b23e3dac9"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n");tmp.replace(p)

def command(*args):return subprocess.check_output(args,text=True).strip()

def cpu_tests():
    failures=[];n=275;ids=[1]*n;ids[5:261]=[77]*256
    r=role_layout(ids,77);assert r["R"]==list(range(261,274)) and r["A_prefill"]==[274]
    base=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16)
    base=base.masked_fill(torch.ones((n,n),dtype=torch.bool).tril(),0)
    tests=[]
    for condition in CONDITIONS:
        allow=set(writers(condition))
        for layer in range(28):
            changed=edit_mask(base,r,True,layer not in allow)
            selected=torch.zeros_like(base,dtype=torch.bool);selected[:,:,274,5:261]=True
            if layer not in allow:selected[:,:,261:274,5:261]=True
            assert torch.equal(base!=changed,selected)
            assert torch.isneginf(changed[selected]).all()
            assert (changed==0).any(-1).all()
            assert torch.equal(changed[:,:,:261],base[:,:,:261])
        tests.append(condition)
    assert len(CONDITIONS)==31 and len(set(CONDITIONS))==31
    assert writers("Keep-L0-L16")==list(range(17)) and writers("Drop-L27")==list(range(27))
    decode=torch.zeros((1,1,1,n+3),dtype=torch.bfloat16)
    changed=edit_mask(decode,r,False,False)
    assert torch.isneginf(changed[0,0,0,5:261]).all() and int((changed!=decode).sum())==256
    for invalid in [[1]*300,[77]*255,[77]*257]:
        try:role_layout(invalid,77)
        except AssertionError:pass
        else:raise AssertionError("Invalid image-token count accepted")
    return dict(passed=True,conditions=tests,layer_mask_cases=31*28,decode_test=True,
                original_mask_unchanged=True,non_target_exact=True,all_queries_have_legal_key=True)

def main():
    assert transformers.__version__=="4.49.0" and torch.__version__=="2.6.0+cu124"
    assert not (OUT/"protocol.json").exists(),"Never overwrite a frozen protocol"
    assert json.loads((OUT/"model_verification.json").read_text())["status"]=="PASS"
    assert json.loads((OUT/"data_audit.json").read_text())["status"]=="PASS"
    history=json.loads((OUT/"historical_sample_audit.json").read_text())
    assert all(x["same_order"] and x["same_set"] and x["n"]==3000 for x in history.values())
    tests=cpu_tests();save("cpu_tests.json",tests)
    h=OUT/"references/sparse/vlm_research/results/qwen_internvl_sparse_visual_3000/sample_manifest.json"
    samples=json.loads(h.read_text())["samples"]
    assert len(samples)==3000 and len({s["image_sha256"] for s in samples})==2409
    assert Counter(x["task"] for x in samples)==Counter({t:500 for t in TASKS})
    for s in samples:s["actual_image_path"]="/home/featurize/data/internvl3-visual-writing-3000/images/image_"+s["image_id"]+".jpg"
    save("sample_manifest.json",dict(samples=samples,questions=3000,image_clusters=2409,
        source_commit="3132d923459a2ab80f38fb5e4108d6c27d3b9016",source_sha256=sha(h),selection="Exact historical order; no resampling"))
    spec=importlib.util.spec_from_file_location("pinned_internvl_conversation",MODEL/"conversation.py")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=json.loads((MODEL/"config.json").read_text());conv=module.get_conv_template(config["template"])
    dummy=SimpleNamespace(conv_template=conv,system_message=conv.system_message)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
    fast=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True)
    bindings=[]
    for sample in samples:
        request,prompt,ids,roles=prepare_cpu(dummy,tokenizer,sample)
        assert fast(prompt)["input_ids"]==ids[0].tolist()
        bindings.append(dict(task=sample["task"],question_id=sample["question_id"],prompt=prompt,
            input_ids=ids[0].tolist(),roles=roles,image_sha256=sample["image_sha256"]))
    save("cpu_input_bindings.json",bindings)
    ordered=sorted(range(3000),key=lambda i:(bindings[i]["roles"]["n"],i))
    smoke_indices=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    assert len(set(smoke_indices))==3
    save("smoke_selection.json",dict(rule="shortest, median-length, longest original prompt; ties historical order",
         indices=smoke_indices,samples=[dict(index=i,task=samples[i]["task"],question_id=samples[i]["question_id"],
                tokens=bindings[i]["roles"]["n"]) for i in smoke_indices]))
    env=dict(executable=sys.executable,python=sys.version,torch=torch.__version__,torch_path=torch.__file__,
        transformers=transformers.__version__,transformers_path=transformers.__file__,
        cuda_runtime=torch.version.cuda,platform=platform.platform(),
        gpu=command("nvidia-smi","--query-gpu=uuid,name,memory.total,driver_version","--format=csv,noheader"),
        gpu_snapshot=command("nvidia-smi"),
        compute_processes=command("nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"))
    assert not env["compute_processes"],"GPU busy; do not terminate anyone"
    assert "RTX 3090" in env["gpu"] and len(env["gpu"].splitlines())==1
    save("environment.json",env)
    model_manifest=json.loads((OUT/"model_verification.json").read_text())
    save("model_manifest.json",model_manifest)
    import transformers.models.qwen2.modeling_qwen2 as q2
    import transformers.generation.utils as gen
    sources=[Path(__file__),Path(__file__).with_name("engine.py"),Path(__file__).with_name("run.py"),
             Path(q2.__file__),Path(gen.__file__)]+list(MODEL.glob("*.py"))
    source_manifest={str(p):sha(p) for p in sources}
    save("source_manifest.json",source_manifest)
    protocol=dict(experiment="InternVL3-8B visual writing Figure 3",revision=REV,model="OpenGVLab/InternVL3-8B",
        model_path=str(MODEL),gpu_uuid=env["gpu"].split(",")[0].strip(),language_layers=28,
        dtype="bfloat16",attention="eager",torch="2.6.0+cu124",transformers="4.49.0",
        batch_size=1,greedy=True,max_new_tokens=32,logits_to_keep=1,use_cache=True,
        max_num=1,actual_visual_tokens=256,normal_eos=True,
        conditions={c:dict(writer_layers=writers(c)) for c in CONDITIONS},formal_requests=93000,
        frozen_deep_boundary=17,roles="I actual 256 IMG_CONTEXT; R after last I and before original P-1; A original P-1 and every decode",
        all_conditions_A_image_off=True,all_writer_is_not_native_full=True,
        mechanism="Native Qwen2 eager attention pre-hook: additive -inf only on selected causal-legal R/A -> original I edges",
        native_components="Unchanged Norm,QKV,RoPE,MLP,residual,vision/projector,positions,real independent DynamicCache",
        native_projection="Explicit logits_to_keep=1; torch.equal for native/chat/no-op and final-layer negative control",
        scoring='Frozen official gqa_prediction: decoded.strip().rstrip(".").lower() == gold.lower()',
        scoring_reference_sha256=sha(OUT/"references/qwen/vlm_research/results/qwen3vl_visual_writing_necessity_3000/official_gqa_prediction.json"),
        sample_manifest_sha256=sha(OUT/"sample_manifest.json"),cpu_input_bindings_sha256=sha(OUT/"cpu_input_bindings.json"),
        source_manifest_sha256=sha(OUT/"source_manifest.json"),model_manifest_sha256=sha(OUT/"model_manifest.json"),
        bootstrap=dict(repetitions=20000,seed=20261110,cluster="Image SHA256",estimator="question-weighted paired whole-image cluster resampling",
                       intervals="ordinary 95%; Bonferroni-28 equal-tailed for 28 layer contrasts; preplanned deep joint ordinary 95%"),
        preprocessing="Pinned official README build_transform/dynamic_preprocess/load_image AST; max_num=1, 448x448 RGB, ImageNet normalization",
        historical_only_text=dict(accuracy_reference_percent=90.33333333333333,semantics_equivalent=True,reused_generations=False,
                                   evidence="3132d923:visual_pathways/sparse_accuracy.py Only Text masks P-1 and all decode, other prefill native"),
        request_order="Frozen historical sample order, each 31 fixed conditions; independent cache per request",
        no_training=True,no_physical_pruning=True,no_performance_claim=True,no_push=True)
    save("protocol.json",protocol)
    save("preflight_complete.json",dict(passed=True,protocol_sha256=sha(OUT/"protocol.json"),cpu_tests=True,
        sample_strict_match=True,all_2409_images_SHA_match=True,model_full_content_verified=True,smoke_indices=smoke_indices))
    print("CPU_PREFLIGHT_PASS",smoke_indices,"protocol",sha(OUT/"protocol.json"),flush=True)

if __name__=="__main__":main()
