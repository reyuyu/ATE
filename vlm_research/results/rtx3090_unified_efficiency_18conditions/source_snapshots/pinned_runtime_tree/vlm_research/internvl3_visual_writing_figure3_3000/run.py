"""Single-model InternVL smoke + 93,000 paired functional generations; hash-bound resume."""
from contextlib import ExitStack
import gc,json,os,sys,time,traceback,subprocess
from pathlib import Path
import torch
from .engine import *
from .prepare import ROOT,OUT,MODEL,REV,sha,save,command

def verify():
    p=json.loads((OUT/"protocol.json").read_text())
    assert sha(OUT/"sample_manifest.json")==p["sample_manifest_sha256"]
    assert sha(OUT/"cpu_input_bindings.json")==p["cpu_input_bindings_sha256"]
    assert sha(OUT/"source_manifest.json")==p["source_manifest_sha256"]
    assert sha(OUT/"model_manifest.json")==p["model_manifest_sha256"]
    for path,h in json.loads((OUT/"source_manifest.json").read_text()).items():assert sha(path)==h,path
    uuid=command("nvidia-smi","--query-gpu=uuid","--format=csv,noheader")
    assert uuid==p["gpu_uuid"] and torch.cuda.device_count()==1
    return p

def equal_scores(a,b):
    return len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))

@torch.inference_mode()
def smoke(model,tokenizer,loader,samples,bindings):
    indices=json.loads((OUT/"smoke_selection.json").read_text())["indices"];checks=[]
    for index in indices:
        s=samples[index];prepared=prepare(model,tokenizer,loader,s)
        assert prepared["binding"]["input_ids"]==bindings[index]["input_ids"]
        assert prepared["binding"]["roles"]==bindings[index]["roles"]
        assert prepared["prompt"]==bindings[index]["prompt"]
        native,native_scores,_=generate(model,tokenizer,prepared,native=True)
        with OfficialChatCapture(model) as official:
            text=model.chat(tokenizer,prepared["pixels"],prepared["request"],
                dict(do_sample=False,num_beams=1,max_new_tokens=32,logits_to_keep=1,
                     pad_token_id=tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())))
        assert official.input_ids[0].tolist()==bindings[index]["input_ids"]
        assert torch.equal(official.embeddings,prepared["inputs"]["inputs_embeds"].cpu())
        assert official.ids==native["token_ids"] and equal_scores(official.scores,native_scores)
        assert text.strip()==native["text"]
        noop,ns,nt=generate(model,tokenizer,prepared,noop=True,audit=True,trace=True)
        assert native["token_ids"]==noop["token_ids"] and equal_scores(native_scores,ns)
        assert len(native["token_ids"])>=2,"Need real incremental decode in smoke"
        aw,aw_scores,aw_trace=generate(model,tokenizer,prepared,audit=True,trace=True)
        assert aw_trace["prefix"]==nt["prefix"],"Image queries/causal prefix must stay native"
        variants=[]
        for c in ["No-Writer","Drop-L0","Drop-L13","Drop-L27","Keep-L0-L16"]:
            row,scores,tr=generate(model,tokenizer,prepared,c,audit=True,trace=True)
            assert tr["prefix"]==nt["prefix"],"Non-target image queries changed"
            assert row["mask_audit"]["blocked_probability_max"]==0
            assert row["mask_audit"]["real_decode"]>=1
            if c=="Drop-L27":
                assert row["token_ids"]==aw["token_ids"] and equal_scores(scores,aw_scores),\
                    "Structural last-layer negative control differs"
            variants.append(dict(condition=c,token_ids=row["token_ids"],text=row["text"],mask_audit=row["mask_audit"]))
        footprint=hooks_count(model)
        try:
            from transformers.cache_utils import DynamicCache
            with WriterMask(model,prepared["roles"],"No-Writer",DynamicCache()):
                raise RuntimeError("Intentional smoke cleanup fixture")
        except RuntimeError as e:assert str(e)=="Intentional smoke cleanup fixture"
        assert hooks_count(model)==footprint
        checks.append(dict(index=index,task=s["task"],question_id=s["question_id"],n=prepared["roles"]["n"],
            native_official_all_logits_torch_equal=True,native_noop_all_logits_torch_equal=True,
            official_visual_embeddings_equal=True,native_noop_tokens_equal=True,Drop_L27_all_logits_torch_equal=True,
            non_target_mask_exact=True,image_query_prefix_native_exact=True,
            native_norm_QKV_MLP_RoPE_residual=True,exception_hook_cleanup=True,variants=variants))
        save("smoke_partial.json",dict(passed=False,completed=len(checks),checks=checks))
        print("SMOKE_SAMPLE_PASS",index,s["question_id"],flush=True)
    report=dict(passed=True,samples=checks,protocol_sha256=sha(OUT/"protocol.json"),
        GPU_UUID=command("nvidia-smi","--query-gpu=uuid","--format=csv,noheader"),
        all_target_probabilities_zero=True,all_native_effective_logits_torch_equal=True,
        last_layer_negative_control_exact=True,exception_cleanup=True)
    save("smoke_summary.json",report);return report

@torch.no_grad()
def main():
    protocol=verify()
    assert not command("nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"),"GPU busy"
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from transformers import AutoModel,AutoTokenizer
    model=None;started=time.time()
    try:
        model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,
            device_map={"":"cuda:0"},use_safetensors=True).eval()
        assert len(model.language_model.model.layers)==28 and model.num_image_token==256
        assert model.language_model.config._attn_implementation=="eager"
        assert {p.dtype for p in model.parameters()}=={torch.bfloat16}
        assert {str(p.device) for p in model.parameters()}=={"cuda:0"}
        assert not any(x.attn.use_flash_attn for x in model.vision_model.encoder.layers)
        tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
        loader=official_preprocessor(MODEL)
        samples=json.loads((OUT/"sample_manifest.json").read_text())["samples"]
        bindings=json.loads((OUT/"cpu_input_bindings.json").read_text())
        assert len(samples)==len(bindings)==3000
        versions=[(p._version,p.data_ptr()) for p in model.parameters()];footprint=hooks_count(model)
        save("runtime_loaded.json",dict(pid=os.getpid(),loaded=True,GPU_UUID=protocol["gpu_uuid"],
            allocated_MiB=torch.cuda.memory_allocated()/1024**2,model_class=type(model).__name__,
            language_class=type(model.language_model).__name__,parameters=sum(p.numel() for p in model.parameters()),
            dtype="bfloat16",attention="eager",model_revision=REV))
        smoke(model,tokenizer,loader,samples,bindings)
        print("ALL_SMOKE_PASS",flush=True)
        protocol_hash=sha(OUT/"protocol.json");sample_hash=sha(OUT/"sample_manifest.json")
        cp=OUT/"checkpoints";cp.mkdir(exist_ok=True)
        total=0;new=0;reused=0;formal_start=time.time()
        for index,sample in enumerate(samples):
            key=sample["task"]+"__"+sample["question_id"];p=cp/(key+".json")
            record=json.loads(p.read_text()) if p.exists() else dict(
                index=index,task=sample["task"],question_id=sample["question_id"],image_sha256=sample["image_sha256"],
                gold=sample["answer"],sample_sha256=digest(sample),protocol_sha256=protocol_hash,
                manifest_sha256=sample_hash,GPU_UUID=protocol["gpu_uuid"],complete=False,conditions={})
            assert record["index"]==index and record["sample_sha256"]==digest(sample)
            assert record["protocol_sha256"]==protocol_hash and record["manifest_sha256"]==sample_hash
            assert record["GPU_UUID"]==protocol["gpu_uuid"] and set(record["conditions"])<=set(CONDITIONS)
            if record["complete"]:
                assert set(record["conditions"])==set(CONDITIONS)
                assert record["conditions"]["Drop-L27"]["all_step_logits_sha256"]==record["conditions"]["All-Writer"]["all_step_logits_sha256"]
                total+=31;reused+=31;continue
            prepared=prepare(model,tokenizer,loader,sample)
            assert prepared["binding"]["input_ids"]==bindings[index]["input_ids"]
            assert prepared["binding"]["roles"]==bindings[index]["roles"] and prepared["prompt"]==bindings[index]["prompt"]
            if "input_binding" in record:assert record["input_binding"]==prepared["input_binding"]
            record.update(input_binding=prepared["input_binding"],input=prepared["binding"])
            for condition in CONDITIONS:
                if condition in record["conditions"]:
                    total+=1;reused+=1;continue
                tick=time.time()
                row,scores,_=generate(model,tokenizer,prepared,condition)
                row.update(correct=int(row["prediction"]==sample["answer"].lower()),seconds=time.time()-tick)
                if condition=="Drop-L27":
                    baseline=record["conditions"]["All-Writer"]
                    assert row["token_ids"]==baseline["token_ids"]
                    assert row["all_step_logits_sha256"]==baseline["all_step_logits_sha256"]
                baseline=row if condition=="All-Writer" else record["conditions"]["All-Writer"]
                row.update(first_token_flip=int(row["token_ids"][0]!=baseline["token_ids"][0]),
                    correct_to_wrong=int(baseline["correct"] and not row["correct"]),
                    wrong_to_correct=int(not baseline["correct"] and row["correct"]))
                record["conditions"][condition]=row
                record["complete"]=set(record["conditions"])==set(CONDITIONS)
                tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(record,ensure_ascii=False,separators=(",",":")));tmp.replace(p)
                total+=1;new+=1
                del scores
            del prepared,record
            if (index+1)%5==0 or index==0:
                assert command("nvidia-smi","--query-gpu=uuid","--format=csv,noheader")==protocol["gpu_uuid"]
                assert versions==[(p._version,p.data_ptr()) for p in model.parameters()]
                assert hooks_count(model)==footprint
                progress=dict(status="RUNNING",samples_complete=index+1,requests_complete=total,
                    requests_target=93000,new=new,reused=reused,formal_seconds=time.time()-formal_start,
                    GPU_UUID=protocol["gpu_uuid"],last_question_id=sample["question_id"])
                save("progress.json",progress);print("PROGRESS",json.dumps(progress),flush=True)
        assert total==93000 and versions==[(p._version,p.data_ptr()) for p in model.parameters()]
        assert hooks_count(model)==footprint
        save("generation_complete.json",dict(status="COMPLETE",requests=total,questions=3000,conditions=31,
            GPU_UUID=protocol["gpu_uuid"],protocol_sha256=protocol_hash,weights_unchanged=True,hooks_clean=True,
            no_nan_inf_oom=True,new=new,reused=reused,seconds=time.time()-formal_start,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2))
    except BaseException:
        save("failure.json",dict(status="STOPPED",traceback=traceback.format_exc(),time=time.time(),
            protocol_sha256=sha(OUT/"protocol.json"),GPU_UUID=protocol["gpu_uuid"],pid=os.getpid()))
        raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()
        save("gpu_release.json",dict(worker_pid=os.getpid(),worker_finished=True,
            allocated_MiB=torch.cuda.memory_allocated()/1024**2,elapsed_seconds=time.time()-started))

if __name__=="__main__":main()
