"""Native / frozen Physical-D quality worker. One model, fresh per-request KV, no timing benchmark."""
from .common import *
from .scoring import anomalies
import argparse,gzip,sqlite3,traceback,gc,importlib.util,functools
import torch,transformers,numpy as np
from transformers import AutoModel,AutoTokenizer
from vlm_research.internvl3_svh_physical_d_pilot import engine as p
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as b
def manifest(bench):
    with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:return [json.loads(x) for x in f]
def environment(model,tok):
    pars=list(model.parameters())
    assert all(x.device.type=="cuda" and x.device.index==0 for x in pars)
    assert all(not x.is_floating_point() or x.dtype==torch.bfloat16 for x in pars)
    assert len(model.language_model.model.layers)==28 and model.num_image_token==256
    assert model.language_model.config._attn_implementation=="eager"
    return {"python":sys.version,"python_executable":sys.executable,"torch":torch.__version__,"torch_import":torch.__file__,"transformers":transformers.__version__,"transformers_import":transformers.__file__,"cuda_runtime":torch.version.cuda,"model_class_file":__import__("inspect").getfile(type(model)),"GPU_UUID":GPU,"model_revision":REV,"weights_all_gpu":True,"all_float_weights_bf16":True,"language_layers":28,"language_attention":"eager","visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used","--format=csv,noheader"],text=True).strip()}
@torch.no_grad()
def load_model():
    assert torch.__version__=="2.6.0+cu124" and transformers.__version__=="4.49.0"
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,device_map={"":"cuda:0"},use_safetensors=True,local_files_only=True).eval()
    model.img_context_token_id=tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model,tok
class Inputs:
    def __init__(self,model,tok):
        self.model=model;self.tok=tok;self.loader=b.official_preprocessor(MODEL);self.verified={}
    @functools.lru_cache(maxsize=8)
    def pixels(self,path,expected):
        path=Path(path)
        actual=sha(path)
        if actual!=expected:raise RuntimeError("IMAGE_SHA_MISMATCH "+str(path))
        self.verified[str(path)]=actual
        return self.loader(str(path),max_num=1).to(dtype=torch.bfloat16)
    def ready(self,r):
        request=request_text(r["benchmark"],r);c=self.model.conv_template.copy();c.system_message=self.model.system_message
        c.append_message(c.roles[0],request);c.append_message(c.roles[1],None)
        prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
        ids=self.tok(prompt,return_tensors="pt")["input_ids"]
        assert prompt==r["prompt"] and sha(prompt)==r["prompt_sha256"] and ids[0].tolist()==r["input_ids"]
        roles=b.role_layout(ids[0].tolist(),self.tok.convert_tokens_to_ids("<IMG_CONTEXT>"));assert roles==r["roles"]
        pix=self.pixels(r["image_path"],r["image_sha256"]);assert tuple(pix.shape)==(1,3,448,448)
        ready={"ids":ids.to("cuda:0"),"pixels":pix.to("cuda:0"),"roles":roles,"request":request,"prompt":prompt}
        binding={"input_ids_sha256":tensor_sha(ids),"pixels_bf16_sha256":tensor_sha(pix),"prompt_sha256":sha(prompt),"image_sha256":r["image_sha256"]}
        return ready,bind(binding),binding
@torch.inference_mode()
def smoke(model,tok,inputs,rows,pre):
    selected=[next(x for x in rows if x["question_id"]==q) for q in pre]
    selected=sorted(selected,key=lambda r:r["input_tokens"])
    selected=[selected[0],selected[len(selected)//2],selected[-1]]
    checks=[]
    for r in selected:
        ready,ib,binding=inputs.ready(r);cap=CAPS[r["benchmark"]]
        native,ns,nc=p.generate(model,tok,ready,"Native",max_tokens=cap,audit=True)
        noop,zs,zc=p.generate(model,tok,ready,"No-op",max_tokens=cap,audit=True)
        assert native["token_ids"]==noop["token_ids"] and len(ns)==len(zs) and all(torch.equal(x,y) for x,y in zip(ns,zs)),"NATIVE_NOOP_FAIL"
        with b.OfficialChatCapture(model) as capture:
            answer=model.chat(tok,ready["pixels"],ready["request"],dict(do_sample=False,num_beams=1,max_new_tokens=cap,min_new_tokens=0,logits_to_keep=1,use_cache=True,pad_token_id=tok.convert_tokens_to_ids(model.conv_template.sep.strip())))
        assert capture.ids==native["token_ids"] and answer.strip()==native["text"],"OFFICIAL_CHAT_TOKEN_FAIL"
        assert capture.input_ids[0].tolist()==r["input_ids"],"OFFICIAL_CHAT_INPUT_FAIL"
        assert len(capture.scores)==len(ns) and all(torch.equal(x.float(),y) for x,y in zip(ns,capture.scores)),"OFFICIAL_CHAT_LOGITS_FAIL"
        phys,ps,pc=p.generate(model,tok,ready,"Physical-D",max_tokens=cap,audit=True)
        assert not phys["empty"] and phys["independent_cache"]
        for k in phys["kv_layout"]:
            assert k["visual_kv"]==0
            assert (k["physical_length"]==0) if k["layer"] in (2,3,4,5) else (k["physical_length"]==r["input_tokens"]-256+phys["generated_tokens"]-1)
        assert phys["audit"]["hidden_exit"]==[{"before":r["input_tokens"],"after":r["input_tokens"]-256,"removed":256}]
        assert phys["audit"]["original_logical_positions"] and phys["audit"]["blocked_probability_max"]==0
        checks.append({"question_id":r["question_id"],"input_binding":binding,"native_noop_official_logits_strict_equal":True,"visual_tokens":256,"pixels_shape":[1,3,448,448],"native_tokens":native["token_ids"],"physical_tokens":phys["token_ids"],"physical_audit":phys["audit"],"native_kv":native["kv_layout"],"physical_kv":phys["kv_layout"],"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER"})
        print("SMOKE_PASS",r["benchmark"],r["question_id"],flush=True)
        del ns,zs,ps,nc,zc,pc,capture,ready;gc.collect()
    return {"status":"PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER","checks":checks,"frozen_physical_source":SOURCE}
def database(bench,ph,src):
    path=RUN/"checkpoints"/(bench+".sqlite");path.parent.mkdir(exist_ok=True)
    db=sqlite3.connect(path);db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS predictions(qid TEXT,method TEXT,record TEXT,PRIMARY KEY(qid,method))")
    for k,v in {"protocol_sha256":ph,"sources_sha256":src,"benchmark":bench}.items():
        old=db.execute("SELECT value FROM metadata WHERE key=?",(k,)).fetchone()
        if old:assert old[0]==v,"CHECKPOINT_BINDING_MISMATCH "+k
        else:db.execute("INSERT INTO metadata VALUES (?,?)",(k,v))
    db.commit();return db
@torch.inference_mode()
def request(model,tok,inputs,r,method,phase,ph,src,ready=None,ib=None,binding=None):
    if ready is None:ready,ib,binding=inputs.ready(r)
    finite=torch.ones((),dtype=torch.bool,device="cuda:0")
    def finite_check(module,args,out):
        nonlocal finite
        finite=finite & torch.isfinite(out.logits).all()
    hook=model.language_model.register_forward_hook(finite_check)
    try:
        result,_,_=p.generate(model,tok,ready,"Native" if method=="native" else "Physical-D",max_tokens=CAPS[r["benchmark"]],audit=False,fixed=False,timed=False)
        assert bool(finite),"NONFINITE_LOGITS"
    finally:hook.remove()
    raw=tok.decode(result["token_ids"],skip_special_tokens=False)
    plain=tok.decode(result["token_ids"],skip_special_tokens=True)
    flags=anomalies(r["benchmark"],result,r)
    return {"benchmark":r["benchmark"],"question_id":r["question_id"],"method":method,"phase":phase,"gold":r["gold"],"image_sha256":r["image_sha256"],"prompt_sha256":r["prompt_sha256"],"input_binding":ib,"input_binding_details":binding,"input_tokens":r["input_tokens"],"visual_tokens":256,"max_new_tokens":CAPS[r["benchmark"]],"protocol_sha256":ph,"source_manifest_sha256":src,"sample_sha256":r["sample_sha256"],"token_ids":result["token_ids"],"raw_decode_with_special_tokens":raw,"raw_decode":plain,"text":result["text"],"generated_tokens":result["generated_tokens"],"eos":result["eos"],"truncated":result["truncated"],"empty":result["empty"],"kv_layout":result["kv_layout"],"KV_live_MiB":result["KV_live_MiB"],"category":r["category"],"group":r["group"],"dataset_name":r.get("dataset_name",""),**flags}
def get(db,q,method):
    old=db.execute("SELECT record FROM predictions WHERE qid=? AND method=?",(q,method)).fetchone()
    return json.loads(old[0]) if old else None
def put(db,res):
    db.execute("INSERT INTO predictions VALUES (?,?,?)",(res["question_id"],res["method"],canonical(res)));db.commit()
def paired(model,tok,inputs,db,r,index,phase,ph,src):
    order=METHODS[index%2:]+METHODS[:index%2];results=[];ready=None
    for method in order:
        old=get(db,r["question_id"],method)
        if old:
            assert old["sample_sha256"]==r["sample_sha256"];results.append(old);continue
        if ready is None:ready,ib,binding=inputs.ready(r)
        res=request(model,tok,inputs,r,method,phase,ph,src,ready,ib,binding);put(db,res);results.append(res)
    assert results[0]["input_binding"]==results[1]["input_binding"]
    return results
def write_preflight(db,bench,pre):
    records=[];failures=[]
    for q in pre:
        pair=[get(db,q,m) for m in METHODS];assert all(pair)
        diff=pair[1]["generated_tokens"]-pair[0]["generated_tokens"]
        for x in pair:
            records.append({k:x.get(k) for k in ["benchmark","question_id","method","prompt_sha256","image_sha256","visual_tokens","input_tokens","generated_tokens","max_new_tokens","eos","truncated","text","parsed_answer","parse_failure","explanation","repetition","flags"]}|{"physical_minus_native_length":diff})
            if x["flags"]:failures.append({"question_id":q,"method":x["method"],"flags":x["flags"],"text":x["text"]})
    # A genuine single OCR long transcription is allowed; consecutive cap hits are not.
    for method in METHODS:
        for a,bx in zip(pre,pre[1:]):
            if get(db,a,method)["truncated"] and get(db,bx,method)["truncated"]:failures.append({"question_ids":[a,bx],"method":method,"flags":["CONSECUTIVE_CAP_HITS"]})
    csvwrite(OUT/bench/"prompt_preflight.csv",records)
    writej(OUT/bench/"preflight_validation.json",{"status":"PASS" if not failures else "PAUSED_OUTPUT_PREFLIGHT","samples":len(pre),"paired_generations":2*len(pre),"failures":failures,"accuracy_not_used_for_prompt_or_cap":True})
    return failures
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--benchmark",required=True,choices=BENCHES);args=parser.parse_args();bench=args.benchmark
    ph=sha(OUT/"protocol.json");src=sha(OUT/"source_manifest.json");sources=readj(OUT/"source_manifest.json")
    for rel,h in sources["runtime_source_sha256"].items():assert sha(RUN/"code"/rel)==h
    assert source_fingerprints()==readj(OUT/"protocol.json")["source_sha256"]
    assert readj(OUT/"cpu_tests.json")["status"]=="PASS"
    gpu_idle();rows=manifest(bench);meta=readj(OUT/"dataset_manifest.json")[bench]
    assert sha(RUN/"manifests"/(bench+".jsonl.gz"))==meta["manifest_sha256"]
    pre=meta["preflight_question_ids"];idx={r["question_id"]:r for r in rows};out=OUT/bench;out.mkdir(parents=True,exist_ok=True)
    db=None;model=None;inputs=None
    try:
        state(bench,"LOADING_MODEL",questions=len(rows));model,tok=load_model();writej(out/"environment.json",environment(model,tok));inputs=Inputs(model,tok)
        sm=smoke(model,tok,inputs,rows,pre);writej(out/"smoke_validation.json",sm)
        db=database(bench,ph,src);state(bench,"PREFLIGHT",samples=len(pre))
        for i,q in enumerate(pre):paired(model,tok,inputs,db,idx[q],i,"preflight",ph,src)
        failures=write_preflight(db,bench,pre)
        if failures:state(bench,"PAUSED_OUTPUT_PREFLIGHT",failures=len(failures),generations=db.execute("SELECT count(*) FROM predictions").fetchone()[0]);return
        writej(out/"preflight_gate.json",{"status":"PASS","unlock_full":True,"protocol_sha256":ph,"preflight_sha256":sha(out/"prompt_preflight.csv")})
        state(bench,"FULL_GENERATION",complete_questions=len(pre),target=len(rows))
        previous={m:False for m in METHODS}
        for i,r in enumerate(rows):
            pair=paired(model,tok,inputs,db,r,i,"formal",ph,src)
            bad=[x for x in pair if x["flags"]]
            for x in pair:
                if previous[x["method"]] and x["truncated"]:bad.append(x)
                previous[x["method"]]=x["truncated"]
            if bad:
                state(bench,"PAUSED_OUTPUT_FORMAL",question_id=r["question_id"],failures=bad,complete_rows=db.execute("SELECT count(*) FROM predictions").fetchone()[0]);return
            if (i+1)%25==0 or i+1==len(rows):
                state(bench,"FULL_GENERATION",complete_questions=i+1,target=len(rows),generation_rows=db.execute("SELECT count(*) FROM predictions").fetchone()[0])
                print("PROGRESS",bench,i+1,len(rows),flush=True)
        assert db.execute("SELECT count(*) FROM predictions").fetchone()[0]==2*len(rows)
        state(bench,"GENERATION_COMPLETE",complete_questions=len(rows),generation_rows=2*len(rows))
        writej(out/"image_consumption_integrity.json",{"status":"PASS","verified_images":inputs.verified,"checked_at_first_consumption":True,"CPU_pixels_cache_size":8})
    except Exception as e:
        err={"status":"FAILED_RUNTIME","error":str(e),"traceback":traceback.format_exc(),"GPU_allocated_MiB":torch.cuda.memory_allocated()/1048576 if torch.cuda.is_initialized() else None,"GPU_reserved_MiB":torch.cuda.memory_reserved()/1048576 if torch.cuda.is_initialized() else None}
        writej(out/"failure.json",err);state(bench,"FAILED_RUNTIME",**{k:v for k,v in err.items() if k!="status"});print(err["traceback"],flush=True)
    finally:
        if db is not None:db.execute("PRAGMA wal_checkpoint(TRUNCATE)");db.close()
        if inputs is not None:inputs.pixels.cache_clear()
        model=None;inputs=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        print("WORKER_EXIT",bench,flush=True)
if __name__=="__main__":main()
