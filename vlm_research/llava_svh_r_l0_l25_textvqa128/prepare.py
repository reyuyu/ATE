from common import *
import torch,platform,subprocess
from cpu_tests import test
def main():
 parent=Path('/home/featurize/ate-runs/llava-svh-r-textvqa128-3090');pout=parent/'results';samples=read(OUT/'sample_manifest.json');bindings=read(OUT/'input_bindings.json')
 assert read(pout/'validation.json')['status']=='PASS' and len(samples['Main'])==128 and len(samples['Smoke'])==3
 assert sha(OUT/'sample_manifest.json')==sha(pout/'sample_manifest.json') and sha(OUT/'input_bindings.json')==sha(pout/'input_bindings.json')
 assert not torch.cuda.is_initialized()
 prepared=RUN/'prepared'
 if not prepared.exists():prepared.symlink_to(parent/'prepared',target_is_directory=True)
 for s in samples['Main']+samples['Smoke']:
  qid=str(s['question_id']);assert sha(s['image_path'])==s['image_sha256'];b=bindings[qid];assert digest(b)==s['input_binding']
  inputs=torch.load(prepared/(qid+'.pt'),map_location='cpu',weights_only=True);assert set(inputs)=={'input_ids','attention_mask','pixel_values'}
  for k,v in inputs.items():assert th(v)==b['tensors'][k]['sha256']
 old=sqlite3.connect('file:'+str(parent/'predictions.sqlite')+'?mode=ro',uri=True);assert old.execute('pragma integrity_check').fetchone()[0]=='ok'
 count=0
 for qid,label,h,raw in old.execute('select qid,method,checksum,record from predictions'):
  r=json.loads(raw);assert digest(r)==h and r['input_binding']==next(x['input_binding'] for x in samples['Main'] if str(x['question_id'])==qid)
  assert r['GPU_UUID']==GPU and r['protocol_sha256']==sha(pout/'protocol.json') and r['source_manifest_sha256']==sha(pout/'source_manifest.json');count+=1
 assert count==256;old.close()
 model_receipt=read(MODEL/'.ate_revision.json');assert model_receipt['revision']==REVISION and model_receipt['status']=='VERIFIED'
 for name,x in model_receipt['files'].items():assert (MODEL/name).stat().st_size==x['bytes']
 write(OUT/'cpu_tests.json',test())
 files={str(p.relative_to(CODE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(CODE.rglob('*.py'))}
 write(OUT/'source_manifest.json',dict(frozen_algorithm_commit='5ee2a0dbe78df336229b55fed14a4a6d0d880bed',source_parent_commit='177c631521d5c9d0436768e11e529ec492c2cbc8',files=files,adapter_change='byte-identical TerminalMiddleWriter; configure cutoff26 and KEEP L0,L1,L6..L25; old cutoff25 strict CPU regression',new_KEEP=list(KEEP),historical_v2_scope='unchanged normal EOS valid numeric gate +9step structural smoke'))
 p=read(pout/'protocol.json');p.update(status='FROZEN_L0_L25_BEFORE_GPU',model_loads=1,formal_conditions=['L0_L25'],formal_requests=128,requested_reaccess_layers=list(range(26)),effective_reaccess_layers=list(KEEP),skipped_unavailable_layers=[2,3,4,5],reused_conditions={'Native':128,'SVH-R_L11_L24':128},source_parent_commit='177c631521d5c9d0436768e11e529ec492c2cbc8',Native_reuse_calibration_ids=[str(x['question_id']) for x in (sorted(samples['Main'],key=lambda s:(s['input_tokens'],str(s['question_id'])))[0],sorted(samples['Main'],key=lambda s:(s['input_tokens'],str(s['question_id'])))[64],sorted(samples['Main'],key=lambda s:(s['input_tokens'],str(s['question_id'])))[-1])],source_manifest_sha256=sha(OUT/'source_manifest.json'),GitHub_branch='codex/llava-svh-r-l0-l25-textvqa128-3090',SVH_R=dict(answer_keep_layers=list(KEEP),visual_Hidden_exit_before_layer=26,physical_exit='beforeL26, original visual Hidden removed; later nonvisual logical positions retained',R_to_image_forbidden_from_layer=20,answer_attention_physical_skip=[2,3,4,5],cache='image KV only L0,L1,L6..L25; skip2..5 no persistent KV; original logical positions'),scope='Only user-requested L0..L25 quality supplement on same frozen128; no speed/other windows/quality benchmarks; Native+L11..24 reused from exact same GPU and input tensors; same-set exploration NOT independent confirmation')
 for k in ('initial_attempt','scope_correction'):p.pop(k,None)
 write(OUT/'protocol.json',p);write(OUT/'input_reuse_validation.json',dict(status='PASS',parent_commit='177c631521d5c9d0436768e11e529ec492c2cbc8',parent_records=256,exact_sample_and_input_manifest_SHA_equal=True,all131_images_and_model_tensors_hash_verified=True,Native_and_prior_SVH_R_same_GPU=True,model_download_receipt_reused=True,CUDA_initialized=torch.cuda.is_initialized(),new_formal=128,reused=256))
 print(json.dumps(dict(status='CPU_AND_SOURCE_FROZEN',CPU=read(OUT/'cpu_tests.json'),source_files=len(files),protocol_sha256=sha(OUT/'protocol.json'))))
if __name__=='__main__':main()
