from common import *
from run import *
@torch.inference_mode()
def diagnose():
 gpu_check();m,p,e=load_model(str(MODEL));a=SimpleNamespace(kind='llava',model=m,processor=p,layers=m.language_model.model.layers,inputs=None)
 dest=OUT/'smoke_diagnostic_v1';dest.mkdir(exist_ok=True);allrows=[];full=[];shapes=[]
 try:
  for s in read(OUT/'sample_manifest.json')['Smoke']:
   l=inputs_for(a,s);qid=str(s['question_id']);outdata={};handles=[]
   def run_capture(method):
    result={};handles=[];state={'step':0}
    for i,layer in enumerate(a.layers):
     def hook(module,args,output,i=i):
      h=output[0];key=(state['step'],i);idx=list(l.text) if h.shape[1]==l.length else list(range(h.shape[1]));result[key]=h[:,idx,:].detach().float().cpu()
      if i==31:state['step']+=1
     handles.append(layer.register_forward_hook(hook))
    try:r,logits,_,trace=generate(a,l,method,9,True,True,True)
    finally:
     for h in handles:h.remove()
    return r,logits,result,trace
   d,dl,dh,dt=run_capture('Dense');v,vl,vh,vt=run_capture('SVH-R')
   for step in range(9):
    row=dict(qid=qid,type='logits',step=step,layer=-1,allclose_gate=bool(torch.allclose(dl[step],vl[step],atol=.10,rtol=.003)),top1_equal=int(dl[step].argmax())==int(vl[step].argmax()),**metrics(dl[step],vl[step]));allrows.append(row)
    for i in range(32):allrows.append(dict(qid=qid,type='text_hidden',step=step,layer=i,**metrics(dh[step,i],vh[step,i])))
   dn,_,_,_=generate(a,l,'Dense',32,False);vn,_,_,_=generate(a,l,'SVH-R',32,False)
   full.append(dict(qid=qid,full_tokens_equal=dn['token_ids']==vn['token_ids'],Dense=dn['text'],Physical=vn['text'],Dense_token_ids=dn['token_ids'],Physical_token_ids=vn['token_ids'],fixed9_tokens_equal=d['token_ids']==v['token_ids']))
   shapes+= [dict(qid=qid,**x) for x in vt['operators']]
   csvwrite(dest/'per_step_layer.csv',allrows);csvwrite(dest/'operators.csv',shapes);write(dest/'complete_tokens.json',full)
  bystep=[]
  for s in read(OUT/'sample_manifest.json')['Smoke']:
   qid=str(s['question_id'])
   for step in range(9):
    rr=[x for x in allrows if x['qid']==qid and x['step']==step and x['type']=='text_hidden']
    nz=[x for x in rr if x['MaxAbs']>0]
    bystep.append(dict(qid=qid,step=step,first_nonzero_layer=nz[0]['layer'] if nz else None,first_MaxAbs=nz[0]['MaxAbs'] if nz else 0,max_hidden_absolute=max(x['MaxAbs'] for x in rr)))
  csvwrite(dest/'first_divergence.csv',bystep)
  write(dest/'summary.json',dict(status='STRUCTURAL_PASS_NUMERICAL_GATE_FAILED',full_tokens=full,all_dense_physical_complete_tokens_equal=all(x['full_tokens_equal'] for x in full),logit_max=max(x['MaxAbs'] for x in allrows if x['type']=='logits'),meanabs_max=max(x['MeanAbs'] for x in allrows if x['type']=='logits'),failed_gate_steps=[x for x in allrows if x['type']=='logits' and not x['allclose_gate']],first_divergences=bystep,algorithm_unchanged=True,no_formal_quality_executed=True))
  print(json.dumps(read(dest/'summary.json'),ensure_ascii=False))
 finally:
  a.model=None;a.layers=None;a.inputs=None;del m,a;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
if __name__=='__main__':diagnose()
