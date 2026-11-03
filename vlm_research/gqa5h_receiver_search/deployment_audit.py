"""Independently retokenize frozen native inputs and exercise the public API."""
import inspect
from transformers import AutoProcessor
from .common import *
from .deploy import build_input,load_selector

def main():
    started=time.monotonic();f=read(P/'winner_freeze.json');selector=load_selector(P);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);index=[]
    assert 'question_id' not in inspect.signature(build_input).parameters
    for split,ids in read(P/'split_manifest.json')['ids'].items():
        for q in ids:
            s=sample(q);sp=spec(q);chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
            actual=build_input(processor.tokenizer,s['question'],chat,s['input_mapping']['prompt_ids'],s['input_mapping']['visual_positions'],sp['R'])
            assert selector(actual)==selector(sp)
            # The frozen Winner has no parser dependency. Remove all POS/dep
            # annotations, ID-like fields and precomputed historical answers.
            for g in actual['groups']:g.pop('pos',None);g.pop('dep',None)
            assert selector(actual)==selector(sp)
            w=selector(actual);last=actual['groups'][-1];assert set(last['positions'])<=set(w) and len(last['positions'])==4 and last['words']==['ASSISTANT']
            pminus1=len(s['input_mapping']['prompt_ids'])-1;assert pminus1 not in w
            index.append(dict(split=split,question_id=q,retokenized_exact=True,parser_free=True,anchor_text='ASSISTANT',anchor_positions=last['positions'],anchor_token_ids=[s['input_mapping']['prompt_ids'][p] for p in last['positions']],Pminus1=pminus1,k=len(w),R=len(sp['R'])))
    assert not torch.cuda.is_initialized();atomic_json(P/'deployment_validation.json',dict(passed=True,cases=len(index),seconds=time.monotonic()-started,uses_only_native_text_and_token_offsets=True,no_Gold_or_model_outputs_read=True,parser_free=True,anchor='Original ASSISTANT complete group (4 subwords), never P-1 colon; no text inserted.',cases_index=index,utc=utc()));print(json.dumps(dict(passed=True,cases=len(index),seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
