"""Recreate the frozen Winner positions from input text/group metadata, CPU only."""
import argparse,importlib.util
from .common import *
def main(output):
    rules=read(P/'selection_rules.json');freeze=read(P/'winner_freeze.json');assert sha(P/'selection_rules.json')==freeze['selection_rules_sha256'];source=ROOT/rules['source_path'];assert sha(source)==rules['source_sha256']
    loader=importlib.util.spec_from_file_location('frozen_winner_for_reproduction',source);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);split=read(P/'split_manifest.json');cases=[]
    for q in split['ids']['FINAL']:
        s=spec(q);w=module.select({k:s[k] for k in rules['input_keys']},rules['config']);cases.append(dict(question_id=q,image_sha256=split['metadata'][q]['image_sha256'],writers=w,k=len(w),R=len(s['R']),R_ratio=len(w)/len(s['R']),input_hashes=s['input_hashes'],input_spec_sha256=sha(P/'input_specs'/f'{q}.json')))
    payload=dict(candidate_id=rules['candidate_id'],source_sha256=sha(source),config_sha256=digest(rules['config']),cases=cases,input_only=True);assert payload==read(P/'final_writer_manifest.json');atomic_json(Path(output),payload);assert sha(output)==freeze['final_writer_manifest_sha256'];assert not torch.cuda.is_initialized();print(json.dumps(dict(passed=True,questions=len(cases),token_manifest_sha256=sha(output),Gold_or_model_outputs_read=False,GPU_used=False)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
