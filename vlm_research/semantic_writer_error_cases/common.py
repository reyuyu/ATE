import csv,hashlib,json,os,subprocess
from pathlib import Path
import torch
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,read,sha256,digest
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz
from semantic_visual_writers.groups import flatten,subsets,intact
from relay_source_causality.common import hardware

BASE='01c82b1d39360fc73bca3d10d12886672ffee370'
HIST=ROOT/'results/gqa3000_token_visual_writing'
OUT=ROOT/'results/semantic_writer_error_cases'
RAW=Path('/data/lk/ATE-resources/experiments/semantic_writer_error_cases')
SEED='20261103-semantic-error-cases-v1'
RANDOM_SEEDS=[20261103,20261107,20261111]

def tie(*args):return hashlib.sha256(('|'.join(map(str,(SEED,)+args))).encode()).hexdigest()
def csvwrite(path,rows):
    rows=list(rows);assert rows;path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists()
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gitsha(path):
    rel=str(Path(path).relative_to(ROOT.parent))
    return hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT.parent)).hexdigest()
def history_file(name):
    p=HIST/name;assert sha256(p)==gitsha(p),name
    return read(p)
def selected():return read(OUT/'case_manifest.json')['cases']
def bind():return sha256(OUT/'candidate_plan.json')
def active():assert not (OUT/'STOP.json').exists(),'Preserve failure; do not continue GPU work'
def source_hashes():
    paths=list((ROOT/'semantic_writer_error_cases').glob('*.py'))
    paths += [ROOT/'gqa3000_token_visual_writing/engine.py',ROOT/'semantic_visual_writers/engine.py',ROOT/'vlm_common.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}

RULES=dict(scope='12 harm cases, exactly two per frozen task, plus four rescue counterexamples. Post-hoc selected cases, not a representative sample.',
    mandatory_question_id='001020372',confirmatory_preference='Prefer the exact frozen 2640 image-disjoint subset; retain mandatory case even if it is outside.',
    harm_selection='Task order frozen. First include mandatory case. For each remaining slot prefer confirmatory, then maximize novel error type, then novel omitted token tags, then novel selected-fraction bin, then SHA256 tie. No repeated image across cases.',
    rescue_selection='Four correct-in-Semantic/wrong-in-All cases, prefer confirmatory, novel task then error type then omitted tags, then SHA tie; distinct images.',
    error_types='yes_no_reversal, numerical_answer, color_choice, relation_location, categorical_other. Defined from existing answer strings/question only.',
    single_candidate_rule='At most six: reserve omitted final Question group, first post-Question lexical template group, final R template group; fill remaining with omitted Question groups by frozen linguistic-tag priority, preferring distinct decoded words. No answer/result-dependent addition.',
    tag_priority=['negation','quantity','comparison','logical_connector','relation_connector','interrogative','auxiliary_function','article','punctuation','other'],
    group_tests=['Semantic plus each of at most six single whole groups','Semantic plus all omitted Question positions','Semantic plus all omitted instruction/template R positions'],
    controls='For each candidate, one matched added-token whole-group subset of interior instruction groups, excluding candidate, endpoints and newline; deterministic first reachable subset. These are positional controls, not proven causally irrelevant tokens.',
    replacement='After a successful single-group addition ONLY, run its prelocked equal-token replacement. Prefer one selected whole group of identical size, otherwise exact whole-group subset; SHA priority independent of outcomes. Failure tests this removal choice, not every equal-budget configuration.',
    limits='No more than six single additions + six positional controls + two grouped additions + six conditional replacements per harm case; maximum240 new greedy generations, excluding at most three fixed smoke cases. No Gold NLL rerun or optimization.',
    interpretation='Exploratory case-level interventions only. A recovered answer does not prove a unique storage token or population-level repair rate.')
