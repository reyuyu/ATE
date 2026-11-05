from ..qwen_keep1000.common import *
BASE='dd464a691c30a0e047ddfb59177ec6a8ad2a40be'
PREVIOUS=ROOT/'vlm_research/results/qwen3vl_keep_l0_21_gqa1000_reproduction'
OUT=ROOT/'vlm_research/results/qwen3vl_keep_l0_24_26_gqa1000'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_keep_l0_24_26_gqa1000')
CONDITIONS={'Keep-L0-L24':24,'Keep-L0-L26':26}

def check_sources():
 f=read(OUT/'source_manifest.json')
 for p,h in f['files'].items():assert sha(ROOT/p)==h,p
 return digest(f)

def reference_path(s,c):
 if c=='All-Writer':return PREVIOUS/f'checkpoints/rank{s["rank"]}/{s["question_id"]}__All-Writer.json'
 return baseline_path(s,'D' if c=='Keep-L0-L21' else c)
