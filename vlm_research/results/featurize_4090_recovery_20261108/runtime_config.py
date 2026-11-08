"""Machine paths and frozen adapters only; importing never starts GPU work."""
from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent
CODE_ROOT=O.parents[2]
DATA_ROOT=Path('/home/featurize/data/ate-recovery')
MODEL=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
REVISION='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
PYTHON=Path('/home/featurize/.venvs/ate-frozen/bin/python')

def verify_frozen_sources():
 for rel,h in json.loads((O/'frozen_sources.json').read_text()).items():
  assert hashlib.sha256((CODE_ROOT/rel).read_bytes()).hexdigest()==h,rel

def configure_scorers():
 from vlm_research.information_horizon_alignment import score
 score.AUTHOR=DATA_ROOT/'sources/Information-Horizon'
 return score

def load_adapter():
 verify_frozen_sources()
 from vlm_research.mme_seed_quality import engine as loader
 from vlm_research.information_horizon_alignment import engine
 loader.MODELS['llava']=MODEL
 configure_scorers()
 return engine.load('B')

# GQA: official_functions()/make_inputs(...,'C',...) from protocol_helpers.py.
# Other six benchmarks: retain frozen Paper-compatible sample prompt/preprocess.
# Physical engine: vlm_research.gqa_testdev_c3.engine.generate, label Native,
# C3-V2 or C3-V2+L11-L24. Each request needs its own fresh KV cache.
