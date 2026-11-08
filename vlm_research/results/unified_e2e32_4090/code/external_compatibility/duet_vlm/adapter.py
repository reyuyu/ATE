from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'e2e32_compatibility'))
from e2e32_official_compat import OfficialCompat
def controller(adapter, layout, trace=False):
    return OfficialCompat(adapter, layout, 'DUET', trace=trace)
