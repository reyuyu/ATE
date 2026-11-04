"""CPU-process-only caching of immutable input JSON; holdout gates stay live."""
from functools import lru_cache
from pathlib import Path
from . import common
def enable():
    original=common.read
    cached=lru_cache(maxsize=4096)(original)
    def read(path):
        p=Path(path)
        if p in [common.P/'split_manifest.json',common.P/'clock.json',common.P/'protocol.json'] or p.parent in [common.P/'input_specs',common.PRIVATE/'samples',common.OLDP/'input_specs',common.OLDPRIVATE/'samples']:
            return cached(p)
        return original(p)
    common.read=read
