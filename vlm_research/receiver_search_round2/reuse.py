"""Bind reused first-round checkpoints to their committed artifact index."""
import subprocess
from .common import *
_index=None
def validate_old_record(record):
    global _index
    path=Path(record.get('source',{}).get('path',''))
    if not path.is_relative_to(OLDP):return
    if _index is None:
        index=OLDP/'artifact_sha_index.csv';blob=subprocess.check_output(['git','show','a1c9d81db89f59142a28295f2f1eeaa2b93e8fe6:vlm_research/results/gqa5h_receiver_search/artifact_sha_index.csv'],cwd=ROOT.parent);assert blob==index.read_bytes();_index={r['path']:r['sha256'] for r in csv.DictReader(index.open())}
    assert sha(path)==_index[str(path.relative_to(OLDP.parents[1]))]
