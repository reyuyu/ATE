"""Frozen SparseVLM-prefix/SVH fusion and explicitly authorized task handoff."""
from pathlib import Path
import sys

# The official pin is isolated here, never installed into the shared environment.
_deps = Path(__file__).resolve().parents[2] / '.runtime-deps'
if _deps.is_dir():
    sys.path.insert(0, str(_deps))
