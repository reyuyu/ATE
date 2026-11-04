"""Adapter for the historically validated functional mask engine.

This changes only the in-process policy table and output guard. Historical
source files, checkpoints and result folders are never modified. Each GPU run
must use its own process and pass fresh Native/No-op and baseline smoke gates.
"""
from .policy import LAYER_SETS,guard

def bind():
 from textvqa_reaccess import engine as impl
 impl.DESIGN={'conditions':LAYER_SETS}
 impl.CONDITIONS=list(LAYER_SETS)
 impl.guard=guard
 return impl
